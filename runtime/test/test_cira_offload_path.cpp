// End-to-end test for the CIRA offload dispatch path:
//   cache-resident wait  (cira_mwait.h)
//   CXL Type-2 control window + doorbell (cira_mmio.h, cira_cxl_job.h)
//
// The window is backed by anonymous shared memory (CIRA_CXL_MMIO_EMULATE), so
// the full submit / doorbell / completion protocol runs without an FPGA. A
// helper thread plays the role of the device firmware: it polls the doorbell,
// executes the call job, and publishes the completion line the host awaits.

#include "CiraRuntime.h"
#include "cira_cxl_job.h"
#include "cira_mmio.h"
#include "cira_mwait.h"
#include "cira_shared_arena.h"

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace {

int g_failures = 0;

void check(bool cond, const char *msg) {
    std::printf("  [%s] %s\n", cond ? " ok " : "FAIL", msg);
    if (!cond)
        ++g_failures;
}

// --- 1. wait backend --------------------------------------------------------

void test_wait_backend() {
    std::printf("wait backend\n");
    cira_wait_backend_t backend = cira_wait_backend();
    std::printf("  detected: %s\n", cira_wait_backend_name(backend));

    // Already-satisfied predicate must return immediately.
    volatile uint32_t flag = CIRA_CXL_COMPLETION_MAGIC;
    check(cira_wait_u32(&flag, CIRA_CXL_COMPLETION_MAGIC, 0), "wait returns immediately when already signalled");

    // A timeout on an unsignalled flag must actually expire.
    volatile uint32_t never = 0;
    check(!cira_wait_u32(&never, CIRA_CXL_COMPLETION_MAGIC, 5 * 1000 * 1000), "wait times out on an unsignalled flag");

    // Cross-thread wake: this is the paper's device-to-host completion.
    static volatile uint32_t signalled = 0;
    std::thread writer([] {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        __atomic_store_n(&signalled, CIRA_CXL_COMPLETION_MAGIC, __ATOMIC_RELEASE);
    });
    check(cira_wait_u32(&signalled, CIRA_CXL_COMPLETION_MAGIC, 2ull * 1000 * 1000 * 1000),
          "wait wakes on a store from another agent");
    writer.join();

    // Forcing the portable backend must still work everywhere.
    cira_wait_set_backend(CIRA_WAIT_SPIN);
    volatile uint32_t done = CIRA_CXL_COMPLETION_MAGIC;
    check(cira_wait_u32(&done, CIRA_CXL_COMPLETION_MAGIC, 0), "spin backend is always available");
    cira_wait_set_backend(backend);
}

// --- 2. MMIO window ---------------------------------------------------------

cira_mmio_window_t *open_emulated_window() {
    cira_mmio_config_t cfg = {};
    cfg.emulate = true;
    cfg.size = CIRA_CXL_CONTROL_BYTES;
    cira_mmio_window_t *w = nullptr;
    if (cira_mmio_open(&w, &cfg) != CIRA_MMIO_OK)
        return nullptr;
    return w;
}

void test_mmio_registers(cira_mmio_window_t *w) {
    std::printf("mmio register access\n");
    check(cira_mmio_base(w) != nullptr, "window is mapped");
    check(cira_mmio_size(w) == CIRA_CXL_CONTROL_BYTES, "window size honoured");
    check(cira_mmio_is_emulated(w), "window reports emulation");

    check(cira_mmio_write32(w, 0x40, 0xcafef00d) == CIRA_MMIO_OK, "write32 in range succeeds");
    check(cira_mmio_read32(w, 0x40) == 0xcafef00d, "read32 returns what was written");

    check(cira_mmio_write64(w, 0x48, 0x0123456789abcdefull) == CIRA_MMIO_OK, "write64 in range succeeds");
    check(cira_mmio_read64(w, 0x48) == 0x0123456789abcdefull, "read64 returns what was written");

    // Out-of-range accesses must be refused rather than corrupting the host.
    check(cira_mmio_write32(w, CIRA_CXL_CONTROL_BYTES, 1) == CIRA_MMIO_ERANGE, "write past the window is rejected");
    check(cira_mmio_write32(w, CIRA_CXL_CONTROL_BYTES - 2, 1) == CIRA_MMIO_ERANGE,
          "write straddling the end is rejected");

    uint8_t payload[128];
    for (size_t i = 0; i < sizeof(payload); ++i)
        payload[i] = (uint8_t)(i * 7);
    check(cira_mmio_write_block(w, 0x80, payload, sizeof(payload)) == CIRA_MMIO_OK, "write_block succeeds");
    uint8_t readback[128] = {};
    check(cira_mmio_read_block(w, 0x80, readback, sizeof(readback)) == CIRA_MMIO_OK, "read_block succeeds");
    check(std::memcmp(payload, readback, sizeof(payload)) == 0, "block round-trips unchanged");
}

// --- 3. doorbell submission -------------------------------------------------

std::atomic<int> g_kernel_calls{0};

// Stands in for a compiler-generated device kernel.
void device_kernel(void **operands, uint32_t num_operands, void *completion) {
    g_kernel_calls.fetch_add(1, std::memory_order_relaxed);
    uint64_t sum = 0;
    for (uint32_t i = 0; i < num_operands; ++i) {
        sum += *static_cast<const uint64_t *>(operands[i]);
    }
    cira_mmio_signal_completion(completion, CIRA_CXL_STATUS_SUCCESS, sum);
}

void test_device_form_submit(cira_mmio_window_t *w) {
    std::printf("device-address submission\n");

    uint64_t a = 40, b = 2;
    void *operands[] = {&a, &b};
    constexpr uint64_t kCompletionHpa = 0x0000'1234'5678'9a80ull;

    uint64_t seq = 0;
    int rc = cira_mmio_submit_call_device(w, (void *)&device_kernel, operands, 2, kCompletionHpa, &seq);
    check(rc == CIRA_MMIO_OK, "device-form submit accepts aligned HPA");
    check(seq != 0, "device-form submit reports a sequence number");

    cira_cxl_call_job_t job = {};
    auto *base = static_cast<volatile uint8_t *>(cira_mmio_base(w));
    std::memcpy(&job,
                const_cast<const void *>(reinterpret_cast<volatile void *>(
                    base + cira_cxl_arg_slot_off(CIRA_CXL_JOB_CALL) + sizeof(cira_cxl_arg_slot_t))),
                sizeof(job));
    check(job.completion_addr == kCompletionHpa, "device-form descriptor preserves completion HPA");
    check(cira_mmio_submit_call_device(w, (void *)&device_kernel, operands, 2, kCompletionHpa + 1, nullptr) ==
              CIRA_MMIO_EINVAL,
          "device-form submit rejects unaligned HPA");
    check(cira_mmio_submit_call_device(w, (void *)&device_kernel, operands, 2, 0, nullptr) == CIRA_MMIO_EINVAL,
          "device-form submit rejects zero HPA");
}

// Minimal device firmware: poll the doorbell, run the job, mirror the status.
void firmware_loop(cira_mmio_window_t *w, std::atomic<bool> *stop) {
    auto *base = static_cast<volatile uint8_t *>(cira_mmio_base(w));
    auto *doorbell = reinterpret_cast<volatile cira_cxl_doorbell_t *>(base + CIRA_CXL_DOORBELL_OFF);
    auto *status = reinterpret_cast<volatile cira_cxl_host_status_t *>(base + CIRA_CXL_STATUS_OFF);

    uint64_t last_seq = 0;
    while (!stop->load(std::memory_order_relaxed)) {
        uint64_t seq = doorbell->seq;
        if (seq == last_seq || !cira_cxl_valid_magic(doorbell->magic)) {
            cira_wait_relax();
            continue;
        }
        last_seq = seq;
        __atomic_thread_fence(__ATOMIC_ACQUIRE);

        uint32_t job_id = doorbell->job_id;
        auto *slot = reinterpret_cast<volatile cira_cxl_arg_slot_t *>(base + cira_cxl_arg_slot_off(job_id));

        uint32_t result = CIRA_CXL_STATUS_BAD_JOB;
        if (cira_cxl_valid_magic(slot->magic) && slot->seq == seq && job_id == CIRA_CXL_JOB_CALL &&
            slot->arg_len >= sizeof(cira_cxl_call_job_t)) {
            cira_cxl_call_job_t job;
            std::memcpy(&job,
                        const_cast<const void *>(reinterpret_cast<volatile void *>(
                            base + cira_cxl_arg_slot_off(job_id) + sizeof(cira_cxl_arg_slot_t))),
                        sizeof(job));
            auto fn = reinterpret_cast<void (*)(void **, uint32_t, void *)>((uintptr_t)job.func_addr);
            fn(reinterpret_cast<void **>((uintptr_t)job.operands_addr), job.num_operands,
               reinterpret_cast<void *>((uintptr_t)job.completion_addr));
            result = CIRA_CXL_STATUS_SUCCESS;
        }

        status->magic = CIRA_CXL_JOB_MAGIC;
        status->version = CIRA_CXL_JOB_VERSION;
        status->job_id = job_id;
        status->status = result;
        __atomic_thread_fence(__ATOMIC_RELEASE);
        status->seq = seq;
    }
}

void test_doorbell_submit(cira_mmio_window_t *w) {
    std::printf("doorbell submission\n");

    std::atomic<bool> stop{false};
    std::thread firmware(firmware_loop, w, &stop);

    alignas(CIRA_CXL_CACHELINE_SIZE) cira_cxl_completion_t completion;
    cira_mmio_arm_completion(&completion);
    check(completion.magic != CIRA_CXL_COMPLETION_MAGIC, "armed completion starts unsignalled");

    uint64_t a = 40, b = 2;
    void *operands[] = {&a, &b};
    uint64_t seq = 0;
    int rc = cira_mmio_submit_call(w, (void *)&device_kernel, operands, 2, &completion, &seq);
    check(rc == CIRA_MMIO_OK, "submit_call rings the doorbell");
    check(seq != 0, "submit_call reports a sequence number");

    check(cira_mmio_wait_completion(&completion, 5ull * 1000 * 1000 * 1000) == CIRA_MMIO_OK,
          "host observes the completion line");
    check(completion.status == CIRA_CXL_STATUS_SUCCESS, "completion status is success");
    check(completion.result == 42, "device kernel computed the expected result");
    check(g_kernel_calls.load() == 1, "device kernel ran exactly once");

    check(cira_mmio_wait_seq(w, seq, 5ull * 1000 * 1000 * 1000) == CIRA_MMIO_OK,
          "device mirrors the sequence back to the host status register");

    // Argument payloads larger than a slot must be refused, not truncated.
    uint8_t oversized[CIRA_CXL_ARG_PAYLOAD_BYTES + 64] = {};
    check(cira_mmio_submit_job(w, CIRA_CXL_JOB_CALL, oversized, sizeof(oversized), 0, nullptr) == CIRA_MMIO_ERANGE,
          "oversized argument payload is rejected");
    check(cira_mmio_submit_job(w, CIRA_CXL_JOB_NOP, nullptr, 0, 0, nullptr) == CIRA_MMIO_EINVAL,
          "NOP job id is rejected");
    check(cira_mmio_submit_job(w, CIRA_CXL_JOB_MAX + 1, nullptr, 0, 0, nullptr) == CIRA_MMIO_EINVAL,
          "unknown job id is rejected");

    stop.store(true, std::memory_order_relaxed);
    firmware.join();
}

void test_kernel_id_submit() {
    std::printf("kernel-id arena submission\n");

    alignas(CIRA_CXL_CACHELINE_SIZE) static uint8_t arena_storage[0x10000] = {};
    char arena_addr[32];
    std::snprintf(arena_addr, sizeof(arena_addr), "0x%llx",
                  static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(arena_storage)));
    setenv("CIRA_CXL_ARENA_ADDR", arena_addr, 1);
    setenv("CIRA_CXL_ARENA_SIZE", "0x10000", 1);
    setenv("CIRA_CXL_ARENA_CONTROL_BYTES", "0x2000", 1);

    cira_shared_arena_t *arena = cira_shared_arena_default();
    check(arena != nullptr, "shared arena opens from the configured mapping");
    if (!arena)
        return;

    cira_mmio_window_t *window = open_emulated_window();
    check(window != nullptr, "kernel path has an emulated MMIO window");
    if (!window)
        return;
    cira_mmio_set_default(window);

    struct TypedArgs {
        uint64_t link_offset;
        uint32_t limit;
        uint32_t padding;
    };
    void *link = cira_shared_arena_from_offset(arena, 0x3450, 8, 8);
    uint64_t link_offset = cira::runtime::cira_offload_arg_offset(link, 8);
    check(link_offset == 0x3450, "remote argument converts to its arena-relative offset");
    TypedArgs typed_args{link_offset, 17, 0};
    void *completion = cira::runtime::cira_future_alloc();
    check(completion != nullptr, "kernel completion allocates from the shared arena");

    constexpr uint64_t kKernelId = 0x0123456789abcdefULL;
    int rc = cira::runtime::cira_offload_submit_kernel(kKernelId, &typed_args, sizeof(typed_args), completion, 0);
    check(rc == cira::runtime::CIRA_KERNEL_SUBMIT_OK, "kernel-id runtime submission succeeds");

    auto *base = static_cast<const std::byte *>(cira_mmio_base(window));
    auto *slot = reinterpret_cast<const cira_cxl_arg_slot_t *>(base + cira_cxl_arg_slot_off(CIRA_CXL_JOB_CALL_V2));
    auto *job = reinterpret_cast<const cira_cxl_call_job_v2_t *>(slot + 1);
    check(job->kernel_id == kKernelId, "descriptor carries the manifest kernel ID");
    check(job->args_size == sizeof(typed_args), "descriptor carries the packed argument size");

    uint64_t completion_offset = UINT64_MAX;
    check(cira_shared_arena_to_offset(arena, completion, CIRA_CXL_CACHELINE_SIZE, &completion_offset) == CIRA_ARENA_OK,
          "completion converts to an arena offset");
    check(job->completion_offset == completion_offset, "descriptor carries the arena-relative completion offset");

    auto *copied_args = static_cast<const TypedArgs *>(
        cira_shared_arena_from_offset(arena, job->args_offset, sizeof(TypedArgs), alignof(TypedArgs)));
    check(copied_args != nullptr, "descriptor argument offset resolves inside the arena");
    check(copied_args && std::memcmp(copied_args, &typed_args, sizeof(typed_args)) == 0,
          "runtime copied the typed argument payload into the arena");
    check(slot->version == CIRA_CXL_JOB_VERSION_KERNEL_ID && slot->job_id == CIRA_CXL_JOB_CALL_V2,
          "kernel submission publishes the v2 slot header");

    auto *doorbell = reinterpret_cast<const cira_cxl_doorbell_t *>(base + CIRA_CXL_DOORBELL_OFF);
    cira_cxl_host_status_t status{CIRA_CXL_JOB_MAGIC, CIRA_CXL_JOB_VERSION_KERNEL_ID, CIRA_CXL_JOB_CALL_V2,
                                  CIRA_CXL_STATUS_SUCCESS, doorbell->seq};
    check(cira_mmio_write_block(window, CIRA_CXL_STATUS_OFF, &status, sizeof(status)) == CIRA_MMIO_OK,
          "firmware status mirrors the tracked v2 sequence");
    cira_mmio_signal_completion(completion, CIRA_CXL_STATUS_SUCCESS, 0);
    (void)cira::runtime::cira_future_await(completion);
    cira::runtime::cira_offload_require_success(rc, completion);
    cira::runtime::cira_future_free(completion);

    uint64_t committed_seq = doorbell->seq;
    check(cira::runtime::cira_offload_arg_offset(&typed_args, 1) == UINT64_MAX,
          "foreign remote argument conversion fails closed");
    void *bad_completion = cira::runtime::cira_future_alloc();
    rc = cira::runtime::cira_offload_submit_kernel(kKernelId, &typed_args, sizeof(typed_args), bad_completion, 0);
    check(rc == cira::runtime::CIRA_KERNEL_SUBMIT_ECONVERT,
          "submission returns the recorded argument conversion failure");
    auto *bad_line = static_cast<cira_cxl_completion_t *>(bad_completion);
    check(bad_line->magic == CIRA_CXL_COMPLETION_MAGIC && bad_line->status == CIRA_CXL_STATUS_BAD_ARGS,
          "conversion failure publishes an explicit error completion");
    check(doorbell->seq == committed_seq, "conversion failure does not commit another doorbell");
    cira::runtime::cira_future_free(bad_completion);

    cira_mmio_set_default(nullptr);
    void *no_mmio_completion = cira::runtime::cira_future_alloc();
    rc = cira::runtime::cira_offload_submit_kernel(kKernelId, &typed_args, sizeof(typed_args), no_mmio_completion, 0);
    check(rc == cira::runtime::CIRA_KERNEL_SUBMIT_ENOMMIO,
          "missing MMIO is returned instead of a successful completion");
    auto *no_mmio_line = static_cast<cira_cxl_completion_t *>(no_mmio_completion);
    check(no_mmio_line->magic == CIRA_CXL_COMPLETION_MAGIC && no_mmio_line->status == CIRA_CXL_STATUS_BAD_JOB,
          "missing MMIO publishes an explicit failure status");
    cira::runtime::cira_future_free(no_mmio_completion);

    setenv("CIRA_OFFLOAD_MODE", "baseline", 1);
    char telemetry_path[] = "/tmp/cira-offload-telemetry.XXXXXX";
    int telemetry_fd = mkstemp(telemetry_path);
    check(telemetry_fd >= 0, "query telemetry test creates an output file");
    if (telemetry_fd >= 0)
        close(telemetry_fd);
    setenv("CIRA_OFFLOAD_TELEMETRY", telemetry_path, 1);
    cira::runtime::cira_offload_set_query_context(6);
    void *baseline_completion = cira::runtime::cira_future_alloc();
    rc = cira::runtime::cira_offload_submit_kernel(kKernelId, &typed_args, sizeof(typed_args), baseline_completion, 0);
    check(rc == cira::runtime::CIRA_KERNEL_SUBMIT_OK,
          "baseline mode skips the missing device and completes the host path");
    auto *baseline_line = static_cast<cira_cxl_completion_t *>(baseline_completion);
    check(baseline_line->magic == CIRA_CXL_COMPLETION_MAGIC &&
              baseline_line->status == CIRA_CXL_STATUS_SUCCESS,
          "baseline mode publishes a successful software completion");
    cira::runtime::cira_offload_require_success(rc, baseline_completion);
    cira::runtime::cira_future_free(baseline_completion);
    std::ifstream telemetry_stream(telemetry_path);
    std::string telemetry((std::istreambuf_iterator<char>(telemetry_stream)),
                          std::istreambuf_iterator<char>());
    check(telemetry.find("\"query_id\":6") != std::string::npos &&
              telemetry.find("\"kernel_id\":\"0x0123456789abcdef\"") != std::string::npos,
          "submit telemetry attributes the kernel to the active query");
    cira::runtime::cira_offload_set_query_context(0);
    unsetenv("CIRA_OFFLOAD_TELEMETRY");
    unlink(telemetry_path);
    unsetenv("CIRA_OFFLOAD_MODE");

    setenv("CIRA_OFFLOAD_MODE", "forced-debug", 1);
    void *native_forced_completion = cira::runtime::cira_future_alloc();
    rc = cira::runtime::cira_offload_submit_kernel(kKernelId, &typed_args, sizeof(typed_args),
                                                   native_forced_completion, 0);
    check(rc == cira::runtime::CIRA_KERNEL_SUBMIT_ENOMMIO,
          "native forced-debug reports a missing device");
    cira::runtime::cira_offload_require_success(rc, native_forced_completion);
    cira::runtime::cira_future_free(native_forced_completion);
    unsetenv("CIRA_OFFLOAD_MODE");

    pid_t child = fork();
    check(child >= 0, "fail-closed completion check starts a child process");
    if (child == 0) {
        alignas(CIRA_CXL_CACHELINE_SIZE) cira_cxl_completion_t failed{};
        failed.magic = CIRA_CXL_COMPLETION_MAGIC;
        failed.status = CIRA_CXL_STATUS_BAD_JOB;
        cira::runtime::cira_offload_require_success(cira::runtime::CIRA_KERNEL_SUBMIT_ESUBMIT, &failed);
        _exit(0);
    }
    if (child > 0) {
        int child_status = 0;
        check(waitpid(child, &child_status, 0) == child, "fail-closed child is collected");
        check(WIFSIGNALED(child_status) && WTERMSIG(child_status) == SIGABRT,
              "failed submission/completion aborts instead of continuing the query");
    }
}

void test_no_device() {
    std::printf("no device configured\n");
    check(cira_mmio_submit_call(nullptr, nullptr, nullptr, 0, nullptr, nullptr) == CIRA_MMIO_ENODEV,
          "submit without a window reports ENODEV");
    check(cira_mmio_wait_seq(nullptr, 1, 0) == CIRA_MMIO_ENODEV, "wait_seq without a window reports ENODEV");
}

} // namespace

int main() {
    std::printf("=== CIRA offload dispatch tests ===\n");

    test_wait_backend();
    test_no_device();

    cira_mmio_window_t *w = open_emulated_window();
    if (!w) {
        std::fprintf(stderr, "[FAIL] could not open emulated MMIO window\n");
        return 1;
    }
    test_mmio_registers(w);
    test_device_form_submit(w);
    cira_mmio_close(w);

    w = open_emulated_window();
    if (!w) {
        std::fprintf(stderr, "[FAIL] could not open emulated MMIO window for pointer submission\n");
        return 1;
    }
    test_doorbell_submit(w);
    cira_mmio_close(w);

    test_kernel_id_submit();

    if (g_failures) {
        std::printf("\n%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("\nAll checks passed\n");
    return 0;
}
