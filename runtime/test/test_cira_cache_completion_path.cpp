#include "CiraRuntime.h"
#include "cira_cxl_job.h"
#include "cira_mmio.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <thread>

using namespace cira::runtime;

namespace {

int failures = 0;

void check(bool condition, const char *message) {
    std::printf("  [%s] %s\n", condition ? " ok " : "FAIL", message);
    if (!condition)
        ++failures;
}

void device_kernel(void **, uint32_t, void *) {}

const cira_cxl_call_job_t *staged_call(const uint8_t *control) {
    return reinterpret_cast<const cira_cxl_call_job_t *>(
        control + cira_cxl_arg_slot_off(CIRA_CXL_JOB_CALL) + sizeof(cira_cxl_arg_slot_t));
}

void test_hardware_submit_requires_trusted_completion_hpa() {
    std::printf("hardware completion HPA gate\n");

    // Keep this test independent of any real CXL.cache device on the host:
    // the first submission must have no trusted translation source.
    setenv("CIRA_CXL_CACHE_DEV", "/definitely/not/a/cira-cache-device", 1);

    alignas(CIRA_CXL_CACHELINE_SIZE) uint8_t control[CIRA_CXL_CONTROL_BYTES] = {};
    cira_mmio_config_t cfg = {};
    cfg.fixed_addr = reinterpret_cast<uintptr_t>(control);
    cfg.size = sizeof(control);

    cira_mmio_window_t *window = nullptr;
    check(cira_mmio_open(&window, &cfg) == CIRA_MMIO_OK, "opens non-emulated hardware-mode window");
    if (!window)
        return;
    check(!cira_mmio_is_emulated(window), "fixed-address window is hardware mode");
    cira_mmio_set_default(window); // Transfers window ownership to the runtime.

    alignas(CIRA_CXL_CACHELINE_SIZE) cira_cxl_completion_t completion = {};
    uint64_t operand = 7;
    void *operands[] = {&operand};

    cira_offload_submit(reinterpret_cast<void *>(&device_kernel), operands, 1, &completion);
    auto *doorbell = reinterpret_cast<const cira_cxl_doorbell_t *>(control + CIRA_CXL_DOORBELL_OFF);
    check(doorbell->seq == 0, "untranslated heap completion does not ring hardware doorbell");
    check(completion.magic == CIRA_CXL_COMPLETION_MAGIC, "untranslated completion falls back to software");

    constexpr uint64_t kCompletionHpa = 0x0000'0004'0000'0000ull;
    check(cira_register_linear_region(&completion, sizeof(completion) - 1, kCompletionHpa, 0) == 0,
          "registers an intentionally undersized completion region");
    std::memset(control, 0, sizeof(control));
    cira_offload_submit(reinterpret_cast<void *>(&device_kernel), operands, 1, &completion);
    doorbell = reinterpret_cast<const cira_cxl_doorbell_t *>(control + CIRA_CXL_DOORBELL_OFF);
    check(doorbell->seq == 0, "partial completion registration does not ring hardware doorbell");
    check(completion.magic == CIRA_CXL_COMPLETION_MAGIC,
          "partial completion registration falls back to software");
    check(cira_unregister_linear_region(&completion) == 0, "unregisters undersized completion region");

    check(cira_register_linear_region(&completion, sizeof(completion), kCompletionHpa, 0) == 0,
          "registers coherent completion HPA");
    std::memset(control, 0, sizeof(control));
    cira_offload_submit(reinterpret_cast<void *>(&device_kernel), operands, 1, &completion);
    doorbell = reinterpret_cast<const cira_cxl_doorbell_t *>(control + CIRA_CXL_DOORBELL_OFF);
    check(doorbell->seq != 0, "registered completion rings hardware doorbell");
    check(staged_call(control)->completion_addr == kCompletionHpa,
          "hardware descriptor contains registered completion HPA");
    const uint64_t first_seq = doorbell->seq;
    check(cira_unregister_linear_region(&completion) != 0,
          "in-flight completion registration cannot be removed");

    // The CXL.cache magic write can become observable before the writer's
    // commit B response.  It must not release the HPA or admit another job.
    cira_mmio_signal_completion(&completion, CIRA_CXL_STATUS_SUCCESS, 0x1234);
    check(cira_unregister_linear_region(&completion) != 0,
          "magic before commit B does not release completion registration");
    setenv("CIRA_CXL_MMIO_TIMEOUT_NS", "1000", 1);
    (void)cira_future_await(&completion);
    unsetenv("CIRA_CXL_MMIO_TIMEOUT_NS");
    check(completion.status == CIRA_CXL_STATUS_COMPLETION_IO,
          "magic without matching terminal status cannot become a successful future");
    check(cira_unregister_linear_region(&completion) != 0,
          "timed-out candidate magic keeps completion registration protected");

    alignas(CIRA_CXL_CACHELINE_SIZE) cira_cxl_completion_t second_completion = {};
    check(cira_register_linear_region(&second_completion, sizeof(second_completion), kCompletionHpa + 64, 0) == 0,
          "registers a second coherent completion HPA");
    cira_offload_submit(reinterpret_cast<void *>(&device_kernel), operands, 1, &second_completion);
    check(doorbell->seq == first_seq,
          "second asynchronous call cannot overwrite the single hardware status mirror");
    check(second_completion.magic == CIRA_CXL_COMPLETION_MAGIC,
          "second call falls back to software while the hardware completion channel is busy");
    check(cira_unregister_linear_region(&second_completion) == 0,
          "software-fallback completion registration can be removed");

    auto *status = reinterpret_cast<volatile cira_cxl_host_status_t *>(control + CIRA_CXL_STATUS_OFF);
    status->magic = CIRA_CXL_JOB_MAGIC;
    status->version = CIRA_CXL_JOB_VERSION;
    status->job_id = CIRA_CXL_JOB_CALL;
    status->status = CIRA_CXL_STATUS_COMPLETION_IO;
    __atomic_thread_fence(__ATOMIC_RELEASE);
    status->seq = first_seq;

    setenv("CIRA_CXL_MMIO_TIMEOUT_NS", "1000", 1);
    (void)cira_future_await(&completion);
    unsetenv("CIRA_CXL_MMIO_TIMEOUT_NS");
    check(completion.magic == CIRA_CXL_COMPLETION_MAGIC,
          "status completion-I/O error publishes a terminal completion");
    check(completion.status == CIRA_CXL_STATUS_COMPLETION_IO,
          "terminal completion preserves the device completion-I/O status");
    check(cira_unregister_linear_region(&completion) == 0,
          "terminal completion registration can be removed");

    cira_mmio_set_default(nullptr);
    unsetenv("CIRA_CXL_CACHE_DEV");
}

void test_hardware_completion_channel_is_single_flight() {
    std::printf("hardware completion channel single flight\n");
    setenv("CIRA_CXL_CACHE_DEV", "/definitely/not/a/cira-cache-device", 1);

    alignas(CIRA_CXL_CACHELINE_SIZE) uint8_t control[CIRA_CXL_CONTROL_BYTES] = {};
    cira_mmio_config_t cfg = {};
    cfg.fixed_addr = reinterpret_cast<uintptr_t>(control);
    cfg.size = sizeof(control);
    cira_mmio_window_t *window = nullptr;
    check(cira_mmio_open(&window, &cfg) == CIRA_MMIO_OK, "opens single-flight hardware window");
    if (!window)
        return;
    cira_mmio_set_default(window);

    alignas(CIRA_CXL_CACHELINE_SIZE) cira_cxl_completion_t completions[2] = {};
    constexpr uint64_t kCompletionHpa = 0x0000'0005'0000'0000ull;
    check(cira_register_linear_region(&completions[0], sizeof(completions[0]), kCompletionHpa, 0) == 0,
          "registers first concurrent completion");
    check(cira_register_linear_region(&completions[1], sizeof(completions[1]), kCompletionHpa + 64, 0) == 0,
          "registers second concurrent completion");

    uint64_t operand = 9;
    void *operands[] = {&operand};
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    auto submit = [&](cira_cxl_completion_t *completion) {
        ready.fetch_add(1, std::memory_order_release);
        while (!go.load(std::memory_order_acquire))
            std::this_thread::yield();
        cira_offload_submit(reinterpret_cast<void *>(&device_kernel), operands, 1, completion);
    };
    std::thread first(submit, &completions[0]);
    std::thread second(submit, &completions[1]);
    while (ready.load(std::memory_order_acquire) != 2)
        std::this_thread::yield();
    go.store(true, std::memory_order_release);
    first.join();
    second.join();

    auto *doorbell = reinterpret_cast<const cira_cxl_doorbell_t *>(control + CIRA_CXL_DOORBELL_OFF);
    const bool first_hardware = completions[0].magic != CIRA_CXL_COMPLETION_MAGIC;
    const bool second_hardware = completions[1].magic != CIRA_CXL_COMPLETION_MAGIC;
    check(doorbell->seq == 1, "concurrent submissions ring at most one hardware doorbell");
    check(first_hardware != second_hardware,
          "exactly one concurrent completion remains pending in hardware");

    cira_cxl_completion_t *hardware_completion = first_hardware ? &completions[0] : &completions[1];
    auto *status = reinterpret_cast<volatile cira_cxl_host_status_t *>(control + CIRA_CXL_STATUS_OFF);
    status->magic = CIRA_CXL_JOB_MAGIC;
    status->version = CIRA_CXL_JOB_VERSION;
    status->job_id = CIRA_CXL_JOB_CALL;
    status->status = CIRA_CXL_STATUS_COMPLETION_IO;
    __atomic_thread_fence(__ATOMIC_RELEASE);
    status->seq = doorbell->seq;
    setenv("CIRA_CXL_MMIO_TIMEOUT_NS", "1000", 1);
    (void)cira_future_await(hardware_completion);
    unsetenv("CIRA_CXL_MMIO_TIMEOUT_NS");
    check(hardware_completion->status == CIRA_CXL_STATUS_COMPLETION_IO,
          "single in-flight hardware completion receives its mirrored error");
    check(cira_unregister_linear_region(&completions[0]) == 0,
          "first concurrent completion registration is released at terminal status");
    check(cira_unregister_linear_region(&completions[1]) == 0,
          "second concurrent completion registration is released at terminal status");

    cira_mmio_set_default(nullptr);
    unsetenv("CIRA_CXL_CACHE_DEV");
}

} // namespace

int main() {
    std::printf("=== CIRA CXL.cache completion tests ===\n");
    test_hardware_submit_requires_trusted_completion_hpa();
    test_hardware_completion_channel_is_single_flight();
    if (failures) {
        std::printf("\n%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("\nAll checks passed\n");
    return 0;
}
