#include "CiraRuntime.h"
#include "cira_cxl_job.h"
#include "cira_mmio.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>

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
    check(cira_register_linear_region(&completion, sizeof(completion), kCompletionHpa, 0) == 0,
          "registers coherent completion HPA");
    std::memset(control, 0, sizeof(control));
    cira_offload_submit(reinterpret_cast<void *>(&device_kernel), operands, 1, &completion);
    doorbell = reinterpret_cast<const cira_cxl_doorbell_t *>(control + CIRA_CXL_DOORBELL_OFF);
    check(doorbell->seq != 0, "registered completion rings hardware doorbell");
    check(staged_call(control)->completion_addr == kCompletionHpa,
          "hardware descriptor contains registered completion HPA");
    check(cira_unregister_linear_region(&completion) == 0, "unregisters coherent completion HPA");

    cira_mmio_set_default(nullptr);
    unsetenv("CIRA_CXL_CACHE_DEV");
}

} // namespace

int main() {
    std::printf("=== CIRA CXL.cache completion tests ===\n");
    test_hardware_submit_requires_trusted_completion_hpa();
    if (failures) {
        std::printf("\n%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("\nAll checks passed\n");
    return 0;
}
