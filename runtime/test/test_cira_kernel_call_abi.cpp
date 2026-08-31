#include "cira_cxl_job.h"
#include "cira_mmio.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>

static_assert(sizeof(cira_cxl_call_job_v1_t) == 32);
static_assert(sizeof(cira_cxl_call_job_t) == sizeof(cira_cxl_call_job_v1_t));
static_assert(offsetof(cira_cxl_call_job_v2_t, kernel_id) == 0);
static_assert(offsetof(cira_cxl_call_job_v2_t, args_offset) == 8);
static_assert(offsetof(cira_cxl_call_job_v2_t, completion_offset) == 16);
static_assert(sizeof(cira_cxl_call_job_v2_t) == 40);

int main() {
    cira_mmio_window_t *window = nullptr;
    cira_mmio_config_t cfg{};
    cfg.emulate = true;
    assert(cira_mmio_open(&window, &cfg) == CIRA_MMIO_OK);

    uint64_t seq = 0;
    assert(cira_mmio_submit_kernel(window, 0x1122334455667788ULL,
                                   0x400, 96, 0x800, 0, &seq) == CIRA_MMIO_OK);

    auto *base = static_cast<const std::byte *>(cira_mmio_base(window));
    auto *slot = reinterpret_cast<const cira_cxl_arg_slot_t *>(
        base + cira_cxl_arg_slot_off(CIRA_CXL_JOB_CALL_V2));
    auto *job = reinterpret_cast<const cira_cxl_call_job_v2_t *>(slot + 1);
    auto *doorbell = reinterpret_cast<const cira_cxl_doorbell_t *>(
        base + CIRA_CXL_DOORBELL_OFF);

    assert(slot->magic == CIRA_CXL_JOB_MAGIC);
    assert(slot->version == CIRA_CXL_JOB_VERSION_KERNEL_ID);
    assert(slot->job_id == CIRA_CXL_JOB_CALL_V2);
    assert(slot->arg_len == sizeof(cira_cxl_call_job_v2_t));
    assert(slot->seq == seq);
    assert(job->kernel_id == 0x1122334455667788ULL);
    assert(job->args_offset == 0x400 && job->args_size == 96);
    assert(job->completion_offset == 0x800);
    assert(job->flags == 0 && job->reserved == 0);
    assert(doorbell->version == CIRA_CXL_JOB_VERSION_KERNEL_ID);
    assert(doorbell->job_id == CIRA_CXL_JOB_CALL_V2);
    assert(doorbell->seq == seq);

    const uint64_t committed_seq = doorbell->seq;
    assert(cira_mmio_submit_kernel(window, 0, 0x400, 96, 0x800, 0, nullptr) ==
           CIRA_MMIO_EINVAL);
    assert(cira_mmio_submit_kernel(window, 1, 0x400, 96, 0x801, 0, nullptr) ==
           CIRA_MMIO_EINVAL);
    assert(cira_mmio_submit_kernel(window, 1,
                                   std::numeric_limits<uint64_t>::max() - 31,
                                   64, 0x800, 0, nullptr) == CIRA_MMIO_ERANGE);
    assert(cira_mmio_submit_kernel(window, 1, 0x400, 96,
                                   std::numeric_limits<uint64_t>::max() - 63,
                                   0, nullptr) == CIRA_MMIO_ERANGE);
    assert(doorbell->seq == committed_seq);

    cira_mmio_close(window);
    return 0;
}
