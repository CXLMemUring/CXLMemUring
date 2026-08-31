#include "cira_rv64_device.h"
#include "cira_rv64_dispatch.generated.h"

static volatile cira_cxl_doorbell_t *const doorbell =
    (volatile cira_cxl_doorbell_t *)(uintptr_t)(CIRA_DEVICE_CONTROL_BASE + CIRA_CXL_DOORBELL_OFF);
static volatile cira_cxl_host_status_t *const host_status =
    (volatile cira_cxl_host_status_t *)(uintptr_t)(CIRA_DEVICE_CONTROL_BASE + CIRA_CXL_STATUS_OFF);
static uint32_t kernel_status;

static int cira_arena_range_valid(uint64_t offset, uint64_t bytes, uint64_t alignment);

__attribute__((noinline)) void cira_device_install_cacheline(uint64_t offset) {
    offset &= ~(uint64_t)(CIRA_CXL_CACHELINE_SIZE - 1);
    if (kernel_status != CIRA_CXL_STATUS_SUCCESS ||
        !cira_arena_range_valid(offset, CIRA_CXL_CACHELINE_SIZE, CIRA_CXL_CACHELINE_SIZE)) {
        kernel_status = CIRA_CXL_STATUS_BAD_ARGS;
        return;
    }
    __asm__ volatile(".insn r 0x0b, 0, 0, x0, %0, x0" : : "r"(offset) : "memory");
}

static uint64_t cira_load_acquire64(volatile const uint64_t *address) {
    uint64_t value = *address;
    cira_device_fence_acquire();
    return value;
}

static void cira_copy_from_volatile(void *destination, volatile const void *source, size_t bytes) {
    uint8_t *out = (uint8_t *)destination;
    volatile const uint8_t *in = (volatile const uint8_t *)source;
    for (size_t index = 0; index < bytes; ++index)
        out[index] = in[index];
}

static int cira_arena_range_valid(uint64_t offset, uint64_t bytes, uint64_t alignment) {
    if (offset < CIRA_CXL_CONTROL_BYTES || (alignment && offset % alignment))
        return 0;
    return offset <= CIRA_DEVICE_ARENA_SIZE && bytes <= CIRA_DEVICE_ARENA_SIZE - offset;
}

void cira_device_kernel_begin(void) { kernel_status = CIRA_CXL_STATUS_SUCCESS; }

uint32_t cira_device_kernel_status(void) { return kernel_status; }

uint64_t cira_device_checked_load(uint64_t offset, uint32_t bytes, uint32_t alignment) {
    if (kernel_status != CIRA_CXL_STATUS_SUCCESS)
        return 0;
    if (!cira_arena_range_valid(offset, bytes, alignment)) {
        kernel_status = CIRA_CXL_STATUS_BAD_ARGS;
        return 0;
    }

    uintptr_t address = (uintptr_t)(CIRA_DEVICE_ARENA_BASE + offset);
    switch (bytes) {
    case 1:
        return *(volatile const uint8_t *)address;
    case 2:
        return *(volatile const uint16_t *)address;
    case 4:
        return *(volatile const uint32_t *)address;
    case 8:
        return *(volatile const uint64_t *)address;
    default:
        kernel_status = CIRA_CXL_STATUS_BAD_ARGS;
        return 0;
    }
}

static volatile cira_cxl_completion_t *cira_completion(uint64_t offset) {
    if (!cira_arena_range_valid(offset, sizeof(cira_cxl_completion_t), CIRA_CXL_CACHELINE_SIZE))
        return (volatile cira_cxl_completion_t *)0;
    return (volatile cira_cxl_completion_t *)(uintptr_t)(CIRA_DEVICE_ARENA_BASE + offset);
}

static void cira_publish_host_status(uint64_t seq, uint32_t status) {
    host_status->magic = 0;
    host_status->version = CIRA_CXL_JOB_VERSION_KERNEL_ID;
    host_status->job_id = CIRA_CXL_JOB_CALL_V2;
    host_status->status = status;
    host_status->seq = seq;
    cira_device_fence_release();
    host_status->magic = CIRA_CXL_JOB_MAGIC;
}

static void cira_publish_completion(volatile cira_cxl_completion_t *completion, uint64_t seq, uint32_t status) {
    if (!completion)
        return;
    completion->magic = 0;
    completion->status = status;
    completion->result = 0;
    completion->cycles = 0;
    completion->timestamp = 0;
    completion->seq = seq;
    cira_device_fence_release();
    completion->magic = CIRA_CXL_COMPLETION_MAGIC;
}

static void cira_publish_error(uint64_t seq, uint32_t status, volatile cira_cxl_completion_t *completion) {
    cira_publish_completion(completion, seq, status);
    cira_publish_host_status(seq, status);
}

static void cira_publish_success(uint64_t seq, volatile cira_cxl_completion_t *completion) {
    cira_publish_completion(completion, seq, CIRA_CXL_STATUS_SUCCESS);
    cira_publish_host_status(seq, CIRA_CXL_STATUS_SUCCESS);
}

static const struct cira_kernel_entry *cira_find_kernel(uint64_t kernel_id) {
    size_t begin = 0;
    size_t end = cira_kernel_table_count;
    while (begin < end) {
        size_t middle = begin + (end - begin) / 2;
        uint64_t candidate = cira_kernel_table[middle].kernel_id;
        if (candidate < kernel_id)
            begin = middle + 1;
        else
            end = middle;
    }
    if (begin == cira_kernel_table_count || cira_kernel_table[begin].kernel_id != kernel_id)
        return (const struct cira_kernel_entry *)0;
    return &cira_kernel_table[begin];
}

static uint32_t cira_snapshot_and_validate(uint64_t seq, cira_cxl_call_job_v2_t *job,
                                           volatile cira_cxl_completion_t **completion) {
    if (!seq || !cira_cxl_valid_magic(doorbell->magic) || doorbell->job_id != CIRA_CXL_JOB_CALL_V2)
        return CIRA_CXL_STATUS_BAD_DESCRIPTOR;
    if (doorbell->version != CIRA_CXL_JOB_VERSION_KERNEL_ID)
        return CIRA_CXL_STATUS_BAD_VERSION;

    volatile const cira_cxl_arg_slot_t *slot =
        (volatile const cira_cxl_arg_slot_t *)(uintptr_t)(CIRA_DEVICE_CONTROL_BASE +
                                                          cira_cxl_arg_slot_off(CIRA_CXL_JOB_CALL_V2));
    if (!cira_cxl_valid_magic(slot->magic) || slot->job_id != CIRA_CXL_JOB_CALL_V2 || slot->seq != seq ||
        slot->arg_len != sizeof(*job))
        return CIRA_CXL_STATUS_BAD_DESCRIPTOR;
    if (slot->version != CIRA_CXL_JOB_VERSION_KERNEL_ID)
        return CIRA_CXL_STATUS_BAD_VERSION;

    cira_copy_from_volatile(job, (volatile const uint8_t *)slot + sizeof(*slot), sizeof(*job));
    cira_device_fence_acquire();
    if (slot->seq != seq || cira_load_acquire64(&doorbell->seq) != seq || job->reserved != 0 ||
        job->flags != doorbell->flags)
        return CIRA_CXL_STATUS_BAD_DESCRIPTOR;
    if (!cira_arena_range_valid(job->args_offset, job->args_size, 8))
        return CIRA_CXL_STATUS_BAD_ARGS;
    *completion = cira_completion(job->completion_offset);
    if (!*completion)
        return CIRA_CXL_STATUS_COMPLETION_IO;
    return CIRA_CXL_STATUS_SUCCESS;
}

void cira_device_main(void) {
    uint64_t last_seq = 0;
    uint64_t highest_seq = 0;
    for (;;) {
        uint64_t seq = cira_load_acquire64(&doorbell->seq);
        if (seq == last_seq) {
            cira_device_pause();
            continue;
        }
        last_seq = seq;
        if (highest_seq && seq <= highest_seq) {
            cira_publish_error(seq, CIRA_CXL_STATUS_STALE_SEQUENCE, (volatile cira_cxl_completion_t *)0);
            continue;
        }
        highest_seq = seq;

        cira_cxl_call_job_v2_t job = {0};
        volatile cira_cxl_completion_t *completion = (volatile cira_cxl_completion_t *)0;
        uint32_t status = cira_snapshot_and_validate(seq, &job, &completion);
        if (status != CIRA_CXL_STATUS_SUCCESS) {
            cira_publish_error(seq, status, completion);
            continue;
        }

        const struct cira_kernel_entry *entry = cira_find_kernel(job.kernel_id);
        if (!entry || entry->args_size != job.args_size) {
            cira_publish_error(seq, CIRA_CXL_STATUS_UNKNOWN_KERNEL, completion);
            continue;
        }

        cira_device_kernel_begin();
        entry->fn(CIRA_DEVICE_ARENA_BASE, job.args_offset);
        status = cira_device_kernel_status();
        if (status == CIRA_CXL_STATUS_SUCCESS)
            cira_publish_success(seq, completion);
        else
            cira_publish_error(seq, status, completion);
    }
}
