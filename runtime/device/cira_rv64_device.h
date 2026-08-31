#ifndef CIRA_RV64_DEVICE_H
#define CIRA_RV64_DEVICE_H

#include "cira_cxl_job.h"

#include <stddef.h>
#include <stdint.h>

#ifndef CIRA_DEVICE_CONTROL_BASE
#define CIRA_DEVICE_CONTROL_BASE UINT64_C(0x40000000)
#endif

#ifndef CIRA_DEVICE_ARENA_BASE
#define CIRA_DEVICE_ARENA_BASE UINT64_C(0x80000000)
#endif

#ifndef CIRA_DEVICE_ARENA_SIZE
#define CIRA_DEVICE_ARENA_SIZE UINT64_C(0x10000000)
#endif

typedef void (*cira_rv64_kernel_fn)(uint64_t arena_base, uint64_t args_offset);

struct cira_kernel_entry {
    uint64_t kernel_id;
    cira_rv64_kernel_fn fn;
    uint32_t args_size;
};

static inline void cira_device_pause(void) { __asm__ volatile("nop" ::: "memory"); }

static inline void cira_device_fence_acquire(void) { __asm__ volatile("fence r, rw" ::: "memory"); }

static inline void cira_device_fence_release(void) { __asm__ volatile("fence rw, w" ::: "memory"); }

// custom-0, funct3=0, funct7=0, rd=x0, rs2=x0, rs1=offset. gem5
// decodes only this helper's instruction as an explicit cache install.
void cira_device_install_cacheline(uint64_t offset);
uint64_t cira_device_checked_load(uint64_t offset, uint32_t bytes, uint32_t alignment);
void cira_device_kernel_begin(void);
uint32_t cira_device_kernel_status(void);

void cira_device_main(void) __attribute__((noreturn));

#endif
