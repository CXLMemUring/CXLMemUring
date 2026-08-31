// Bounded shared CXL-memory arena.
//
// Wire descriptors carry offsets returned by this API, never host virtual
// pointers. Host and device may therefore map the same backing memory at
// different virtual addresses.

#ifndef CIRA_SHARED_ARENA_H
#define CIRA_SHARED_ARENA_H

#include <stdint.h>

#include "cira_cxl_job.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cira_shared_arena cira_shared_arena_t;

#define CIRA_ARENA_OK 0
#define CIRA_ARENA_EINVAL -1
#define CIRA_ARENA_ERANGE -2
#define CIRA_ARENA_ENOMEM -3

int cira_shared_arena_open(cira_shared_arena_t **out, void *base,
                           uint64_t bytes, uint64_t control_bytes);
void cira_shared_arena_close(cira_shared_arena_t *arena);

int cira_shared_arena_alloc(cira_shared_arena_t *arena, uint64_t bytes,
                            uint64_t alignment, uint64_t *out_offset);
int cira_shared_arena_to_offset(const cira_shared_arena_t *arena,
                                const void *ptr, uint64_t bytes,
                                uint64_t *out_offset);
void *cira_shared_arena_from_offset(const cira_shared_arena_t *arena,
                                    uint64_t offset, uint64_t bytes,
                                    uint64_t alignment);

// Lazily opens the process-wide arena from CIRA_CXL_ARENA_ADDR,
// CIRA_CXL_ARENA_SIZE, and CIRA_CXL_ARENA_CONTROL_BYTES. In strict mode
// (CIRA_CXL_ARENA_STRICT=1 or CIRA_CXL_MMIO_STRICT=1), ADDR is mandatory.
// Without strict mode, a configured SIZE with no ADDR gets anonymous backing.
cira_shared_arena_t *cira_shared_arena_default(void);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CIRA_SHARED_ARENA_H
