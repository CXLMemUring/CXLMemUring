// Vortex GPU Intrinsics
// Declarations for Vortex SIMT GPU built-in functions

#ifndef VX_INTRINSICS_H
#define VX_INTRINSICS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Vortex CSR addresses
#define VX_CSR_MSCRATCH     0x340
#define VX_CSR_THREAD_ID    0xCC0
#define VX_CSR_WARP_ID      0xCC1
#define VX_CSR_CORE_ID      0xCC2
#define VX_CSR_LANE_ID      0xCC3
#define VX_CSR_NT           0xFC0
#define VX_CSR_NW           0xFC1
#define VX_CSR_NC           0xFC2

// CSR read/write macros
#define csr_read(csr) ({ \
    uint32_t __val; \
    asm volatile ("csrr %0, %1" : "=r"(__val) : "i"(csr)); \
    __val; \
})

#define csr_write(csr, val) ({ \
    asm volatile ("csrw %0, %1" :: "i"(csr), "r"(val)); \
})

// Kernel function callback type
typedef void (*vx_kernel_func_cb)(void* arg);

// Thread identification (must be before spawn functions)
static inline uint32_t vx_thread_id() {
    uint32_t tid;
    asm volatile ("csrr %0, %1" : "=r"(tid) : "i"(0xCC0));
    return tid;
}

static inline uint32_t vx_warp_id() {
    uint32_t wid;
    asm volatile ("csrr %0, %1" : "=r"(wid) : "i"(0xCC1));
    return wid;
}

static inline uint32_t vx_core_id() {
    uint32_t cid;
    asm volatile ("csrr %0, %1" : "=r"(cid) : "i"(0xCC2));
    return cid;
}

static inline uint32_t vx_lane_id() {
    uint32_t lid;
    asm volatile ("csrr %0, %1" : "=r"(lid) : "i"(0xCC3));
    return lid;
}

static inline uint32_t vx_num_threads() {
    uint32_t nt;
    asm volatile ("csrr %0, %1" : "=r"(nt) : "i"(0xFC0));
    return nt;
}

static inline uint32_t vx_num_warps() {
    uint32_t nw;
    asm volatile ("csrr %0, %1" : "=r"(nw) : "i"(0xFC1));
    return nw;
}

static inline uint32_t vx_num_cores() {
    uint32_t nc;
    asm volatile ("csrr %0, %1" : "=r"(nc) : "i"(0xFC2));
    return nc;
}

// Spawn API
static inline void vx_spawn_tasks(uint32_t num_tasks, vx_kernel_func_cb callback, void* arg) {
    // Simplified spawn - in real Vortex this distributes work across threads
    uint32_t tid = vx_thread_id();
    uint32_t num_threads = vx_num_threads();

    for (uint32_t i = tid; i < num_tasks; i += num_threads) {
        callback(arg);
    }
}

// Spawn with specific task count per warp
static inline void vx_spawn_warps(uint32_t num_warps, vx_kernel_func_cb callback, void* arg) {
    callback(arg);
}

// Vortex spawn threads API (main kernel entry point)
static inline int vx_spawn_threads(uint32_t dimension, const uint32_t* grid_size,
                                   uint32_t offset, vx_kernel_func_cb callback, void* arg) {
    callback(arg);
    return 0;
}

// Barriers
static inline void vx_warp_barrier() {
    asm volatile ("fence" ::: "memory");
}

static inline void vx_barrier(uint32_t barrierId, uint32_t count) {
    asm volatile (".insn s 0x6b, 1, %0, 0(%1)" :: "r"(count), "r"(barrierId));
}

// Atomic operations
static inline uint32_t vx_atomic_add(volatile uint32_t* addr, uint32_t value) {
    uint32_t result;
    asm volatile ("amoadd.w %0, %1, (%2)" : "=r"(result) : "r"(value), "r"(addr) : "memory");
    return result;
}

static inline uint32_t vx_atomic_swap(volatile uint32_t* addr, uint32_t value) {
    uint32_t result;
    asm volatile ("amoswap.w %0, %1, (%2)" : "=r"(result) : "r"(value), "r"(addr) : "memory");
    return result;
}

static inline uint32_t vx_atomic_max(volatile uint32_t* addr, uint32_t value) {
    uint32_t result;
    asm volatile ("amomaxu.w %0, %1, (%2)" : "=r"(result) : "r"(value), "r"(addr) : "memory");
    return result;
}

static inline uint32_t vx_atomic_min(volatile uint32_t* addr, uint32_t value) {
    uint32_t result;
    asm volatile ("amominu.w %0, %1, (%2)" : "=r"(result) : "r"(value), "r"(addr) : "memory");
    return result;
}

// TMC (Thread Mask Control) for divergent execution
static inline void vx_tmc(uint32_t mask) {
    asm volatile (".insn r 0x6b, 0, 0, x0, %0, x0" :: "r"(mask));
}

// Split/join for SIMT divergence
static inline void vx_split(uint32_t mask) {
    asm volatile (".insn r 0x6b, 2, 0, x0, %0, x0" :: "r"(mask));
}

static inline void vx_join() {
    asm volatile (".insn r 0x6b, 3, 0, x0, x0, x0");
}

// Prefetch
static inline void vx_prefetch(const void* addr) {
    asm volatile ("prefetch.r (%0)" :: "r"(addr));
}

#ifdef __cplusplus
}
#endif

#endif // VX_INTRINSICS_H
