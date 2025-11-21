// vortex_protocol.h
// Vortex RISC-V SIMT Protocol Definitions
// Provides CUDA-like kernel launch interface for Vortex GPUs

#ifndef VORTEX_PROTOCOL_H
#define VORTEX_PROTOCOL_H

#include <stdint.h>
#include <cstddef>
#include "shared_protocol.h"

// CUDA-like dim3 structure for grid/block dimensions
struct dim3 {
    uint32_t x, y, z;

    dim3(uint32_t x_ = 1, uint32_t y_ = 1, uint32_t z_ = 1)
        : x(x_), y(y_), z(z_) {}
};

// Extended operation types for Vortex SIMT
enum VortexOperationType {
    OP_BATCH_INSERT = 10,      // Batched insert (SIMT kernel)
    OP_BATCH_DELETE = 11,      // Batched delete (SIMT kernel)
    OP_BATCH_SEARCH = 12,      // Batched search (SIMT kernel)
    OP_KERNEL_LAUNCH = 13,     // Generic kernel launch
    OP_GRAPH_TRAVERSE = 14,    // Graph traversal kernel
    OP_VECTOR_ADD = 15,        // Vector addition (example)
    OP_REDUCE = 16,            // Parallel reduction
    OP_SCAN = 17,              // Parallel prefix scan
    OP_SORT = 18              // Parallel sort
};

// Kernel parameters for batched insert
struct VortexInsertParams {
    BTreeNode* nodes;          // Array of nodes to insert into
    int* keys;                 // Array of keys to insert
    uint32_t num_keys;         // Number of insertions
    uint32_t* results;         // Output status for each insert
};

// Kernel parameters for batched search
struct VortexSearchParams {
    const BTreeNode* nodes;    // Array of nodes to search
    int* keys;                 // Array of keys to search for
    uint32_t num_queries;      // Number of queries
    int* results;              // Output indices (-1 if not found)
};

// Kernel parameters for batched delete
struct VortexDeleteParams {
    BTreeNode* nodes;
    int* keys;
    uint32_t num_keys;
    uint32_t* results;
};

// Generic kernel launch parameters
struct VortexKernelLaunch {
    dim3 grid_dim;             // Grid dimensions (blocks)
    dim3 block_dim;            // Block dimensions (threads per block)
    void* kernel_func;         // Pointer to kernel function
    void* kernel_args;         // Pointer to kernel arguments
    size_t shared_mem_size;    // Shared memory per block (bytes)
    uint64_t stream_id;        // Stream for asynchronous execution
};

// Graph traversal kernel parameters
struct VortexGraphParams {
    struct Edge {
        uint32_t from;
        uint32_t to;
        float weight;
    };

    struct Vertex {
        uint32_t id;
        float value;
        uint32_t degree;
        uint32_t edge_offset;  // Offset into edge array
    };

    Edge* edges;               // CSR edge array
    Vertex* vertices;          // Vertex array
    uint32_t num_vertices;
    uint32_t num_edges;
    uint32_t* frontier;        // BFS/DFS frontier queue
    uint32_t* visited;         // Visited bitmap
    float* results;            // Output per vertex
};

// Vector operation parameters
struct VortexVectorParams {
    float* input_a;
    float* input_b;
    float* output;
    uint32_t num_elements;
};

// Reduction operation parameters
struct VortexReduceParams {
    enum Op { SUM, MIN, MAX, PROD };

    float* input;
    float* output;
    uint32_t num_elements;
    Op operation;
};

// Scan (prefix sum) parameters
struct VortexScanParams {
    enum Op { SUM, MIN, MAX };

    float* input;
    float* output;
    uint32_t num_elements;
    Op operation;
    bool inclusive;            // Inclusive vs exclusive scan
};

// Sort parameters
struct VortexSortParams {
    int* keys;
    int* values;               // Optional payload
    uint32_t num_elements;
    bool ascending;
};

// Extended request structure for Vortex operations
struct VortexOpRequest {
    uint8_t op_type;           // VortexOperationType
    uint64_t kernel_id;        // Unique kernel identifier
    uint64_t stream_id;        // Stream for async execution

    union {
        VortexKernelLaunch launch;
        VortexInsertParams insert;
        VortexSearchParams search;
        VortexDeleteParams delete_op;
        VortexGraphParams graph;
        VortexVectorParams vector;
        VortexReduceParams reduce;
        VortexScanParams scan;
        VortexSortParams sort;
    } params;
};

// Kernel execution status
struct VortexKernelStatus {
    uint64_t kernel_id;
    enum State {
        PENDING = 0,
        RUNNING = 1,
        COMPLETED = 2,
        FAILED = 3
    } state;

    uint32_t warps_executed;
    uint32_t threads_executed;
    uint64_t cycles;           // Execution time in cycles
    uint32_t divergent_warps;  // Number of warps with divergence
    float efficiency;          // Warp execution efficiency (0-1)
};

// Vortex runtime configuration
struct VortexConfig {
    uint32_t warp_size;        // Threads per warp (default: 32)
    uint32_t max_warps;        // Maximum concurrent warps
    uint32_t max_threads;      // Total thread capacity
    uint32_t max_blocks;       // Maximum blocks per grid
    uint32_t shared_mem_per_block;  // Shared memory size
    uint32_t l1_cache_size;    // L1 cache size (bytes)
    uint32_t l2_cache_size;    // L2 cache size (bytes)
    bool enable_prefetch;      // Hardware prefetcher
    bool enable_coalescing;    // Memory coalescing
};

// Vortex performance counters
struct VortexPerfCounters {
    uint64_t total_kernels;
    uint64_t total_warps;
    uint64_t total_threads;
    uint64_t total_cycles;
    uint64_t memory_transactions;
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t branch_divergences;
    float avg_warp_efficiency;
    float avg_memory_efficiency;
};

// Warp intrinsics (mirroring CUDA)
namespace vortex_intrinsics {
    // Thread indexing
    extern "C" uint32_t __vx_thread_id();      // Global thread ID
    extern "C" uint32_t __vx_warp_id();        // Warp ID
    extern "C" uint32_t __vx_lane_id();        // Lane ID within warp
    extern "C" uint32_t __vx_block_id();       // Block ID

    // Synchronization
    extern "C" void __vx_barrier();            // Block-level barrier
    extern "C" void __vx_warp_barrier();       // Warp-level barrier
    extern "C" void __vx_fence();              // Memory fence

    // Warp-level primitives
    extern "C" uint32_t __vx_ballot(int predicate);           // Ballot vote
    extern "C" uint32_t __vx_all(int predicate);              // All threads true
    extern "C" uint32_t __vx_any(int predicate);              // Any thread true
    extern "C" uint32_t __vx_shfl(uint32_t var, int lane);    // Warp shuffle
    extern "C" uint32_t __vx_shfl_xor(uint32_t var, int mask);// Shuffle XOR

    // Atomic operations
    extern "C" int __vx_atomic_add(int* addr, int val);
    extern "C" int __vx_atomic_cas(int* addr, int compare, int val);
    extern "C" int __vx_atomic_exch(int* addr, int val);
    extern "C" int __vx_atomic_min(int* addr, int val);
    extern "C" int __vx_atomic_max(int* addr, int val);

    // Vector memory operations
    extern "C" void __vx_vector_load(void* dst, const void* src, size_t size);
    extern "C" void __vx_vector_store(void* dst, const void* src, size_t size);

    // Prefetch hints
    extern "C" void __vx_prefetch_l1(const void* addr);
    extern "C" void __vx_prefetch_l2(const void* addr);

    // Performance counters
    extern "C" uint64_t __vx_read_cycle();
    extern "C" uint64_t __vx_read_instret();
}

// Helper macros for kernel definition
#define VORTEX_KERNEL __attribute__((noinline))
#define VORTEX_DEVICE __attribute__((always_inline))
#define VORTEX_HOST
#define VORTEX_GLOBAL __attribute__((address_space(1)))
#define VORTEX_SHARED __attribute__((address_space(3)))

// Memory space qualifiers
#define VORTEX_CONSTANT __attribute__((address_space(2)))
#define VORTEX_LOCAL __attribute__((address_space(4)))

// Launch configuration helpers
#define VORTEX_THREADS_PER_BLOCK 256
#define VORTEX_NUM_BLOCKS(n) (((n) + VORTEX_THREADS_PER_BLOCK - 1) / VORTEX_THREADS_PER_BLOCK)

// Kernel launch macro (similar to CUDA <<<>>> syntax)
#define VORTEX_LAUNCH(kernel, grid, block, args) \
    vortex_launch_kernel((void*)kernel, grid, block, args)

// Runtime API functions
extern "C" {
    // Device management
    int vortex_device_count();
    int vortex_device_get(int device_id);
    int vortex_device_reset(int device_id);
    int vortex_device_sync(int device_id);

    // Memory management
    void* vortex_malloc(size_t size);
    void vortex_free(void* ptr);
    int vortex_memcpy_h2d(void* dst, const void* src, size_t size);
    int vortex_memcpy_d2h(void* dst, const void* src, size_t size);
    int vortex_memcpy_d2d(void* dst, const void* src, size_t size);

    // Kernel launch
    int vortex_launch_kernel(void* kernel, dim3 grid, dim3 block, void* args);
    int vortex_launch_async(void* kernel, dim3 grid, dim3 block, void* args, uint64_t stream);

    // Stream management
    uint64_t vortex_stream_create();
    int vortex_stream_destroy(uint64_t stream);
    int vortex_stream_sync(uint64_t stream);
    int vortex_stream_query(uint64_t stream);

    // Performance monitoring
    int vortex_get_perf_counters(VortexPerfCounters* counters);
    int vortex_reset_perf_counters();
    int vortex_get_kernel_status(uint64_t kernel_id, VortexKernelStatus* status);

    // Configuration
    int vortex_get_config(VortexConfig* config);
    int vortex_set_config(const VortexConfig* config);
}

#endif // VORTEX_PROTOCOL_H
