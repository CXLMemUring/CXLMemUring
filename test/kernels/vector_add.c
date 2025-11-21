// Simple Vector Addition Kernel for Vortex SIMT
// This will be compiled to RISC-V ISA by the CIRA compiler

#include <stdint.h>

// Vortex intrinsics (would be provided by compiler)
extern uint32_t vx_thread_id();
extern uint32_t vx_num_threads();
extern void vx_barrier();

// Kernel entry point
// Each thread computes one element of the output vector
__attribute__((section(".text.kernel")))
void vector_add_kernel(const float* a, const float* b, float* c, uint32_t n) {
    // Get thread ID (SIMT model)
    uint32_t tid = vx_thread_id();
    uint32_t num_threads = vx_num_threads();

    // Grid-stride loop to handle cases where N > num_threads
    for (uint32_t i = tid; i < n; i += num_threads) {
        c[i] = a[i] + b[i];
    }

    // Synchronize all threads
    vx_barrier();
}

// Alternative: CUDA-like kernel with explicit grid/block indexing
__attribute__((section(".text.kernel")))
void vector_add_cuda_style(const float* a, const float* b, float* c, uint32_t n) {
    // These would be provided by the runtime/compiler
    extern uint32_t vx_block_id_x();
    extern uint32_t vx_thread_id_x();
    extern uint32_t vx_block_dim_x();

    uint32_t block_id = vx_block_id_x();
    uint32_t thread_id = vx_thread_id_x();
    uint32_t block_size = vx_block_dim_x();

    uint32_t i = block_id * block_size + thread_id;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Reduction kernel example - demonstrates warp-level operations
__attribute__((section(".text.kernel")))
float reduction_sum(const float* data, uint32_t n) {
    extern uint32_t vx_warp_id();
    extern uint32_t vx_lane_id();
    extern float vx_warp_reduce_add_f32(float value);

    uint32_t tid = vx_thread_id();
    uint32_t warp_id = vx_warp_id();
    uint32_t lane_id = vx_lane_id();

    // Each thread loads its element
    float value = (tid < n) ? data[tid] : 0.0f;

    // Warp-level reduction
    float warp_sum = vx_warp_reduce_add_f32(value);

    // First thread in each warp writes result
    __attribute__((shared)) float warp_sums[32];  // Assuming max 32 warps
    if (lane_id == 0) {
        warp_sums[warp_id] = warp_sum;
    }

    vx_barrier();

    // Final reduction by first warp
    if (warp_id == 0) {
        float final_value = (lane_id < 32) ? warp_sums[lane_id] : 0.0f;
        float total = vx_warp_reduce_add_f32(final_value);

        if (lane_id == 0) {
            return total;
        }
    }

    return 0.0f;
}
