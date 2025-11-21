// Simple vector addition kernel for Vortex GPGPU
// Demonstrates basic SIMT execution and memory access

#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <stdint.h>

// Kernel argument structure
typedef struct {
    uint64_t a_addr;      // Input array A address
    uint64_t b_addr;      // Input array B address
    uint64_t c_addr;      // Output array C address
    uint32_t num_elements; // Number of elements
} kernel_arg_t;

// Kernel body executed by each thread
void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    // Get pointers from device addresses
    float* a = (float*)arg->a_addr;
    float* b = (float*)arg->b_addr;
    float* c = (float*)arg->c_addr;
    uint32_t n = arg->num_elements;

    // Get thread ID and total thread count
    uint32_t tid = vx_thread_id();
    uint32_t num_threads = vx_num_threads();

    // Each thread processes multiple elements in a strided pattern
    for (uint32_t i = tid; i < n; i += num_threads) {
        c[i] = a[i] + b[i];
    }

    // Synchronize all threads (barrier id 0, 4 warps based on Vortex config)
    vx_barrier(0, 4);
}

// Main entry point (required for Vortex kernel)
int main() {
    // Get kernel arguments from CSR
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);

    // Spawn threads and execute kernel body
    // Parameters: num_groups, local_work_size, global_work_offset, kernel_func, args
    return vx_spawn_threads(1, &arg->num_elements, nullptr,
                           (vx_kernel_func_cb)kernel_body, arg);
}
