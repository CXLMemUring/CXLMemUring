// OpenCL Kernel for Parallel Reduction
// Demonstrates warp-level operations on Vortex

__kernel void reduce_sum(__global const float* input,
                         __global float* output,
                         __local float* scratch,
                         const unsigned int n) {
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_size = get_local_size(0);

    // Load data into local memory with bounds check
    float sum = 0.0f;
    if (global_id < n) {
        sum = input[global_id];
    }

    // Store in local memory
    scratch[local_id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform reduction in local memory (tree-based)
    for (int offset = group_size / 2; offset > 0; offset >>= 1) {
        if (local_id < offset) {
            scratch[local_id] += scratch[local_id + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write result for this work-group
    if (local_id == 0) {
        atomic_add(output, scratch[0]);
    }
}
