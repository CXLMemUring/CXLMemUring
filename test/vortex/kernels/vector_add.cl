// OpenCL Kernel for Vector Addition
// Will be compiled to Vortex RISC-V ISA and run on RTL simulator

__kernel void vector_add(__global const float* a,
                         __global const float* b,
                         __global float* c,
                         const unsigned int n) {
    // Get global thread ID
    int gid = get_global_id(0);

    // Bounds check
    if (gid < n) {
        c[gid] = a[gid] + b[gid];
    }
}
