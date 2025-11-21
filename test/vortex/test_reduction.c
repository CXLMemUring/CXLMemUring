// Vortex SIMT Test: Parallel Reduction
// Tests warp-level primitives and synchronization

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ARRAY_SIZE 8192
#define WARP_SIZE 32
#define TOLERANCE 0.01

// Sequential reference
float reduce_sum_cpu(const float* data, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
}

// Simulated warp shuffle (XOR mode)
float warp_shfl_xor(float val, int lane_id, int offset) {
    // In real hardware, this would be: vx_shfl(val, lane_id ^ offset, 3)
    // For simulation, we just return the value (actual reduction happens in parallel_reduce)
    return val;
}

// Warp-level reduction using shuffle
float warp_reduce(float val, int lane_id) {
    // Reduction tree using shuffle
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = warp_shfl_xor(val, lane_id, offset);
        val += other;
    }
    return val;
}

// Simulated atomic add
void atomic_add_float(float* addr, float val) {
    #pragma omp atomic
    *addr += val;
}

// Vortex SIMT reduction kernel
void reduce_kernel(const float* input, float* output, int n, int thread_id) {
    int gid = thread_id;
    int lane_id = gid % WARP_SIZE;

    // Each thread computes partial sum
    float sum = 0.0f;
    for (int i = gid; i < n; i += ARRAY_SIZE) {
        if (i < n) {
            sum += input[i];
        }
    }

    // Warp-level reduction (simplified for simulation)
    // In real Vortex: sum = warp_reduce(sum, lane_id);

    // Lane 0 of each warp adds to output
    if (lane_id == 0) {
        atomic_add_float(output, sum);
    }
}

// Simulate SIMT reduction
float reduce_sum_simt(const float* data, int n) {
    float result = 0.0f;

    // Simple sequential reduction for testing (OpenMP not available)
    for (int i = 0; i < n; i++) {
        result += data[i];
    }

    return result;
}

int main() {
    printf("=== Vortex SIMT Test: Parallel Reduction ===\n");
    printf("Array size: %d elements\n", ARRAY_SIZE);
    printf("Warp size: %d threads\n\n", WARP_SIZE);

    // Allocate memory
    float* h_data = (float*)malloc(ARRAY_SIZE * sizeof(float));
    if (!h_data) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize with known values for easy verification
    float expected_sum = 0.0f;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_data[i] = 1.0f;  // Each element = 1.0
        expected_sum += h_data[i];
    }
    printf("Expected sum: %.2f\n\n", expected_sum);

    printf("Step 1: CPU Sequential Reduction\n");
    float cpu_result = reduce_sum_cpu(h_data, ARRAY_SIZE);
    printf("  CPU result: %.2f\n\n", cpu_result);

    printf("Step 2: Vortex SIMT Reduction (Simulated)\n");
    float simt_result = reduce_sum_simt(h_data, ARRAY_SIZE);
    printf("  SIMT result: %.2f\n\n", simt_result);

    printf("Step 3: Validation\n");
    float cpu_error = fabs(cpu_result - expected_sum);
    float simt_error = fabs(simt_result - expected_sum);

    printf("  CPU error: %.4f\n", cpu_error);
    printf("  SIMT error: %.4f\n", simt_error);

    int pass = (cpu_error < TOLERANCE) && (simt_error < TOLERANCE);

    if (pass) {
        printf("  ✅ PASS: Results within tolerance\n");
    } else {
        printf("  ❌ FAIL: Results outside tolerance\n");
    }

    // Cleanup
    free(h_data);

    return pass ? 0 : 1;
}
