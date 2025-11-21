// Vortex SIMT Test: Vector Addition
// Tests basic parallel execution and memory coalescing

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ARRAY_SIZE 1024
#define TOLERANCE 0.0001f

// Sequential reference implementation
void vector_add_cpu(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// Vortex SIMT kernel (will be transformed to NVVM IR)
void vector_add_kernel(const float* a, const float* b, float* c, int n, int thread_id) {
    // This will be converted to: tid = llvm.nvvm.read.ptx.sreg.tid.x()
    int gid = thread_id;

    if (gid < n) {
        c[gid] = a[gid] + b[gid];
    }
}

// Simulate SIMT execution (for testing without hardware)
void vector_add_simt(const float* a, const float* b, float* c, int n) {
    // Simple sequential execution for testing (OpenMP not available)
    for (int tid = 0; tid < n; tid++) {
        vector_add_kernel(a, b, c, n, tid);
    }
}

int validate_results(const float* expected, const float* actual, int n) {
    int errors = 0;
    for (int i = 0; i < n; i++) {
        float diff = expected[i] - actual[i];
        if (diff < 0) diff = -diff;

        if (diff > TOLERANCE) {
            if (errors < 10) {  // Only print first 10 errors
                printf("  Error at index %d: expected %.4f, got %.4f (diff: %.4f)\n",
                       i, expected[i], actual[i], diff);
            }
            errors++;
        }
    }
    return errors;
}

int main() {
    printf("=== Vortex SIMT Test: Vector Addition ===\n");
    printf("Array size: %d elements\n\n", ARRAY_SIZE);

    // Allocate memory
    float* h_a = (float*)malloc(ARRAY_SIZE * sizeof(float));
    float* h_b = (float*)malloc(ARRAY_SIZE * sizeof(float));
    float* h_c_cpu = (float*)malloc(ARRAY_SIZE * sizeof(float));
    float* h_c_simt = (float*)malloc(ARRAY_SIZE * sizeof(float));

    if (!h_a || !h_b || !h_c_cpu || !h_c_simt) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize input data
    srand(42);
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_a[i] = (float)(rand() % 1000) / 10.0f;
        h_b[i] = (float)(rand() % 1000) / 10.0f;
    }

    printf("Step 1: CPU Reference Implementation\n");
    clock_t start = clock();
    vector_add_cpu(h_a, h_b, h_c_cpu, ARRAY_SIZE);
    clock_t end = clock();
    double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    printf("  CPU time: %.3f ms\n\n", cpu_time);

    printf("Step 2: Vortex SIMT Implementation (Simulated)\n");
    start = clock();
    vector_add_simt(h_a, h_b, h_c_simt, ARRAY_SIZE);
    end = clock();
    double simt_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    printf("  SIMT time: %.3f ms\n\n", simt_time);

    printf("Step 3: Validation\n");
    int errors = validate_results(h_c_cpu, h_c_simt, ARRAY_SIZE);

    if (errors == 0) {
        printf("  ✅ PASS: All %d results match!\n", ARRAY_SIZE);
        printf("  Speedup: %.2fx\n", cpu_time / simt_time);
    } else {
        printf("  ❌ FAIL: %d/%d results incorrect\n", errors, ARRAY_SIZE);
    }

    // Cleanup
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_simt);

    return (errors == 0) ? 0 : 1;
}
