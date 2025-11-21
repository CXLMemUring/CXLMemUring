// Offload Profiler - CPU-side timing for prefetcher distance optimization
// Measures H2D/D2H transfer latencies and kernel execution timing

#ifndef OFFLOAD_PROFILER_H
#define OFFLOAD_PROFILER_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

// Timing data for a single offload operation
typedef struct {
    // Transfer sizes
    size_t h2d_bytes;
    size_t d2h_bytes;

    // Timing (nanoseconds)
    uint64_t h2d_latency_ns;
    uint64_t kernel_latency_ns;
    uint64_t d2h_latency_ns;
    uint64_t total_latency_ns;

    // Derived metrics
    double h2d_bandwidth_gbps;
    double d2h_bandwidth_gbps;
    double kernel_throughput;  // elements/second

    // For prefetcher tuning
    uint64_t optimal_prefetch_distance_bytes;
    uint64_t min_overlap_window_ns;
} offload_timing_t;

// Profile collection context
typedef struct {
    struct timespec start_time;
    struct timespec h2d_start;
    struct timespec h2d_end;
    struct timespec kernel_start;
    struct timespec kernel_end;
    struct timespec d2h_start;
    struct timespec d2h_end;

    size_t h2d_bytes;
    size_t d2h_bytes;
    size_t num_elements;
} offload_profile_ctx_t;

// Start profiling an offload operation
static inline void offload_profile_start(offload_profile_ctx_t* ctx) {
    clock_gettime(CLOCK_MONOTONIC, &ctx->start_time);
    ctx->h2d_bytes = 0;
    ctx->d2h_bytes = 0;
    ctx->num_elements = 0;
}

// Mark H2D transfer start
static inline void offload_profile_h2d_start(offload_profile_ctx_t* ctx, size_t bytes) {
    clock_gettime(CLOCK_MONOTONIC, &ctx->h2d_start);
    ctx->h2d_bytes = bytes;
}

// Mark H2D transfer end
static inline void offload_profile_h2d_end(offload_profile_ctx_t* ctx) {
    clock_gettime(CLOCK_MONOTONIC, &ctx->h2d_end);
}

// Mark kernel execution start
static inline void offload_profile_kernel_start(offload_profile_ctx_t* ctx, size_t num_elements) {
    clock_gettime(CLOCK_MONOTONIC, &ctx->kernel_start);
    ctx->num_elements = num_elements;
}

// Mark kernel execution end
static inline void offload_profile_kernel_end(offload_profile_ctx_t* ctx) {
    clock_gettime(CLOCK_MONOTONIC, &ctx->kernel_end);
}

// Mark D2H transfer start
static inline void offload_profile_d2h_start(offload_profile_ctx_t* ctx, size_t bytes) {
    clock_gettime(CLOCK_MONOTONIC, &ctx->d2h_start);
    ctx->d2h_bytes = bytes;
}

// Mark D2H transfer end
static inline void offload_profile_d2h_end(offload_profile_ctx_t* ctx) {
    clock_gettime(CLOCK_MONOTONIC, &ctx->d2h_end);
}

// Calculate timing results
static inline void offload_profile_finish(offload_profile_ctx_t* ctx, offload_timing_t* timing) {
    // Helper to convert timespec diff to nanoseconds
    #define TIMESPEC_TO_NS(ts) ((uint64_t)(ts).tv_sec * 1000000000ULL + (ts).tv_nsec)
    #define TIMESPEC_DIFF_NS(end, start) (TIMESPEC_TO_NS(end) - TIMESPEC_TO_NS(start))

    timing->h2d_bytes = ctx->h2d_bytes;
    timing->d2h_bytes = ctx->d2h_bytes;

    timing->h2d_latency_ns = TIMESPEC_DIFF_NS(ctx->h2d_end, ctx->h2d_start);
    timing->kernel_latency_ns = TIMESPEC_DIFF_NS(ctx->kernel_end, ctx->kernel_start);
    timing->d2h_latency_ns = TIMESPEC_DIFF_NS(ctx->d2h_end, ctx->d2h_start);
    timing->total_latency_ns = TIMESPEC_DIFF_NS(ctx->d2h_end, ctx->start_time);

    // Calculate bandwidth (GB/s)
    if (timing->h2d_latency_ns > 0) {
        timing->h2d_bandwidth_gbps = (double)ctx->h2d_bytes / timing->h2d_latency_ns;
    } else {
        timing->h2d_bandwidth_gbps = 0;
    }

    if (timing->d2h_latency_ns > 0) {
        timing->d2h_bandwidth_gbps = (double)ctx->d2h_bytes / timing->d2h_latency_ns;
    } else {
        timing->d2h_bandwidth_gbps = 0;
    }

    // Kernel throughput
    if (timing->kernel_latency_ns > 0) {
        timing->kernel_throughput = (double)ctx->num_elements * 1e9 / timing->kernel_latency_ns;
    } else {
        timing->kernel_throughput = 0;
    }

    // Calculate optimal prefetch distance
    // Prefetch distance = bytes that can be transferred during kernel execution
    double avg_bandwidth = (timing->h2d_bandwidth_gbps + timing->d2h_bandwidth_gbps) / 2.0;
    if (avg_bandwidth > 0) {
        timing->optimal_prefetch_distance_bytes =
            (uint64_t)(avg_bandwidth * timing->kernel_latency_ns);
    } else {
        timing->optimal_prefetch_distance_bytes = ctx->h2d_bytes;
    }

    // Minimum overlap window = time needed for smallest useful prefetch
    // Assume 64-byte cache line as minimum
    if (avg_bandwidth > 0) {
        timing->min_overlap_window_ns = (uint64_t)(64.0 / avg_bandwidth);
    } else {
        timing->min_overlap_window_ns = 1000; // 1us default
    }

    #undef TIMESPEC_TO_NS
    #undef TIMESPEC_DIFF_NS
}

// Output timing to JSON for compiler consumption
static inline void offload_profile_to_json(const offload_timing_t* timing,
                                           const char* kernel_name,
                                           const char* output_path) {
    FILE* f = fopen(output_path, "w");
    if (!f) return;

    fprintf(f, "{\n");
    fprintf(f, "  \"kernel\": \"%s\",\n", kernel_name);
    fprintf(f, "  \"timing\": {\n");
    fprintf(f, "    \"h2d_bytes\": %zu,\n", timing->h2d_bytes);
    fprintf(f, "    \"d2h_bytes\": %zu,\n", timing->d2h_bytes);
    fprintf(f, "    \"h2d_latency_ns\": %lu,\n", timing->h2d_latency_ns);
    fprintf(f, "    \"kernel_latency_ns\": %lu,\n", timing->kernel_latency_ns);
    fprintf(f, "    \"d2h_latency_ns\": %lu,\n", timing->d2h_latency_ns);
    fprintf(f, "    \"total_latency_ns\": %lu\n", timing->total_latency_ns);
    fprintf(f, "  },\n");
    fprintf(f, "  \"bandwidth\": {\n");
    fprintf(f, "    \"h2d_gbps\": %.6f,\n", timing->h2d_bandwidth_gbps);
    fprintf(f, "    \"d2h_gbps\": %.6f\n", timing->d2h_bandwidth_gbps);
    fprintf(f, "  },\n");
    fprintf(f, "  \"prefetch_hints\": {\n");
    fprintf(f, "    \"optimal_distance_bytes\": %lu,\n", timing->optimal_prefetch_distance_bytes);
    fprintf(f, "    \"min_overlap_window_ns\": %lu,\n", timing->min_overlap_window_ns);
    fprintf(f, "    \"kernel_throughput_eps\": %.2f\n", timing->kernel_throughput);
    fprintf(f, "  }\n");
    fprintf(f, "}\n");

    fclose(f);
}

// Print timing summary to stdout
static inline void offload_profile_print(const offload_timing_t* timing, const char* kernel_name) {
    printf("\n=== Offload Timing Profile: %s ===\n", kernel_name);
    printf("Transfer Sizes:\n");
    printf("  H2D: %zu bytes\n", timing->h2d_bytes);
    printf("  D2H: %zu bytes\n", timing->d2h_bytes);
    printf("\nLatencies:\n");
    printf("  H2D Transfer:    %10lu ns  (%.3f ms)\n",
           timing->h2d_latency_ns, timing->h2d_latency_ns / 1e6);
    printf("  Kernel Exec:     %10lu ns  (%.3f ms)\n",
           timing->kernel_latency_ns, timing->kernel_latency_ns / 1e6);
    printf("  D2H Transfer:    %10lu ns  (%.3f ms)\n",
           timing->d2h_latency_ns, timing->d2h_latency_ns / 1e6);
    printf("  Total:           %10lu ns  (%.3f ms)\n",
           timing->total_latency_ns, timing->total_latency_ns / 1e6);
    printf("\nBandwidth:\n");
    printf("  H2D: %.4f GB/s\n", timing->h2d_bandwidth_gbps);
    printf("  D2H: %.4f GB/s\n", timing->d2h_bandwidth_gbps);
    printf("\nPrefetcher Optimization Hints:\n");
    printf("  Optimal prefetch distance: %lu bytes\n", timing->optimal_prefetch_distance_bytes);
    printf("  Min overlap window:        %lu ns\n", timing->min_overlap_window_ns);
    printf("  Kernel throughput:         %.2f elements/sec\n", timing->kernel_throughput);
    printf("==========================================\n\n");
}

#ifdef __cplusplus
}
#endif

#endif // OFFLOAD_PROFILER_H
