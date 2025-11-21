// MCF Profiler - CPU-side timing for MCF hotspot analysis and offloading decisions
// Profiles primal_bea_mpp, price_out_impl, and flow_cost for x86 baseline and Vortex offload comparison

#ifndef MCF_PROFILER_H
#define MCF_PROFILER_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#ifdef __cplusplus
extern "C" {
#endif

// MCF-specific function profiling results
typedef struct {
    // Function invocation counts
    uint64_t primal_bea_mpp_calls;
    uint64_t price_out_impl_calls;
    uint64_t flow_cost_calls;
    uint64_t refresh_potential_calls;

    // Total time per function (nanoseconds)
    uint64_t primal_bea_mpp_total_ns;
    uint64_t price_out_impl_total_ns;
    uint64_t flow_cost_total_ns;
    uint64_t refresh_potential_total_ns;

    // Arc/element counts processed
    uint64_t total_arcs_priced;
    uint64_t total_implicit_arcs;
    uint64_t total_flow_arcs;

    // Average time per call (derived)
    double primal_bea_mpp_avg_ns;
    double price_out_impl_avg_ns;
    double flow_cost_avg_ns;

    // Throughput metrics
    double arcs_per_sec_pricing;
    double arcs_per_sec_implicit;
    double arcs_per_sec_flow;

    // Overall execution
    uint64_t total_simplex_iterations;
    uint64_t total_execution_ns;
} mcf_profile_result_t;

// Profile collection context for MCF
typedef struct {
    struct timespec func_start;
    struct timespec execution_start;

    // Accumulation
    uint64_t primal_bea_mpp_calls;
    uint64_t price_out_impl_calls;
    uint64_t flow_cost_calls;
    uint64_t refresh_potential_calls;

    uint64_t primal_bea_mpp_total_ns;
    uint64_t price_out_impl_total_ns;
    uint64_t flow_cost_total_ns;
    uint64_t refresh_potential_total_ns;

    uint64_t total_arcs_priced;
    uint64_t total_implicit_arcs;
    uint64_t total_flow_arcs;
    uint64_t total_simplex_iterations;

    // Offload tracking
    uint64_t offload_count;
    uint64_t offload_total_h2d_ns;
    uint64_t offload_total_kernel_ns;
    uint64_t offload_total_d2h_ns;
    uint64_t offload_total_bytes_h2d;
    uint64_t offload_total_bytes_d2h;
} mcf_profile_ctx_t;

// Global profiler context (extern to share across translation units)
extern mcf_profile_ctx_t __mcf_profile_ctx;
extern int __mcf_profiling_enabled;

// Define the globals in the compilation unit that defines MCF_PROFILER_MAIN
#ifdef MCF_PROFILER_MAIN
mcf_profile_ctx_t __mcf_profile_ctx = {0};
int __mcf_profiling_enabled = 0;
#endif

// Helper to convert timespec diff to nanoseconds
static inline uint64_t __mcf_timespec_diff_ns(struct timespec* end, struct timespec* start) {
    uint64_t end_ns = (uint64_t)end->tv_sec * 1000000000ULL + end->tv_nsec;
    uint64_t start_ns = (uint64_t)start->tv_sec * 1000000000ULL + start->tv_nsec;
    return end_ns - start_ns;
}

// Initialize profiling
static inline void mcf_profile_init(void) {
    __mcf_profiling_enabled = 1;
    memset(&__mcf_profile_ctx, 0, sizeof(mcf_profile_ctx_t));
    clock_gettime(CLOCK_MONOTONIC, &__mcf_profile_ctx.execution_start);
}

// Function entry/exit markers
static inline void mcf_profile_primal_bea_mpp_start(void) {
    if (!__mcf_profiling_enabled) return;
    clock_gettime(CLOCK_MONOTONIC, &__mcf_profile_ctx.func_start);
}

static inline void mcf_profile_primal_bea_mpp_end(long arcs_priced) {
    if (!__mcf_profiling_enabled) return;
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    __mcf_profile_ctx.primal_bea_mpp_calls++;
    __mcf_profile_ctx.primal_bea_mpp_total_ns += __mcf_timespec_diff_ns(&end, &__mcf_profile_ctx.func_start);
    __mcf_profile_ctx.total_arcs_priced += arcs_priced;
}

static inline void mcf_profile_price_out_impl_start(void) {
    if (!__mcf_profiling_enabled) return;
    clock_gettime(CLOCK_MONOTONIC, &__mcf_profile_ctx.func_start);
}

static inline void mcf_profile_price_out_impl_end(long new_arcs) {
    if (!__mcf_profiling_enabled) return;
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    __mcf_profile_ctx.price_out_impl_calls++;
    __mcf_profile_ctx.price_out_impl_total_ns += __mcf_timespec_diff_ns(&end, &__mcf_profile_ctx.func_start);
    __mcf_profile_ctx.total_implicit_arcs += new_arcs;
}

static inline void mcf_profile_flow_cost_start(void) {
    if (!__mcf_profiling_enabled) return;
    clock_gettime(CLOCK_MONOTONIC, &__mcf_profile_ctx.func_start);
}

static inline void mcf_profile_flow_cost_end(long num_arcs) {
    if (!__mcf_profiling_enabled) return;
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    __mcf_profile_ctx.flow_cost_calls++;
    __mcf_profile_ctx.flow_cost_total_ns += __mcf_timespec_diff_ns(&end, &__mcf_profile_ctx.func_start);
    __mcf_profile_ctx.total_flow_arcs += num_arcs;
}

static inline void mcf_profile_refresh_potential_start(void) {
    if (!__mcf_profiling_enabled) return;
    clock_gettime(CLOCK_MONOTONIC, &__mcf_profile_ctx.func_start);
}

static inline void mcf_profile_refresh_potential_end(void) {
    if (!__mcf_profiling_enabled) return;
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    __mcf_profile_ctx.refresh_potential_calls++;
    __mcf_profile_ctx.refresh_potential_total_ns += __mcf_timespec_diff_ns(&end, &__mcf_profile_ctx.func_start);
}

static inline void mcf_profile_simplex_iteration(void) {
    if (!__mcf_profiling_enabled) return;
    __mcf_profile_ctx.total_simplex_iterations++;
}

// Offload tracking
static inline void mcf_profile_offload_h2d(uint64_t bytes, uint64_t latency_ns) {
    if (!__mcf_profiling_enabled) return;
    __mcf_profile_ctx.offload_total_h2d_ns += latency_ns;
    __mcf_profile_ctx.offload_total_bytes_h2d += bytes;
}

static inline void mcf_profile_offload_kernel(uint64_t latency_ns) {
    if (!__mcf_profiling_enabled) return;
    __mcf_profile_ctx.offload_total_kernel_ns += latency_ns;
    __mcf_profile_ctx.offload_count++;
}

static inline void mcf_profile_offload_d2h(uint64_t bytes, uint64_t latency_ns) {
    if (!__mcf_profiling_enabled) return;
    __mcf_profile_ctx.offload_total_d2h_ns += latency_ns;
    __mcf_profile_ctx.offload_total_bytes_d2h += bytes;
}

// Finalize and compute results
static inline void mcf_profile_finish(mcf_profile_result_t* result) {
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);

    result->primal_bea_mpp_calls = __mcf_profile_ctx.primal_bea_mpp_calls;
    result->price_out_impl_calls = __mcf_profile_ctx.price_out_impl_calls;
    result->flow_cost_calls = __mcf_profile_ctx.flow_cost_calls;
    result->refresh_potential_calls = __mcf_profile_ctx.refresh_potential_calls;

    result->primal_bea_mpp_total_ns = __mcf_profile_ctx.primal_bea_mpp_total_ns;
    result->price_out_impl_total_ns = __mcf_profile_ctx.price_out_impl_total_ns;
    result->flow_cost_total_ns = __mcf_profile_ctx.flow_cost_total_ns;
    result->refresh_potential_total_ns = __mcf_profile_ctx.refresh_potential_total_ns;

    result->total_arcs_priced = __mcf_profile_ctx.total_arcs_priced;
    result->total_implicit_arcs = __mcf_profile_ctx.total_implicit_arcs;
    result->total_flow_arcs = __mcf_profile_ctx.total_flow_arcs;

    result->total_simplex_iterations = __mcf_profile_ctx.total_simplex_iterations;
    result->total_execution_ns = __mcf_timespec_diff_ns(&end, &__mcf_profile_ctx.execution_start);

    // Compute averages
    if (result->primal_bea_mpp_calls > 0) {
        result->primal_bea_mpp_avg_ns = (double)result->primal_bea_mpp_total_ns / result->primal_bea_mpp_calls;
    }
    if (result->price_out_impl_calls > 0) {
        result->price_out_impl_avg_ns = (double)result->price_out_impl_total_ns / result->price_out_impl_calls;
    }
    if (result->flow_cost_calls > 0) {
        result->flow_cost_avg_ns = (double)result->flow_cost_total_ns / result->flow_cost_calls;
    }

    // Compute throughput
    if (result->primal_bea_mpp_total_ns > 0) {
        result->arcs_per_sec_pricing = (double)result->total_arcs_priced * 1e9 / result->primal_bea_mpp_total_ns;
    }
    if (result->price_out_impl_total_ns > 0) {
        result->arcs_per_sec_implicit = (double)result->total_implicit_arcs * 1e9 / result->price_out_impl_total_ns;
    }
    if (result->flow_cost_total_ns > 0) {
        result->arcs_per_sec_flow = (double)result->total_flow_arcs * 1e9 / result->flow_cost_total_ns;
    }
}

// Output to JSON for compiler consumption
static inline void mcf_profile_to_json(const mcf_profile_result_t* result, const char* output_path) {
    FILE* f = fopen(output_path, "w");
    if (!f) return;

    fprintf(f, "{\n");
    fprintf(f, "  \"profile_type\": \"mcf_baseline\",\n");
    fprintf(f, "  \"target\": \"x86_64\",\n");
    fprintf(f, "  \"functions\": {\n");

    // primal_bea_mpp
    fprintf(f, "    \"primal_bea_mpp\": {\n");
    fprintf(f, "      \"calls\": %lu,\n", result->primal_bea_mpp_calls);
    fprintf(f, "      \"total_ns\": %lu,\n", result->primal_bea_mpp_total_ns);
    fprintf(f, "      \"avg_ns\": %.2f,\n", result->primal_bea_mpp_avg_ns);
    fprintf(f, "      \"arcs_processed\": %lu,\n", result->total_arcs_priced);
    fprintf(f, "      \"throughput_arcs_per_sec\": %.2f,\n", result->arcs_per_sec_pricing);
    fprintf(f, "      \"offload_candidate\": true,\n");
    fprintf(f, "      \"parallelism\": \"embarrassingly_parallel\"\n");
    fprintf(f, "    },\n");

    // price_out_impl
    fprintf(f, "    \"price_out_impl\": {\n");
    fprintf(f, "      \"calls\": %lu,\n", result->price_out_impl_calls);
    fprintf(f, "      \"total_ns\": %lu,\n", result->price_out_impl_total_ns);
    fprintf(f, "      \"avg_ns\": %.2f,\n", result->price_out_impl_avg_ns);
    fprintf(f, "      \"arcs_generated\": %lu,\n", result->total_implicit_arcs);
    fprintf(f, "      \"throughput_arcs_per_sec\": %.2f,\n", result->arcs_per_sec_implicit);
    fprintf(f, "      \"offload_candidate\": true,\n");
    fprintf(f, "      \"parallelism\": \"data_dependent\"\n");
    fprintf(f, "    },\n");

    // flow_cost
    fprintf(f, "    \"flow_cost\": {\n");
    fprintf(f, "      \"calls\": %lu,\n", result->flow_cost_calls);
    fprintf(f, "      \"total_ns\": %lu,\n", result->flow_cost_total_ns);
    fprintf(f, "      \"avg_ns\": %.2f,\n", result->flow_cost_avg_ns);
    fprintf(f, "      \"arcs_processed\": %lu,\n", result->total_flow_arcs);
    fprintf(f, "      \"throughput_arcs_per_sec\": %.2f,\n", result->arcs_per_sec_flow);
    fprintf(f, "      \"offload_candidate\": true,\n");
    fprintf(f, "      \"parallelism\": \"reduction\"\n");
    fprintf(f, "    },\n");

    // refresh_potential
    fprintf(f, "    \"refresh_potential\": {\n");
    fprintf(f, "      \"calls\": %lu,\n", result->refresh_potential_calls);
    fprintf(f, "      \"total_ns\": %lu,\n", result->refresh_potential_total_ns);
    fprintf(f, "      \"offload_candidate\": false,\n");
    fprintf(f, "      \"parallelism\": \"tree_traversal\"\n");
    fprintf(f, "    }\n");
    fprintf(f, "  },\n");

    // Overall stats
    fprintf(f, "  \"overall\": {\n");
    fprintf(f, "    \"simplex_iterations\": %lu,\n", result->total_simplex_iterations);
    fprintf(f, "    \"total_execution_ns\": %lu,\n", result->total_execution_ns);
    fprintf(f, "    \"total_execution_ms\": %.2f\n", result->total_execution_ns / 1e6);
    fprintf(f, "  },\n");

    // Offload hints for compiler
    fprintf(f, "  \"offload_hints\": {\n");
    fprintf(f, "    \"primary_target\": \"primal_bea_mpp\",\n");
    fprintf(f, "    \"secondary_target\": \"price_out_impl\",\n");
    fprintf(f, "    \"expected_speedup_pricing\": %.1f,\n",
            result->primal_bea_mpp_total_ns > 0 ? 10.0 : 1.0); // Estimated from parallel nature
    fprintf(f, "    \"min_arcs_for_offload\": 1000,\n");
    fprintf(f, "    \"data_transfer_cost_factor\": 0.1\n");
    fprintf(f, "  }\n");

    fprintf(f, "}\n");
    fclose(f);
}

// Print summary
static inline void mcf_profile_print(const mcf_profile_result_t* result) {
    printf("\n=== MCF Profiling Results (x86 Baseline) ===\n\n");

    printf("Function Timings:\n");
    printf("  primal_bea_mpp:\n");
    printf("    Calls: %lu, Total: %.2f ms, Avg: %.2f us\n",
           result->primal_bea_mpp_calls,
           result->primal_bea_mpp_total_ns / 1e6,
           result->primal_bea_mpp_avg_ns / 1e3);
    printf("    Arcs priced: %lu, Throughput: %.2f M arcs/sec\n",
           result->total_arcs_priced,
           result->arcs_per_sec_pricing / 1e6);

    printf("  price_out_impl:\n");
    printf("    Calls: %lu, Total: %.2f ms, Avg: %.2f us\n",
           result->price_out_impl_calls,
           result->price_out_impl_total_ns / 1e6,
           result->price_out_impl_avg_ns / 1e3);
    printf("    Implicit arcs: %lu, Throughput: %.2f M arcs/sec\n",
           result->total_implicit_arcs,
           result->arcs_per_sec_implicit / 1e6);

    printf("  flow_cost:\n");
    printf("    Calls: %lu, Total: %.2f ms, Avg: %.2f us\n",
           result->flow_cost_calls,
           result->flow_cost_total_ns / 1e6,
           result->flow_cost_avg_ns / 1e3);

    printf("  refresh_potential:\n");
    printf("    Calls: %lu, Total: %.2f ms\n",
           result->refresh_potential_calls,
           result->refresh_potential_total_ns / 1e6);

    printf("\nOverall:\n");
    printf("  Simplex iterations: %lu\n", result->total_simplex_iterations);
    printf("  Total execution time: %.2f ms\n", result->total_execution_ns / 1e6);

    // Breakdown
    double total = result->primal_bea_mpp_total_ns + result->price_out_impl_total_ns +
                   result->flow_cost_total_ns + result->refresh_potential_total_ns;
    if (total > 0) {
        printf("\nTime Breakdown:\n");
        printf("  primal_bea_mpp:    %5.1f%%\n", 100.0 * result->primal_bea_mpp_total_ns / total);
        printf("  price_out_impl:    %5.1f%%\n", 100.0 * result->price_out_impl_total_ns / total);
        printf("  flow_cost:         %5.1f%%\n", 100.0 * result->flow_cost_total_ns / total);
        printf("  refresh_potential: %5.1f%%\n", 100.0 * result->refresh_potential_total_ns / total);
    }

    printf("=============================================\n\n");
}

#ifdef __cplusplus
}
#endif

#endif // MCF_PROFILER_H
