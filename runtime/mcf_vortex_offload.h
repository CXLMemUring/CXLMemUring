// MCF Vortex Offload Driver
// Manages data transfer and kernel execution between x86 host and Vortex GPU
//
// Profile-guided optimizations:
// - Dominator tree analysis for H2D/D2H placement
// - Liveness analysis for minimal data transfer
// - Overhead heuristics for offload decisions

#ifndef MCF_VORTEX_OFFLOAD_H
#define MCF_VORTEX_OFFLOAD_H

#include <stdint.h>
#include <stddef.h>

//==============================================================================
// Profile-guided offload annotations (from per_offload_point_profile.json)
//==============================================================================

// Offload decisions from overhead analysis
#define OFFLOAD_PRICING_KERNEL 2        // Conditional offload
#define PRICING_KERNEL_MIN_ELEMENTS 150 // Crossover point from analysis
#define PRICING_KERNEL_CAN_HOIST_H2D 1  // Dominator analysis: can hoist
#define PRICING_KERNEL_CAN_SINK_D2H 0   // Dominator analysis: cannot sink

// Data transfer sizes from liveness analysis
#define PRICING_KERNEL_H2D_BYTES 2848   // Live-in values
#define PRICING_KERNEL_D2H_BYTES 2816   // Live-out values

// Compiler optimization hints
#define PRICING_KERNEL_PREFETCH_DISTANCE 64
#define PRICING_KERNEL_THREAD_COARSENING 4

#ifdef __cplusplus
extern "C" {
#endif

// Kernel argument structure (must match mcf_pricing.c)
typedef struct {
    uint64_t arc_costs_addr;
    uint64_t tail_potentials_addr;
    uint64_t head_potentials_addr;
    uint64_t arc_idents_addr;
    uint64_t red_costs_addr;
    uint64_t candidates_addr;
    uint64_t candidate_count_addr;
    uint32_t num_arcs;
    uint32_t group_stride;
    uint32_t group_offset;
} mcf_kernel_args_t;

// Offload timing statistics
typedef struct {
    uint64_t h2d_time_ns;
    uint64_t kernel_time_ns;
    uint64_t d2h_time_ns;
    uint64_t h2d_bytes;
    uint64_t d2h_bytes;
    uint32_t num_calls;
} mcf_offload_stats_t;

// Initialize Vortex offloader
// kernel_path: path to .vxbin kernel file
// Returns 0 on success, -1 on error
int mcf_vortex_init(const char* kernel_path);

// Allocate device buffers for given number of arcs
int mcf_vortex_alloc_buffers(size_t num_arcs);

// Upload arc data to device (H2D)
int mcf_vortex_upload(
    const int64_t* arc_costs,
    const int64_t* tail_potentials,
    const int64_t* head_potentials,
    const int32_t* arc_idents,
    size_t num_arcs
);

// Run pricing kernel on Vortex
// Returns number of candidates found
int mcf_vortex_run_pricing(
    size_t num_arcs,
    uint32_t group_stride,
    uint32_t group_offset,
    uint32_t* num_candidates
);

// Download results from device (D2H)
int mcf_vortex_download(
    uint32_t* candidate_indices,
    int64_t* reduced_costs,
    size_t max_candidates
);

// Get offload statistics
void mcf_vortex_get_stats(mcf_offload_stats_t* stats);

// Reset statistics
void mcf_vortex_reset_stats(void);

// Cleanup and release resources
void mcf_vortex_cleanup(void);

// Check if Vortex offloading is available
int mcf_vortex_available(void);

//==============================================================================
// Profile-guided offload decision functions
//==============================================================================

// Should we offload pricing kernel? Based on overhead analysis crossover point.
// Returns 1 if offload is beneficial, 0 otherwise.
static inline int should_offload_pricing(size_t num_arcs) {
#if OFFLOAD_PRICING_KERNEL == 0
    (void)num_arcs;
    return 0;  // CPU only - analysis showed not beneficial
#elif OFFLOAD_PRICING_KERNEL == 1
    (void)num_arcs;
    return 1;  // Always offload
#else
    // Conditional: offload only above crossover point
    return num_arcs >= PRICING_KERNEL_MIN_ELEMENTS;
#endif
}

// Dominator-tree guided H2D transfer
// Analysis: PRICING_KERNEL_CAN_HOIST_H2D = 1
// Optimal placement: before do-while loop (hoist out of loop)
//
// Usage:
//   if (PRICING_KERNEL_CAN_HOIST_H2D) {
//       // Call once before entering do-while loop
//       mcf_vortex_upload_hoisted(arcs, num_arcs);
//   }
int mcf_vortex_upload_hoisted(
    const int64_t* arc_costs,
    const int64_t* tail_potentials,
    const int64_t* head_potentials,
    const int32_t* arc_idents,
    size_t num_arcs
);

// Liveness-guided D2H transfer
// Analysis: PRICING_KERNEL_CAN_SINK_D2H = 0
// D2H must happen per iteration since basket_size is checked in loop condition
//
// Live-out values from liveness analysis:
//   - red_costs: computed reduced costs
//   - is_candidate: dual infeasibility flags
//   - basket_size: incremented for each candidate
int mcf_vortex_download_per_iteration(
    uint32_t* candidate_indices,
    int64_t* reduced_costs,
    size_t max_candidates,
    size_t* basket_size
);

// Combined offload execution with profile-guided placement
// Handles H2D hoisting and D2H per-iteration automatically
int mcf_vortex_pricing_offload(
    const int64_t* arc_costs,
    const int64_t* tail_potentials,
    const int64_t* head_potentials,
    const int32_t* arc_idents,
    size_t num_arcs,
    uint32_t group_stride,
    uint32_t group_offset,
    uint32_t* candidate_indices,
    int64_t* reduced_costs,
    size_t* basket_size,
    int h2d_hoisted  // 1 if H2D already done (hoisted), 0 otherwise
);

#ifdef __cplusplus
}
#endif

#endif // MCF_VORTEX_OFFLOAD_H
