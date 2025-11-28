// MCF Vortex Runtime - Profile-guided H2D/D2H optimization
// Uses dominator tree analysis for transfer placement

#ifndef MCF_VORTEX_RUNTIME_H
#define MCF_VORTEX_RUNTIME_H

#include <stdint.h>
#include <stdlib.h>

// Include offload annotations
#include "mcf_offload_annotations.h"

// Transfer buffer management
typedef struct {
    void *host_ptr;
    uint64_t device_ptr;
    size_t size;
    int dirty;  // 1 if host has newer data
} transfer_buffer_t;

// Offload context
typedef struct {
    // Device handle
    void *device;

    // Transfer buffers (based on liveness analysis)
    transfer_buffer_t arc_costs;
    transfer_buffer_t tail_potentials;
    transfer_buffer_t head_potentials;
    transfer_buffer_t arc_idents;
    transfer_buffer_t red_costs;
    transfer_buffer_t is_candidate;

    // Timing
    uint64_t h2d_cycles;
    uint64_t d2h_cycles;
    uint64_t kernel_cycles;

    // Stats
    int h2d_count;
    int d2h_count;
    int kernel_invocations;
} offload_context_t;

// Initialize offload context
int offload_init(offload_context_t *ctx, int num_arcs);

// Cleanup
void offload_cleanup(offload_context_t *ctx);

// H2D transfer with hoisting optimization
// Only transfers if data changed since last transfer
int offload_h2d(offload_context_t *ctx,
                int *arc_costs, int *tail_pot, int *head_pot, int *idents,
                int num_arcs, int force);

// D2H transfer with sinking optimization
// Defers transfer until data is actually needed
int offload_d2h(offload_context_t *ctx,
                int *red_costs, int *is_candidate,
                int num_arcs);

// Launch pricing kernel
// Conditional offload based on element count
int offload_pricing_kernel(offload_context_t *ctx,
                          int num_arcs,
                          int group_size,
                          int group_pos,
                          int *candidate_count);

// Check if offload is beneficial based on profile
static inline int should_offload_pricing(int num_arcs) {
#if OFFLOAD_PRICING_KERNEL == 0
    return 0;  // CPU only
#elif OFFLOAD_PRICING_KERNEL == 1
    return 1;  // Always offload
#else
    return num_arcs >= PRICING_KERNEL_MIN_ELEMENTS;  // Conditional
#endif
}

// Get profiling stats
void offload_get_stats(offload_context_t *ctx,
                      uint64_t *h2d_ns, uint64_t *d2h_ns, uint64_t *kernel_ns);

#endif // MCF_VORTEX_RUNTIME_H
