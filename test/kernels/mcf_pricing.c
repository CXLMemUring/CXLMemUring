// MCF Arc Pricing Kernel for Vortex RISC-V SIMT GPU
// Implements parallel arc pricing for primal_bea_mpp function

#include <stdint.h>
#include "vx_intrinsics.h"

// Vortex spawn API stub (actual implementation provided by Vortex runtime)
#ifndef __UNIFORM__
#define __UNIFORM__
#endif

// Arc identifiers (must match defines.h)
#define BASIC     0
#define AT_LOWER  1
#define AT_UPPER  2

// Kernel argument structure
typedef struct {
    // Input arrays (device addresses)
    uint64_t arc_costs_addr;        // cost_t* arc costs
    uint64_t arc_idents_addr;       // int* arc identifiers
    uint64_t tail_potentials_addr;  // cost_t* tail node potentials
    uint64_t head_potentials_addr;  // cost_t* head node potentials

    // Output arrays
    uint64_t red_costs_addr;        // cost_t* reduced costs output
    uint64_t candidates_addr;       // uint32_t* candidate arc indices
    uint64_t candidate_count_addr;  // uint32_t* atomic counter for candidates

    // Parameters
    uint32_t num_arcs;              // Total number of arcs
    uint32_t group_stride;          // NR_GROUP_STATE for strided access
    uint32_t group_offset;          // GROUP_POS_STATE starting offset
} mcf_pricing_args_t;

// Check if arc is dual infeasible (candidate for entering basis)
static inline int bea_is_dual_infeasible(int ident, int32_t red_cost) {
    return (((red_cost < 0) & (ident == AT_LOWER)) |
            ((red_cost > 0) & (ident == AT_UPPER)));
}

// Kernel body - executed by each thread
void kernel_body(mcf_pricing_args_t* __UNIFORM__ arg) {
    // Get pointers from device addresses
    int32_t* arc_costs = (int32_t*)arg->arc_costs_addr;
    int* arc_idents = (int*)arg->arc_idents_addr;
    int32_t* tail_potentials = (int32_t*)arg->tail_potentials_addr;
    int32_t* head_potentials = (int32_t*)arg->head_potentials_addr;
    int32_t* red_costs = (int32_t*)arg->red_costs_addr;
    uint32_t* candidates = (uint32_t*)arg->candidates_addr;
    uint32_t* candidate_count = (uint32_t*)arg->candidate_count_addr;

    uint32_t tid = vx_thread_id();
    uint32_t num_threads = vx_num_threads();

    // Each thread processes arcs with striding
    // Starting from group_offset, stepping by group_stride
    uint32_t arc_idx = arg->group_offset + tid * arg->group_stride;

    while (arc_idx < arg->num_arcs) {
        int ident = arc_idents[arc_idx];

        // Only process non-basic arcs
        if (ident > BASIC) {
            // Compute reduced cost
            int32_t cost = arc_costs[arc_idx];
            int32_t tail_pot = tail_potentials[arc_idx];
            int32_t head_pot = head_potentials[arc_idx];
            int32_t red_cost = cost - tail_pot + head_pot;

            // Store reduced cost
            red_costs[arc_idx] = red_cost;

            // Check if this arc is a candidate
            if (bea_is_dual_infeasible(ident, red_cost)) {
                // Atomic add to get candidate slot
                uint32_t slot = vx_atomic_add(candidate_count, 1);
                candidates[slot] = arc_idx;
            }
        }

        // Move to next arc (strided)
        arc_idx += num_threads * arg->group_stride;
    }

    // Synchronize before returning
    vx_barrier(0, 4);
}

// Main entry point
int main() {
    mcf_pricing_args_t* arg = (mcf_pricing_args_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, &arg->num_arcs, 0,
                           (vx_kernel_func_cb)kernel_body, arg);
}

// Alternative kernel: Batch pricing with local reduction
// This version collects top-K candidates locally before global sync
typedef struct {
    uint64_t arc_costs_addr;
    uint64_t arc_idents_addr;
    uint64_t tail_potentials_addr;
    uint64_t head_potentials_addr;
    uint64_t output_indices_addr;   // Per-thread top candidates
    uint64_t output_costs_addr;     // Per-thread candidate costs
    uint32_t num_arcs;
    uint32_t local_basket_size;     // Max candidates per thread
} mcf_batch_pricing_args_t;

// Shared memory for warp-level reduction
__attribute__((shared)) uint32_t warp_candidates[32];
__attribute__((shared)) int32_t warp_costs[32];

void batch_pricing_kernel(mcf_batch_pricing_args_t* __UNIFORM__ arg) {
    int32_t* arc_costs = (int32_t*)arg->arc_costs_addr;
    int* arc_idents = (int*)arg->arc_idents_addr;
    int32_t* tail_potentials = (int32_t*)arg->tail_potentials_addr;
    int32_t* head_potentials = (int32_t*)arg->head_potentials_addr;
    uint32_t* output_indices = (uint32_t*)arg->output_indices_addr;
    int32_t* output_costs = (int32_t*)arg->output_costs_addr;

    uint32_t tid = vx_thread_id();
    uint32_t num_threads = vx_num_threads();
    uint32_t lane_id = vx_lane_id();

    // Local candidate basket
    uint32_t local_count = 0;
    uint32_t max_local = arg->local_basket_size;

    // Track best candidate locally
    int32_t best_cost = 0;
    uint32_t best_idx = 0xFFFFFFFF;

    // Process arcs with thread-strided pattern
    for (uint32_t arc_idx = tid; arc_idx < arg->num_arcs; arc_idx += num_threads) {
        int ident = arc_idents[arc_idx];

        if (ident > BASIC) {
            int32_t cost = arc_costs[arc_idx];
            int32_t tail_pot = tail_potentials[arc_idx];
            int32_t head_pot = head_potentials[arc_idx];
            int32_t red_cost = cost - tail_pot + head_pot;

            if (bea_is_dual_infeasible(ident, red_cost)) {
                // Track absolute cost for comparison
                int32_t abs_cost = (red_cost < 0) ? -red_cost : red_cost;

                if (abs_cost > best_cost || best_idx == 0xFFFFFFFF) {
                    best_cost = abs_cost;
                    best_idx = arc_idx;
                }
                local_count++;
            }
        }
    }

    // Warp-level reduction to find best candidate in warp
    warp_candidates[lane_id] = best_idx;
    warp_costs[lane_id] = best_cost;
    vx_warp_barrier();

    // Reduction within warp
    for (int offset = 16; offset > 0; offset >>= 1) {
        if (lane_id < offset) {
            if (warp_costs[lane_id + offset] > warp_costs[lane_id]) {
                warp_costs[lane_id] = warp_costs[lane_id + offset];
                warp_candidates[lane_id] = warp_candidates[lane_id + offset];
            }
        }
        vx_warp_barrier();
    }

    // Lane 0 writes warp result
    if (lane_id == 0) {
        uint32_t warp_id = vx_warp_id();
        output_indices[warp_id] = warp_candidates[0];
        output_costs[warp_id] = warp_costs[0];
    }

    vx_barrier(0, 4);
}

// Reduction kernel for flow cost computation
typedef struct {
    uint64_t arc_costs_addr;
    uint64_t arc_flows_addr;
    uint64_t arc_idents_addr;
    uint64_t tail_numbers_addr;
    uint64_t head_numbers_addr;
    uint64_t partial_sums_addr;  // Output: partial sums per warp
    uint32_t num_arcs;
    int32_t bigM;
} mcf_flow_cost_args_t;

void flow_cost_kernel(mcf_flow_cost_args_t* __UNIFORM__ arg) {
    int32_t* arc_costs = (int32_t*)arg->arc_costs_addr;
    int32_t* arc_flows = (int32_t*)arg->arc_flows_addr;
    int* arc_idents = (int*)arg->arc_idents_addr;
    int* tail_numbers = (int*)arg->tail_numbers_addr;
    int* head_numbers = (int*)arg->head_numbers_addr;
    int64_t* partial_sums = (int64_t*)arg->partial_sums_addr;

    uint32_t tid = vx_thread_id();
    uint32_t num_threads = vx_num_threads();
    uint32_t lane_id = vx_lane_id();
    uint32_t warp_id = vx_warp_id();

    int64_t local_sum = 0;
    int32_t local_fleet = 0;

    // Process arcs
    for (uint32_t arc_idx = tid; arc_idx < arg->num_arcs; arc_idx += num_threads) {
        int32_t flow = arc_flows[arc_idx];

        if (flow) {
            int tail_num = tail_numbers[arc_idx];
            int head_num = head_numbers[arc_idx];

            // Skip super source/sink arcs
            if (!((tail_num < 0) & (head_num > 0))) {
                if (tail_num == 0) {
                    local_sum += (arc_costs[arc_idx] - arg->bigM);
                    local_fleet++;
                } else {
                    local_sum += arc_costs[arc_idx];
                }
            }
        }
    }

    // Warp-level reduction
    __attribute__((shared)) int64_t warp_sums[32];
    __attribute__((shared)) int32_t warp_fleets[32];

    warp_sums[lane_id] = local_sum;
    warp_fleets[lane_id] = local_fleet;
    vx_warp_barrier();

    for (int offset = 16; offset > 0; offset >>= 1) {
        if (lane_id < offset) {
            warp_sums[lane_id] += warp_sums[lane_id + offset];
            warp_fleets[lane_id] += warp_fleets[lane_id + offset];
        }
        vx_warp_barrier();
    }

    // Lane 0 writes warp result
    if (lane_id == 0) {
        // Combine fleet and operational cost
        int64_t warp_total = (int64_t)warp_fleets[0] * arg->bigM + warp_sums[0];
        partial_sums[warp_id] = warp_total;
    }

    vx_barrier(0, 4);
}
