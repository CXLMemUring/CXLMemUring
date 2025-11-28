// Optimized MCF Pricing Kernel for Vortex
// Generated with profile-guided liveness analysis

#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include <vx_print.h>

// Arc identifiers
#define BASIC     0
#define AT_LOWER  1
#define AT_UPPER  2

// Kernel arguments - minimal based on liveness analysis
typedef struct {
    // Live-in (H2D)
    int *arc_costs;        // arc->cost
    int *tail_potentials;  // arc->tail->potential
    int *head_potentials;  // arc->head->potential
    int *arc_idents;       // arc->ident
    int num_arcs;
    int group_size;        // NR_GROUP_STATE
    int group_pos;         // GROUP_POS_STATE

    // Live-out (D2H)
    int *red_costs;        // computed reduced costs
    int *is_candidate;     // dual infeasibility flag
    int *candidate_count;  // atomic counter
} pricing_args_t;

// Cycle counter for profiling
#define VX_CSR_MCYCLE 0xB00
inline uint32_t get_cycles() {
    uint32_t cycles;
    asm volatile ("csrr %0, %1" : "=r"(cycles) : "i"(VX_CSR_MCYCLE));
    return cycles;
}

// Kernel: compute reduced costs with strided access pattern
void pricing_kernel_strided(pricing_args_t *__UNIFORM__ args) {
    int tid = vx_thread_id();
    int num_threads = vx_num_threads() * vx_num_warps() * vx_num_cores();

    // Strided access pattern matching original MCF
    for (int idx = args->group_pos + tid * args->group_size;
         idx < args->num_arcs;
         idx += num_threads * args->group_size) {

        // Compute reduced cost
        int red_cost = args->arc_costs[idx]
                     - args->tail_potentials[idx]
                     + args->head_potentials[idx];

        // Store result
        args->red_costs[idx] = red_cost;

        // Check dual infeasibility
        int ident = args->arc_idents[idx];
        int is_cand = ((ident == AT_LOWER && red_cost < 0) ||
                       (ident == AT_UPPER && red_cost > 0)) ? 1 : 0;
        args->is_candidate[idx] = is_cand;

        // Count candidates (reduction)
        if (is_cand) {
            // Simple increment - will be reduced on host
            args->candidate_count[tid]++;
        }
    }
}

// Main test harness
int main() {
    uint32_t start = get_cycles();

    vx_printf("Optimized Pricing Kernel\n");

    // Kernel would be launched by host
    // This is just for testing

    uint32_t end = get_cycles();
    vx_printf("PERF: cycles=%d\n", end - start);

    return 0;
}
