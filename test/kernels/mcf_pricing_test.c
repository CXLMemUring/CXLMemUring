// MCF Pricing Kernel Test Harness
// Sets up test data and runs the pricing kernel

#include <stdint.h>
#include "vx_intrinsics.h"

// Simplified kernel argument structure
typedef struct {
    int32_t* arc_costs;
    int32_t* tail_potentials;
    int32_t* head_potentials;
    int32_t* arc_idents;
    int32_t* red_costs;
    uint32_t* candidates;
    uint32_t* candidate_count;
    uint32_t num_arcs;
} kernel_args_t;

// Arc identifiers
#define BASIC     0
#define AT_LOWER  1
#define AT_UPPER  2

// Test data - small dataset for simulation
#define TEST_NUM_ARCS 64

// Allocate test data in local memory
static int32_t test_arc_costs[TEST_NUM_ARCS];
static int32_t test_tail_pot[TEST_NUM_ARCS];
static int32_t test_head_pot[TEST_NUM_ARCS];
static int32_t test_idents[TEST_NUM_ARCS];
static int32_t test_red_costs[TEST_NUM_ARCS];
static uint32_t test_candidates[TEST_NUM_ARCS];
static uint32_t test_candidate_count;
static kernel_args_t test_args;

// Initialize test data
void init_test_data(void) {
    for (uint32_t i = 0; i < TEST_NUM_ARCS; i++) {
        // Create some arcs with varying costs and potentials
        test_arc_costs[i] = 100 + (i * 7) % 200;
        test_tail_pot[i] = 50 + (i * 3) % 100;
        test_head_pot[i] = 30 + (i * 5) % 100;

        // Mix of arc types
        if (i % 3 == 0) {
            test_idents[i] = AT_LOWER;
        } else if (i % 3 == 1) {
            test_idents[i] = AT_UPPER;
        } else {
            test_idents[i] = BASIC;
        }

        test_red_costs[i] = 0;
        test_candidates[i] = 0;
    }

    test_candidate_count = 0;

    // Set up arguments
    test_args.arc_costs = test_arc_costs;
    test_args.tail_potentials = test_tail_pot;
    test_args.head_potentials = test_head_pot;
    test_args.arc_idents = test_idents;
    test_args.red_costs = test_red_costs;
    test_args.candidates = test_candidates;
    test_args.candidate_count = &test_candidate_count;
    test_args.num_arcs = TEST_NUM_ARCS;
}

// Kernel function - price arcs and find candidates
void price_arcs(kernel_args_t* args) {
    uint32_t tid = vx_thread_id();
    uint32_t num_threads = vx_num_threads();

    // Each thread processes a subset of arcs
    for (uint32_t i = tid; i < args->num_arcs; i += num_threads) {
        int32_t cost = args->arc_costs[i];
        int32_t tail_pot = args->tail_potentials[i];
        int32_t head_pot = args->head_potentials[i];
        int32_t ident = args->arc_idents[i];

        // Compute reduced cost
        int32_t red_cost = cost - tail_pot + head_pot;
        args->red_costs[i] = red_cost;

        // Check if dual infeasible (candidate for entering basis)
        int is_candidate = 0;
        if (ident == AT_LOWER && red_cost < 0) {
            is_candidate = 1;
        } else if (ident == AT_UPPER && red_cost > 0) {
            is_candidate = 1;
        }

        // Add to candidate list
        if (is_candidate) {
            uint32_t slot = vx_atomic_add(args->candidate_count, 1);
            if (slot < args->num_arcs) {
                args->candidates[slot] = i;
            }
        }
    }
}

// Main entry point
int main() {
    // Initialize test data
    init_test_data();

    // Run the pricing kernel
    price_arcs(&test_args);

    // Barrier to ensure all threads complete
    vx_warp_barrier();

    // Return number of candidates found (for verification)
    return (int)test_candidate_count;
}
