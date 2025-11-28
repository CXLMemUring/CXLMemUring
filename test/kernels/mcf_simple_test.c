// Simple MCF Pricing Kernel Test - No Atomics
// Tests basic parallel arc pricing

#include <stdint.h>
#include "vx_intrinsics.h"

// Test data size
#define NUM_ARCS 32

// Arc identifiers
#define BASIC     0
#define AT_LOWER  1
#define AT_UPPER  2

// Global test data
static int32_t arc_costs[NUM_ARCS];
static int32_t tail_pot[NUM_ARCS];
static int32_t head_pot[NUM_ARCS];
static int32_t arc_idents[NUM_ARCS];
static int32_t red_costs[NUM_ARCS];
static int32_t is_candidate[NUM_ARCS];

// Initialize test data
void init_data(void) {
    for (int i = 0; i < NUM_ARCS; i++) {
        arc_costs[i] = 100 + (i * 7) % 200;
        tail_pot[i] = 50 + (i * 3) % 100;
        head_pot[i] = 30 + (i * 5) % 100;
        arc_idents[i] = (i % 3 == 0) ? AT_LOWER : ((i % 3 == 1) ? AT_UPPER : BASIC);
        red_costs[i] = 0;
        is_candidate[i] = 0;
    }
}

// Kernel function - price arcs (no atomics)
void price_arcs_kernel(void) {
    uint32_t tid = vx_thread_id();
    uint32_t num_threads = vx_num_threads();

    // Each thread processes a subset of arcs
    for (uint32_t i = tid; i < NUM_ARCS; i += num_threads) {
        // Compute reduced cost
        int32_t red_cost = arc_costs[i] - tail_pot[i] + head_pot[i];
        red_costs[i] = red_cost;

        // Check if dual infeasible
        int32_t ident = arc_idents[i];
        if ((ident == AT_LOWER && red_cost < 0) ||
            (ident == AT_UPPER && red_cost > 0)) {
            is_candidate[i] = 1;
        }
    }
}

// Main entry point
int main() {
    // Thread 0 initializes data
    if (vx_thread_id() == 0) {
        init_data();
    }

    // Barrier to ensure data is initialized
    vx_warp_barrier();

    // All threads run pricing
    price_arcs_kernel();

    // Barrier to ensure all threads complete
    vx_warp_barrier();

    // Count candidates (thread 0 only)
    int count = 0;
    if (vx_thread_id() == 0) {
        for (int i = 0; i < NUM_ARCS; i++) {
            if (is_candidate[i]) {
                count++;
            }
        }
    }

    return count;
}
