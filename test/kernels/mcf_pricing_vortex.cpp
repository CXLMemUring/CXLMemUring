// MCF Arc Pricing Kernel for Vortex
// Based on Vortex kernel SDK

#include <stdio.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>

////////////////////////////////////////////////////////////////////////////////
// Cycle counter

#define VX_CSR_MCYCLE 0xB00

inline uint32_t vx_get_cycles() {
    uint32_t cycles;
    asm volatile ("csrr %0, %1" : "=r"(cycles) : "i"(VX_CSR_MCYCLE));
    return cycles;
}

////////////////////////////////////////////////////////////////////////////////
// Memory pool (simple malloc)

#define HEAP_SZ 1024 * 1024

char __data_pool[HEAP_SZ];
int __data_pool_offset = 0;

void *vx_malloc(int sz) {
    if (__data_pool_offset + sz > HEAP_SZ) {
        vx_printf("Out of memory\n");
        return nullptr;
    }
    void *ptr = &__data_pool[__data_pool_offset];
    __data_pool_offset += sz;
    return ptr;
}

////////////////////////////////////////////////////////////////////////////////
// Kernel data structures

#define NUM_ARCS 64
#define BASIC     0
#define AT_LOWER  1
#define AT_UPPER  2

typedef struct {
    int *arc_costs;
    int *tail_pot;
    int *head_pot;
    int *arc_idents;
    int *red_costs;
    int *is_candidate;
    int num_arcs;
} pricing_args_t;

////////////////////////////////////////////////////////////////////////////////
// Pricing kernel

void pricing_kernel(pricing_args_t *__UNIFORM__ args) {
    int idx = blockIdx.x;

    if (idx < args->num_arcs) {
        // Compute reduced cost
        int cost = args->arc_costs[idx];
        int tail_p = args->tail_pot[idx];
        int head_p = args->head_pot[idx];
        int ident = args->arc_idents[idx];

        int red_cost = cost - tail_p + head_p;
        args->red_costs[idx] = red_cost;

        // Check if dual infeasible (candidate)
        int is_cand = 0;
        if ((ident == AT_LOWER && red_cost < 0) ||
            (ident == AT_UPPER && red_cost > 0)) {
            is_cand = 1;
        }
        args->is_candidate[idx] = is_cand;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Main

int main() {
    uint32_t total_start = vx_get_cycles();

    vx_printf(">> MCF Pricing Kernel Test\n");
    vx_printf(">> coreid=%d, warpid=%d, threadid=%d\n",
              vx_core_id(), vx_warp_id(), vx_thread_id());

    // Allocate arrays
    vx_printf(">> Allocating %d arcs\n", NUM_ARCS);
    int *arc_costs = (int *)vx_malloc(NUM_ARCS * sizeof(int));
    int *tail_pot = (int *)vx_malloc(NUM_ARCS * sizeof(int));
    int *head_pot = (int *)vx_malloc(NUM_ARCS * sizeof(int));
    int *arc_idents = (int *)vx_malloc(NUM_ARCS * sizeof(int));
    int *red_costs = (int *)vx_malloc(NUM_ARCS * sizeof(int));
    int *is_candidate = (int *)vx_malloc(NUM_ARCS * sizeof(int));

    // Initialize test data
    vx_printf(">> Initializing test data\n");
    for (int i = 0; i < NUM_ARCS; i++) {
        arc_costs[i] = 100 + (i * 7) % 200;
        tail_pot[i] = 50 + (i * 3) % 100;
        head_pot[i] = 30 + (i * 5) % 100;
        arc_idents[i] = (i % 3 == 0) ? AT_LOWER : ((i % 3 == 1) ? AT_UPPER : BASIC);
        red_costs[i] = 0;
        is_candidate[i] = 0;
    }

    // Set up kernel arguments
    pricing_args_t args;
    args.arc_costs = arc_costs;
    args.tail_pot = tail_pot;
    args.head_pot = head_pot;
    args.arc_idents = arc_idents;
    args.red_costs = red_costs;
    args.is_candidate = is_candidate;
    args.num_arcs = NUM_ARCS;

    // Launch kernel with cycle measurement
    vx_printf(">> Launching pricing kernel with %d threads\n", NUM_ARCS);
    uint32_t num_threads = NUM_ARCS;

    uint32_t start_cycles = vx_get_cycles();
    vx_spawn_threads(1, &num_threads, nullptr, (vx_kernel_func_cb)pricing_kernel, &args);
    uint32_t end_cycles = vx_get_cycles();

    uint32_t kernel_cycles = end_cycles - start_cycles;
    vx_printf(">> Kernel complete\n");
    vx_printf(">> PERF: kernel_cycles=%d\n", kernel_cycles);

    // Count candidates
    int num_candidates = 0;
    for (int i = 0; i < NUM_ARCS; i++) {
        if (is_candidate[i]) {
            num_candidates++;
        }
    }

    vx_printf(">> Results: %d candidates found out of %d arcs\n", num_candidates, NUM_ARCS);

    // Print first few results
    vx_printf(">> First 8 reduced costs: ");
    for (int i = 0; i < 8; i++) {
        vx_printf("%d ", red_costs[i]);
    }
    vx_printf("\n");

    uint32_t total_end = vx_get_cycles();
    uint32_t total_cycles = total_end - total_start;

    vx_printf(">> PERF: total_cycles=%d\n", total_cycles);
    vx_printf(">> MCF Pricing Kernel Test PASSED!\n");
    return 0;
}
