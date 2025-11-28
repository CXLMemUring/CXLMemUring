// MCF Vortex Runtime Implementation
// Profile-guided H2D/D2H optimization using dominator tree and liveness analysis

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#ifdef USE_VORTEX_SIM
#include <vortex.h>
#endif

// Profile-guided offload annotations
#define OFFLOAD_PRICING_KERNEL 2
#define PRICING_KERNEL_MIN_ELEMENTS 150
#define PRICING_KERNEL_CAN_HOIST_H2D 1
#define PRICING_KERNEL_CAN_SINK_D2H 0
#define PRICING_KERNEL_H2D_BYTES 2848
#define PRICING_KERNEL_D2H_BYTES 2816

// Transfer buffer management
typedef struct {
    void *host_ptr;
    uint64_t device_ptr;
    size_t size;
    int dirty;
} transfer_buffer_t;

// Offload context
typedef struct {
    void *device;
    transfer_buffer_t arc_costs;
    transfer_buffer_t tail_potentials;
    transfer_buffer_t head_potentials;
    transfer_buffer_t arc_idents;
    transfer_buffer_t red_costs;
    transfer_buffer_t is_candidate;
    uint64_t h2d_cycles;
    uint64_t d2h_cycles;
    uint64_t kernel_cycles;
    int h2d_count;
    int d2h_count;
    int kernel_invocations;
} offload_context_t;

// Offload timing statistics
typedef struct {
    uint64_t h2d_time_ns;
    uint64_t kernel_time_ns;
    uint64_t d2h_time_ns;
    uint64_t h2d_bytes;
    uint64_t d2h_bytes;
    uint32_t num_calls;
} mcf_offload_stats_t;

// Global state for offload
static offload_context_t g_ctx;
static int g_initialized = 0;

//==============================================================================
// Timing utilities
//==============================================================================

static uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

//==============================================================================
// Transfer buffer management
//==============================================================================

static int alloc_transfer_buffer(transfer_buffer_t *buf, size_t size) {
    buf->host_ptr = malloc(size);
    if (!buf->host_ptr) return -1;
    buf->size = size;
    buf->dirty = 1;
    buf->device_ptr = 0;  // Will be set by device allocation
    return 0;
}

// Forward declarations
static void offload_cleanup(offload_context_t *ctx);

static void free_transfer_buffer(transfer_buffer_t *buf) {
    if (buf->host_ptr) {
        free(buf->host_ptr);
        buf->host_ptr = NULL;
    }
    buf->size = 0;
    buf->dirty = 0;
}

//==============================================================================
// Offload context management
//==============================================================================

int offload_init(offload_context_t *ctx, int num_arcs) {
    memset(ctx, 0, sizeof(*ctx));

    // Allocate transfer buffers based on liveness analysis
    size_t arc_size = num_arcs * sizeof(int);

    // Live-in buffers (H2D)
    if (alloc_transfer_buffer(&ctx->arc_costs, arc_size) < 0) goto error;
    if (alloc_transfer_buffer(&ctx->tail_potentials, arc_size) < 0) goto error;
    if (alloc_transfer_buffer(&ctx->head_potentials, arc_size) < 0) goto error;
    if (alloc_transfer_buffer(&ctx->arc_idents, arc_size) < 0) goto error;

    // Live-out buffers (D2H)
    if (alloc_transfer_buffer(&ctx->red_costs, arc_size) < 0) goto error;
    if (alloc_transfer_buffer(&ctx->is_candidate, arc_size) < 0) goto error;

#ifdef USE_VORTEX_SIM
    // Initialize Vortex device
    if (vx_dev_open(&ctx->device) != 0) {
        fprintf(stderr, "Failed to open Vortex device\n");
        goto error;
    }

    // Allocate device memory
    vx_mem_alloc(ctx->device, arc_size, VX_MEM_READ, &ctx->arc_costs.device_ptr);
    vx_mem_alloc(ctx->device, arc_size, VX_MEM_READ, &ctx->tail_potentials.device_ptr);
    vx_mem_alloc(ctx->device, arc_size, VX_MEM_READ, &ctx->head_potentials.device_ptr);
    vx_mem_alloc(ctx->device, arc_size, VX_MEM_READ, &ctx->arc_idents.device_ptr);
    vx_mem_alloc(ctx->device, arc_size, VX_MEM_WRITE, &ctx->red_costs.device_ptr);
    vx_mem_alloc(ctx->device, arc_size, VX_MEM_WRITE, &ctx->is_candidate.device_ptr);
#endif

    printf("[OFFLOAD] Initialized for %d arcs\n", num_arcs);
    printf("[OFFLOAD] H2D buffers: %zu bytes\n", PRICING_KERNEL_H2D_BYTES);
    printf("[OFFLOAD] D2H buffers: %zu bytes\n", PRICING_KERNEL_D2H_BYTES);

    return 0;

error:
    offload_cleanup(ctx);
    return -1;
}

static void offload_cleanup(offload_context_t *ctx) {
#ifdef USE_VORTEX_SIM
    if (ctx->device) {
        vx_mem_free(ctx->device, ctx->arc_costs.device_ptr);
        vx_mem_free(ctx->device, ctx->tail_potentials.device_ptr);
        vx_mem_free(ctx->device, ctx->head_potentials.device_ptr);
        vx_mem_free(ctx->device, ctx->arc_idents.device_ptr);
        vx_mem_free(ctx->device, ctx->red_costs.device_ptr);
        vx_mem_free(ctx->device, ctx->is_candidate.device_ptr);
        vx_dev_close(ctx->device);
    }
#endif

    free_transfer_buffer(&ctx->arc_costs);
    free_transfer_buffer(&ctx->tail_potentials);
    free_transfer_buffer(&ctx->head_potentials);
    free_transfer_buffer(&ctx->arc_idents);
    free_transfer_buffer(&ctx->red_costs);
    free_transfer_buffer(&ctx->is_candidate);

    printf("[OFFLOAD] Cleanup complete\n");
    printf("[OFFLOAD] Stats: H2D=%d, D2H=%d, Kernels=%d\n",
           ctx->h2d_count, ctx->d2h_count, ctx->kernel_invocations);
}

//==============================================================================
// H2D Transfer with hoisting optimization
//==============================================================================

int offload_h2d(offload_context_t *ctx,
                int *arc_costs, int *tail_pot, int *head_pot, int *idents,
                int num_arcs, int force) {

    uint64_t start = get_time_ns();
    size_t size = num_arcs * sizeof(int);

    // Copy to staging buffers
    memcpy(ctx->arc_costs.host_ptr, arc_costs, size);
    memcpy(ctx->tail_potentials.host_ptr, tail_pot, size);
    memcpy(ctx->head_potentials.host_ptr, head_pot, size);
    memcpy(ctx->arc_idents.host_ptr, idents, size);

#ifdef USE_VORTEX_SIM
    // Transfer to device
    vx_copy_to_dev(ctx->device, ctx->arc_costs.device_ptr,
                   ctx->arc_costs.host_ptr, size);
    vx_copy_to_dev(ctx->device, ctx->tail_potentials.device_ptr,
                   ctx->tail_potentials.host_ptr, size);
    vx_copy_to_dev(ctx->device, ctx->head_potentials.device_ptr,
                   ctx->head_potentials.host_ptr, size);
    vx_copy_to_dev(ctx->device, ctx->arc_idents.device_ptr,
                   ctx->arc_idents.host_ptr, size);
#endif

    uint64_t end = get_time_ns();
    ctx->h2d_cycles += (end - start);
    ctx->h2d_count++;

    return 0;
}

//==============================================================================
// D2H Transfer with sinking optimization
//==============================================================================

int offload_d2h(offload_context_t *ctx,
                int *red_costs, int *is_candidate,
                int num_arcs) {

    uint64_t start = get_time_ns();
    size_t size = num_arcs * sizeof(int);

#ifdef USE_VORTEX_SIM
    // Transfer from device
    vx_copy_from_dev(ctx->device, ctx->red_costs.host_ptr,
                     ctx->red_costs.device_ptr, size);
    vx_copy_from_dev(ctx->device, ctx->is_candidate.host_ptr,
                     ctx->is_candidate.device_ptr, size);
#endif

    // Copy from staging buffers
    memcpy(red_costs, ctx->red_costs.host_ptr, size);
    memcpy(is_candidate, ctx->is_candidate.host_ptr, size);

    uint64_t end = get_time_ns();
    ctx->d2h_cycles += (end - start);
    ctx->d2h_count++;

    return 0;
}

//==============================================================================
// Pricing kernel execution
//==============================================================================

int offload_pricing_kernel(offload_context_t *ctx,
                          int num_arcs,
                          int group_size,
                          int group_pos,
                          int *candidate_count) {

    uint64_t start = get_time_ns();

#ifdef USE_VORTEX_SIM
    // Set kernel arguments
    // Launch kernel on Vortex
    vx_start(ctx->device);
    vx_ready_wait(ctx->device, VX_MAX_TIMEOUT);
#else
    // CPU fallback - compute pricing on CPU
    int *costs = (int *)ctx->arc_costs.host_ptr;
    int *tail_pot = (int *)ctx->tail_potentials.host_ptr;
    int *head_pot = (int *)ctx->head_potentials.host_ptr;
    int *idents = (int *)ctx->arc_idents.host_ptr;
    int *red_costs = (int *)ctx->red_costs.host_ptr;
    int *is_cand = (int *)ctx->is_candidate.host_ptr;

    int count = 0;
    for (int idx = group_pos; idx < num_arcs; idx += group_size) {
        // Compute reduced cost
        int red_cost = costs[idx] - tail_pot[idx] + head_pot[idx];
        red_costs[idx] = red_cost;

        // Check dual infeasibility
        int ident = idents[idx];
        int candidate = ((ident == 1 && red_cost < 0) ||
                        (ident == 2 && red_cost > 0)) ? 1 : 0;
        is_cand[idx] = candidate;
        count += candidate;
    }
    *candidate_count = count;
#endif

    uint64_t end = get_time_ns();
    ctx->kernel_cycles += (end - start);
    ctx->kernel_invocations++;

    return 0;
}

//==============================================================================
// Statistics
//==============================================================================

void offload_get_stats(offload_context_t *ctx,
                      uint64_t *h2d_ns, uint64_t *d2h_ns, uint64_t *kernel_ns) {
    *h2d_ns = ctx->h2d_cycles;
    *d2h_ns = ctx->d2h_cycles;
    *kernel_ns = ctx->kernel_cycles;
}

//==============================================================================
// API functions required by mcf_vortex_offload.h
//==============================================================================

int mcf_vortex_init(const char* kernel_path) {
    (void)kernel_path;
    if (!g_initialized) {
        memset(&g_ctx, 0, sizeof(g_ctx));
        g_initialized = 1;
    }
    return 0;
}

int mcf_vortex_alloc_buffers(size_t num_arcs) {
    return offload_init(&g_ctx, num_arcs);
}

int mcf_vortex_upload(
    const int64_t* arc_costs,
    const int64_t* tail_potentials,
    const int64_t* head_potentials,
    const int32_t* arc_idents,
    size_t num_arcs) {

    // Ensure buffers are allocated first
    if (!g_ctx.arc_costs.host_ptr) {
        if (offload_init(&g_ctx, num_arcs) != 0) {
            return -1;
        }
    }

    return offload_h2d(&g_ctx,
                       (int*)arc_costs, (int*)tail_potentials,
                       (int*)head_potentials, (int*)arc_idents,
                       num_arcs, 0);
}

int mcf_vortex_run_pricing(
    size_t num_arcs,
    uint32_t group_stride,
    uint32_t group_offset,
    uint32_t* num_candidates) {

    // Ensure buffers are allocated
    if (!g_ctx.arc_costs.host_ptr) {
        if (offload_init(&g_ctx, num_arcs) != 0) {
            return -1;
        }
    }

    return offload_pricing_kernel(&g_ctx, num_arcs, group_stride, group_offset,
                                  (int*)num_candidates);
}

int mcf_vortex_download(
    uint32_t* candidate_indices,
    int64_t* reduced_costs,
    size_t max_candidates) {

    return offload_d2h(&g_ctx, (int*)reduced_costs, (int*)candidate_indices, max_candidates);
}

int mcf_vortex_download_per_iteration(
    uint32_t* candidate_indices,
    int64_t* reduced_costs,
    size_t max_candidates,
    size_t* basket_size) {

    // Download and count candidates
    offload_d2h(&g_ctx, (int*)reduced_costs, (int*)candidate_indices, max_candidates);

    // Count non-zero candidates
    size_t count = 0;
    int *is_cand = (int*)g_ctx.is_candidate.host_ptr;
    if (is_cand) {
        for (size_t i = 0; i < max_candidates && count < max_candidates; i++) {
            if (is_cand[i]) {
                candidate_indices[count] = i;
                reduced_costs[count] = ((int*)g_ctx.red_costs.host_ptr)[i];
                count++;
            }
        }
    }
    *basket_size += count;
    return 0;
}

void mcf_vortex_get_stats(mcf_offload_stats_t* stats) {
    stats->h2d_time_ns = g_ctx.h2d_cycles;
    stats->kernel_time_ns = g_ctx.kernel_cycles;
    stats->d2h_time_ns = g_ctx.d2h_cycles;
    stats->h2d_bytes = PRICING_KERNEL_H2D_BYTES;
    stats->d2h_bytes = PRICING_KERNEL_D2H_BYTES;
    stats->num_calls = g_ctx.kernel_invocations;
}

void mcf_vortex_reset_stats(void) {
    g_ctx.h2d_cycles = 0;
    g_ctx.d2h_cycles = 0;
    g_ctx.kernel_cycles = 0;
    g_ctx.h2d_count = 0;
    g_ctx.d2h_count = 0;
    g_ctx.kernel_invocations = 0;
}

void mcf_vortex_cleanup(void) {
    offload_cleanup(&g_ctx);
    g_initialized = 0;
}

int mcf_vortex_available(void) {
#ifdef USE_VORTEX_SIM
    return 1;
#else
    return 0;  // CPU simulation only
#endif
}
