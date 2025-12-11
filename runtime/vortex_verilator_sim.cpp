// Vortex Verilator Simulation Implementation
// Two-pass execution runtime for cycle-accurate timing simulation

#include "vortex_verilator_sim.h"
#include "vortex_device.h"
#include "offload_profiler.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <dlfcn.h>
#include <sys/mman.h>
#include <fcntl.h>

// JSON parsing (simple implementation)
#include <ctype.h>

//===----------------------------------------------------------------------===//
// Verilator Simulation Context
//===----------------------------------------------------------------------===//

typedef struct {
    // Verilator model handle (dynamically loaded)
    void* model_handle;
    void* vortex_model;

    // Function pointers for Verilator interface
    void (*vx_eval)(void*);
    void (*vx_step)(void*, uint64_t);
    uint64_t (*vx_get_cycles)(void*);
    int (*vx_upload_kernel)(void*, const void*, size_t);
    int (*vx_copy_to_dev)(void*, uint64_t, const void*, size_t);
    int (*vx_copy_from_dev)(void*, void*, uint64_t, size_t);
    int (*vx_start)(void*);
    int (*vx_ready_wait)(void*, uint64_t);

    // Simulated memory (for standalone testing)
    uint8_t* sim_memory;
    size_t sim_memory_size;

    // Configuration
    uint64_t cxl_latency_cycles;
    double clock_freq_mhz;

    // Statistics
    uint64_t total_cycles;
    uint64_t memory_stall_cycles;
    uint64_t cache_hits;
    uint64_t cache_misses;

    // Memory latency model
    cxl_latency_model_t latency_model;
} verilator_sim_ctx_t;

//===----------------------------------------------------------------------===//
// Verilator Interface Implementation
//===----------------------------------------------------------------------===//

int vortex_sim_init(void** sim_handle, const char* sim_path) {
    verilator_sim_ctx_t* ctx = (verilator_sim_ctx_t*)calloc(1, sizeof(verilator_sim_ctx_t));
    if (!ctx) return -1;

    // Set default configuration
    ctx->clock_freq_mhz = VORTEX_CLOCK_FREQ_MHZ;
    ctx->cxl_latency_cycles = ns_to_cycles(CXL_BASE_LATENCY_NS, ctx->clock_freq_mhz);
    cxl_latency_model_init(&ctx->latency_model);

    // Allocate simulated memory (512MB default)
    ctx->sim_memory_size = 512 * 1024 * 1024;
    ctx->sim_memory = (uint8_t*)mmap(NULL, ctx->sim_memory_size,
                                      PROT_READ | PROT_WRITE,
                                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ctx->sim_memory == MAP_FAILED) {
        free(ctx);
        return -1;
    }

    // Try to load Verilator model if path provided
    if (sim_path && strlen(sim_path) > 0) {
        ctx->model_handle = dlopen(sim_path, RTLD_NOW);
        if (ctx->model_handle) {
            // Load function pointers
            ctx->vx_eval = (void (*)(void*))dlsym(ctx->model_handle, "vx_eval");
            ctx->vx_step = (void (*)(void*, uint64_t))dlsym(ctx->model_handle, "vx_step");
            ctx->vx_get_cycles = (uint64_t (*)(void*))dlsym(ctx->model_handle, "vx_get_cycles");
            ctx->vx_upload_kernel = (int (*)(void*, const void*, size_t))dlsym(ctx->model_handle, "vx_upload_kernel");
            ctx->vx_copy_to_dev = (int (*)(void*, uint64_t, const void*, size_t))dlsym(ctx->model_handle, "vx_copy_to_dev");
            ctx->vx_copy_from_dev = (int (*)(void*, void*, uint64_t, size_t))dlsym(ctx->model_handle, "vx_copy_from_dev");
            ctx->vx_start = (int (*)(void*))dlsym(ctx->model_handle, "vx_start");
            ctx->vx_ready_wait = (int (*)(void*, uint64_t))dlsym(ctx->model_handle, "vx_ready_wait");

            // Create Vortex model instance
            void* (*create_model)(void) = (void* (*)(void))dlsym(ctx->model_handle, "vx_create");
            if (create_model) {
                ctx->vortex_model = create_model();
            }
        } else {
            fprintf(stderr, "Warning: Could not load Verilator model from %s: %s\n",
                    sim_path, dlerror());
            fprintf(stderr, "Running in estimation mode\n");
        }
    }

    *sim_handle = ctx;
    return 0;
}

int vortex_sim_set_memory_latency(void* sim_handle, uint64_t latency_ns) {
    verilator_sim_ctx_t* ctx = (verilator_sim_ctx_t*)sim_handle;
    if (!ctx) return -1;

    ctx->latency_model.base_latency_ns = latency_ns;
    ctx->cxl_latency_cycles = ns_to_cycles(latency_ns, ctx->clock_freq_mhz);
    return 0;
}

int vortex_sim_load_kernel(void* sim_handle, const void* binary, size_t size) {
    verilator_sim_ctx_t* ctx = (verilator_sim_ctx_t*)sim_handle;
    if (!ctx) return -1;

    if (ctx->vx_upload_kernel && ctx->vortex_model) {
        return ctx->vx_upload_kernel(ctx->vortex_model, binary, size);
    }

    // Fallback: copy to simulated memory at offset 0
    if (size <= ctx->sim_memory_size) {
        memcpy(ctx->sim_memory, binary, size);
        return 0;
    }
    return -1;
}

int vortex_sim_upload_data(void* sim_handle, uint64_t dev_addr,
                           const void* host_data, size_t size) {
    verilator_sim_ctx_t* ctx = (verilator_sim_ctx_t*)sim_handle;
    if (!ctx) return -1;

    if (ctx->vx_copy_to_dev && ctx->vortex_model) {
        return ctx->vx_copy_to_dev(ctx->vortex_model, dev_addr, host_data, size);
    }

    // Fallback: copy to simulated memory
    if (dev_addr + size <= ctx->sim_memory_size) {
        memcpy(ctx->sim_memory + dev_addr, host_data, size);
        return 0;
    }
    return -1;
}

int vortex_sim_download_data(void* sim_handle, void* host_data,
                             uint64_t dev_addr, size_t size) {
    verilator_sim_ctx_t* ctx = (verilator_sim_ctx_t*)sim_handle;
    if (!ctx) return -1;

    if (ctx->vx_copy_from_dev && ctx->vortex_model) {
        return ctx->vx_copy_from_dev(ctx->vortex_model, host_data, dev_addr, size);
    }

    // Fallback: copy from simulated memory
    if (dev_addr + size <= ctx->sim_memory_size) {
        memcpy(host_data, ctx->sim_memory + dev_addr, size);
        return 0;
    }
    return -1;
}

// Estimate cycles for a kernel based on workload characteristics
static uint64_t estimate_kernel_cycles(verilator_sim_ctx_t* ctx,
                                        size_t num_elements,
                                        bool is_memory_bound) {
    // Base compute cycles (assume 10 cycles per element for simple ops)
    uint64_t compute_cycles = num_elements * 10;

    if (is_memory_bound) {
        // Add memory stall cycles based on CXL latency
        // Assume 50% of accesses hit cache
        uint64_t cache_hit_cycles = (num_elements / 2) * 5;  // LLC hit
        uint64_t cache_miss_cycles = (num_elements / 2) * ctx->cxl_latency_cycles;
        ctx->memory_stall_cycles = cache_miss_cycles;
        ctx->cache_hits = num_elements / 2;
        ctx->cache_misses = num_elements / 2;

        return compute_cycles + cache_hit_cycles + cache_miss_cycles;
    }

    ctx->memory_stall_cycles = 0;
    ctx->cache_hits = num_elements;
    ctx->cache_misses = 0;
    return compute_cycles;
}

int vortex_sim_run(void* sim_handle, vortex_sim_timing_t* timing) {
    verilator_sim_ctx_t* ctx = (verilator_sim_ctx_t*)sim_handle;
    if (!ctx || !timing) return -1;

    memset(timing, 0, sizeof(vortex_sim_timing_t));

    if (ctx->vx_start && ctx->vx_ready_wait && ctx->vortex_model) {
        // Real Verilator simulation
        uint64_t start_cycles = 0;
        if (ctx->vx_get_cycles) {
            start_cycles = ctx->vx_get_cycles(ctx->vortex_model);
        }

        int ret = ctx->vx_start(ctx->vortex_model);
        if (ret != 0) return ret;

        ret = ctx->vx_ready_wait(ctx->vortex_model, 60000); // 60s timeout
        if (ret != 0) return ret;

        if (ctx->vx_get_cycles) {
            timing->total_cycles = ctx->vx_get_cycles(ctx->vortex_model) - start_cycles;
        }
    } else {
        // Estimation mode - use heuristics
        // Assume a moderate workload of 10000 elements
        timing->total_cycles = estimate_kernel_cycles(ctx, 10000, true);
        timing->memory_stall_cycles = ctx->memory_stall_cycles;
        timing->cache_hits = ctx->cache_hits;
        timing->cache_misses = ctx->cache_misses;
    }

    // Fill in timing structure
    timing->compute_cycles = timing->total_cycles - timing->memory_stall_cycles;
    timing->total_time_ns = cycles_to_ns(timing->total_cycles, ctx->clock_freq_mhz);
    timing->compute_time_ns = cycles_to_ns(timing->compute_cycles, ctx->clock_freq_mhz);
    timing->memory_stall_time_ns = cycles_to_ns(timing->memory_stall_cycles, ctx->clock_freq_mhz);
    timing->memory_requests = timing->cache_hits + timing->cache_misses;

    // Estimate warp occupancy
    timing->avg_warp_occupancy = 0.75; // Assume 75% average occupancy

    ctx->total_cycles = timing->total_cycles;
    return 0;
}

// Thread function for async simulation
typedef struct {
    verilator_sim_ctx_t* ctx;
    vortex_sim_timing_t* timing;
    int result;
    bool complete;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} async_sim_state_t;

static void* async_sim_thread(void* arg) {
    async_sim_state_t* state = (async_sim_state_t*)arg;

    state->result = vortex_sim_run(state->ctx, state->timing);

    pthread_mutex_lock(&state->mutex);
    state->complete = true;
    pthread_cond_signal(&state->cond);
    pthread_mutex_unlock(&state->mutex);

    return NULL;
}

int vortex_sim_run_async(void* sim_handle, pthread_t* thread) {
    verilator_sim_ctx_t* ctx = (verilator_sim_ctx_t*)sim_handle;
    if (!ctx || !thread) return -1;

    // Allocate async state (will be freed by wait function)
    async_sim_state_t* state = (async_sim_state_t*)calloc(1, sizeof(async_sim_state_t));
    if (!state) return -1;

    state->ctx = ctx;
    state->timing = (vortex_sim_timing_t*)calloc(1, sizeof(vortex_sim_timing_t));
    state->complete = false;
    pthread_mutex_init(&state->mutex, NULL);
    pthread_cond_init(&state->cond, NULL);

    int ret = pthread_create(thread, NULL, async_sim_thread, state);
    if (ret != 0) {
        free(state->timing);
        free(state);
        return ret;
    }

    return 0;
}

int vortex_sim_wait(void* sim_handle, pthread_t thread,
                    vortex_sim_timing_t* timing) {
    void* thread_result;
    int ret = pthread_join(thread, &thread_result);
    if (ret != 0) return ret;

    // The timing was stored in the async state
    // For now, we need a way to pass it back
    // This is simplified - real impl would use a proper async context
    return 0;
}

int vortex_sim_reset(void* sim_handle) {
    verilator_sim_ctx_t* ctx = (verilator_sim_ctx_t*)sim_handle;
    if (!ctx) return -1;

    ctx->total_cycles = 0;
    ctx->memory_stall_cycles = 0;
    ctx->cache_hits = 0;
    ctx->cache_misses = 0;

    // Clear simulated memory
    memset(ctx->sim_memory, 0, ctx->sim_memory_size);

    return 0;
}

void vortex_sim_destroy(void* sim_handle) {
    verilator_sim_ctx_t* ctx = (verilator_sim_ctx_t*)sim_handle;
    if (!ctx) return;

    if (ctx->vortex_model) {
        void (*destroy_model)(void*) = (void (*)(void*))dlsym(ctx->model_handle, "vx_destroy");
        if (destroy_model) {
            destroy_model(ctx->vortex_model);
        }
    }

    if (ctx->model_handle) {
        dlclose(ctx->model_handle);
    }

    if (ctx->sim_memory != MAP_FAILED) {
        munmap(ctx->sim_memory, ctx->sim_memory_size);
    }

    free(ctx);
}

//===----------------------------------------------------------------------===//
// Two-Pass Execution Runtime
//===----------------------------------------------------------------------===//

int twopass_init(twopass_context_t* ctx, const char* sim_path) {
    if (!ctx) return -1;

    memset(ctx, 0, sizeof(twopass_context_t));

    ctx->mode = TWOPASS_MODE_DISABLED;
    ctx->cxl_latency_ns = CXL_BASE_LATENCY_NS;
    ctx->clock_freq_mhz = VORTEX_CLOCK_FREQ_MHZ;

    // Allocate region storage
    ctx->regions_capacity = 64;
    ctx->regions = (offload_region_profile_t*)calloc(
        ctx->regions_capacity, sizeof(offload_region_profile_t));
    if (!ctx->regions) return -1;

    // Initialize Verilator simulation
    int ret = vortex_sim_init(&ctx->verilator_sim, sim_path);
    if (ret != 0) {
        free(ctx->regions);
        return ret;
    }

    // Initialize threading primitives
    pthread_mutex_init(&ctx->sim_mutex, NULL);
    pthread_cond_init(&ctx->sim_cond, NULL);

    return 0;
}

void twopass_set_profiling_mode(twopass_context_t* ctx) {
    if (!ctx) return;
    ctx->mode = TWOPASS_MODE_PROFILING;
    printf("[TwoPass] Entering profiling mode\n");
}

int twopass_set_injection_mode(twopass_context_t* ctx,
                                const char* profile_path) {
    if (!ctx) return -1;

    ctx->mode = TWOPASS_MODE_INJECTION;
    printf("[TwoPass] Entering injection mode\n");

    if (profile_path) {
        return twopass_load_profile(ctx, profile_path);
    }
    return 0;
}

uint32_t twopass_register_region(twopass_context_t* ctx,
                                  const char* region_name) {
    if (!ctx || !ctx->regions) return (uint32_t)-1;

    if (ctx->num_regions >= ctx->regions_capacity) {
        // Grow capacity
        uint32_t new_capacity = ctx->regions_capacity * 2;
        offload_region_profile_t* new_regions = (offload_region_profile_t*)realloc(
            ctx->regions, new_capacity * sizeof(offload_region_profile_t));
        if (!new_regions) return (uint32_t)-1;
        ctx->regions = new_regions;
        ctx->regions_capacity = new_capacity;
    }

    uint32_t id = ctx->num_regions++;
    offload_region_profile_t* region = &ctx->regions[id];
    memset(region, 0, sizeof(offload_region_profile_t));
    region->region_id = id;
    region->region_name = region_name ? strdup(region_name) : NULL;

    return id;
}

void twopass_host_work_start(twopass_context_t* ctx, uint32_t region_id) {
    if (!ctx || ctx->mode != TWOPASS_MODE_PROFILING) return;

    clock_gettime(CLOCK_MONOTONIC, &ctx->host_work_start);
    ctx->current_region_id = region_id;
    ctx->host_work_active = true;
}

void twopass_host_work_end(twopass_context_t* ctx, uint32_t region_id) {
    if (!ctx || ctx->mode != TWOPASS_MODE_PROFILING || !ctx->host_work_active) return;

    struct timespec end_time;
    clock_gettime(CLOCK_MONOTONIC, &end_time);

    // Calculate T_host (wall-clock time for independent work)
    uint64_t host_ns = (end_time.tv_sec - ctx->host_work_start.tv_sec) * 1000000000ULL +
                       (end_time.tv_nsec - ctx->host_work_start.tv_nsec);

    if (region_id < ctx->num_regions) {
        ctx->regions[region_id].host_independent_work_ns = host_ns;
    }

    ctx->host_work_active = false;
}

int twopass_launch_kernel(twopass_context_t* ctx, uint32_t region_id,
                          const void* kernel_binary, size_t binary_size,
                          const void* args, size_t args_size) {
    if (!ctx) return -1;

    if (ctx->mode != TWOPASS_MODE_PROFILING) {
        // In injection mode, don't actually run simulation
        return 0;
    }

    if (region_id >= ctx->num_regions) return -1;
    offload_region_profile_t* region = &ctx->regions[region_id];

    // Load kernel into simulation
    int ret = vortex_sim_load_kernel(ctx->verilator_sim, kernel_binary, binary_size);
    if (ret != 0) return ret;

    // Upload kernel arguments
    ret = vortex_sim_upload_data(ctx->verilator_sim, 0x10000, args, args_size);
    if (ret != 0) return ret;

    // Run simulation and collect T_vortex
    ret = vortex_sim_run(ctx->verilator_sim, &region->vortex_timing);
    if (ret != 0) return ret;

    printf("[TwoPass] Region %u '%s': Vortex simulation completed\n",
           region_id, region->region_name ? region->region_name : "unnamed");
    printf("  Total cycles: %lu (%.3f us)\n",
           region->vortex_timing.total_cycles,
           region->vortex_timing.total_time_ns / 1000.0);
    printf("  Memory stalls: %lu cycles (%.3f us)\n",
           region->vortex_timing.memory_stall_cycles,
           region->vortex_timing.memory_stall_time_ns / 1000.0);

    return 0;
}

void twopass_sync_point(twopass_context_t* ctx, uint32_t region_id) {
    if (!ctx) return;

    if (region_id >= ctx->num_regions) return;
    offload_region_profile_t* region = &ctx->regions[region_id];

    if (ctx->mode == TWOPASS_MODE_PROFILING) {
        // Calculate injection delay
        int64_t t_host = (int64_t)region->host_independent_work_ns;
        int64_t t_vortex = (int64_t)region->vortex_timing.total_time_ns;

        region->injection_delay_ns = t_vortex - t_host;
        region->latency_hidden = (t_host >= t_vortex);

        printf("[TwoPass] Region %u sync point:\n", region_id);
        printf("  T_host:   %ld ns (independent work)\n", t_host);
        printf("  T_vortex: %ld ns (simulated kernel)\n", t_vortex);

        if (region->latency_hidden) {
            printf("  Latency HIDDEN (T_host >= T_vortex)\n");
        } else {
            printf("  Injection delay: %ld ns (%.3f us)\n",
                   region->injection_delay_ns,
                   region->injection_delay_ns / 1000.0);
        }

        // Calculate optimal prefetch depth
        // Prefetch should cover the memory stall portion
        uint64_t stall_ns = region->vortex_timing.memory_stall_time_ns;
        // Assume we can prefetch one cache line per 100ns
        region->optimal_prefetch_depth = (uint32_t)(stall_ns / 100);
        if (region->optimal_prefetch_depth < 4) region->optimal_prefetch_depth = 4;
        if (region->optimal_prefetch_depth > 64) region->optimal_prefetch_depth = 64;

    } else if (ctx->mode == TWOPASS_MODE_INJECTION) {
        // Inject delay if T_vortex > T_host
        if (region->injection_delay_ns > 0) {
            printf("[TwoPass] Injecting delay of %ld ns for region %u\n",
                   region->injection_delay_ns, region_id);

            // Use usleep for microsecond delays
            if (region->injection_delay_ns >= 1000) {
                usleep((useconds_t)(region->injection_delay_ns / 1000));
            } else {
                // For sub-microsecond, use busy wait
                struct timespec start, now;
                clock_gettime(CLOCK_MONOTONIC, &start);
                uint64_t target_ns = region->injection_delay_ns;
                do {
                    clock_gettime(CLOCK_MONOTONIC, &now);
                } while ((now.tv_sec - start.tv_sec) * 1000000000ULL +
                         (now.tv_nsec - start.tv_nsec) < target_ns);
            }
        }
    }
}

//===----------------------------------------------------------------------===//
// Profile Serialization
//===----------------------------------------------------------------------===//

int twopass_save_profile(twopass_context_t* ctx, const char* output_path) {
    if (!ctx || !output_path) return -1;

    FILE* f = fopen(output_path, "w");
    if (!f) return -1;

    fprintf(f, "{\n");
    fprintf(f, "  \"num_regions\": %u,\n", ctx->num_regions);
    fprintf(f, "  \"clock_freq_mhz\": %.2f,\n", ctx->clock_freq_mhz);
    fprintf(f, "  \"cxl_latency_ns\": %lu,\n", ctx->cxl_latency_ns);
    fprintf(f, "  \"regions\": [\n");

    for (uint32_t i = 0; i < ctx->num_regions; i++) {
        offload_region_profile_t* r = &ctx->regions[i];
        fprintf(f, "    {\n");
        fprintf(f, "      \"region_id\": %u,\n", r->region_id);
        fprintf(f, "      \"region_name\": \"%s\",\n", r->region_name ? r->region_name : "");
        fprintf(f, "      \"host_independent_work_ns\": %lu,\n", r->host_independent_work_ns);
        fprintf(f, "      \"vortex_timing\": {\n");
        fprintf(f, "        \"total_cycles\": %lu,\n", r->vortex_timing.total_cycles);
        fprintf(f, "        \"total_time_ns\": %lu,\n", r->vortex_timing.total_time_ns);
        fprintf(f, "        \"compute_cycles\": %lu,\n", r->vortex_timing.compute_cycles);
        fprintf(f, "        \"memory_stall_cycles\": %lu,\n", r->vortex_timing.memory_stall_cycles);
        fprintf(f, "        \"cache_hits\": %lu,\n", r->vortex_timing.cache_hits);
        fprintf(f, "        \"cache_misses\": %lu\n", r->vortex_timing.cache_misses);
        fprintf(f, "      },\n");
        fprintf(f, "      \"injection_delay_ns\": %ld,\n", r->injection_delay_ns);
        fprintf(f, "      \"latency_hidden\": %s,\n", r->latency_hidden ? "true" : "false");
        fprintf(f, "      \"optimal_prefetch_depth\": %u\n", r->optimal_prefetch_depth);
        fprintf(f, "    }%s\n", (i < ctx->num_regions - 1) ? "," : "");
    }

    fprintf(f, "  ]\n");
    fprintf(f, "}\n");

    fclose(f);
    printf("[TwoPass] Saved profile to %s\n", output_path);
    return 0;
}

// Simple JSON number parser
static int64_t parse_json_int(const char* str, const char* key) {
    const char* pos = strstr(str, key);
    if (!pos) return 0;
    pos = strchr(pos, ':');
    if (!pos) return 0;
    pos++;
    while (*pos && isspace(*pos)) pos++;
    return strtoll(pos, NULL, 10);
}

static bool parse_json_bool(const char* str, const char* key) {
    const char* pos = strstr(str, key);
    if (!pos) return false;
    pos = strchr(pos, ':');
    if (!pos) return false;
    pos++;
    while (*pos && isspace(*pos)) pos++;
    return (strncmp(pos, "true", 4) == 0);
}

int twopass_load_profile(twopass_context_t* ctx, const char* input_path) {
    if (!ctx || !input_path) return -1;

    FILE* f = fopen(input_path, "r");
    if (!f) {
        fprintf(stderr, "[TwoPass] Could not open profile file: %s\n", input_path);
        return -1;
    }

    // Read entire file
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* content = (char*)malloc(fsize + 1);
    if (!content) {
        fclose(f);
        return -1;
    }
    fread(content, 1, fsize, f);
    content[fsize] = 0;
    fclose(f);

    // Parse basic fields
    ctx->num_regions = (uint32_t)parse_json_int(content, "\"num_regions\"");
    ctx->clock_freq_mhz = (double)parse_json_int(content, "\"clock_freq_mhz\"");
    ctx->cxl_latency_ns = (uint64_t)parse_json_int(content, "\"cxl_latency_ns\"");

    // Parse regions (simplified - real impl would use proper JSON parser)
    const char* regions_start = strstr(content, "\"regions\"");
    if (regions_start) {
        const char* pos = regions_start;
        uint32_t region_idx = 0;

        while ((pos = strstr(pos, "\"region_id\"")) != NULL && region_idx < ctx->num_regions) {
            if (region_idx >= ctx->regions_capacity) {
                // Grow if needed
                uint32_t new_cap = ctx->regions_capacity * 2;
                offload_region_profile_t* new_regions = (offload_region_profile_t*)realloc(
                    ctx->regions, new_cap * sizeof(offload_region_profile_t));
                if (new_regions) {
                    ctx->regions = new_regions;
                    ctx->regions_capacity = new_cap;
                }
            }

            offload_region_profile_t* r = &ctx->regions[region_idx];
            memset(r, 0, sizeof(offload_region_profile_t));

            r->region_id = (uint32_t)parse_json_int(pos, "\"region_id\"");
            r->host_independent_work_ns = (uint64_t)parse_json_int(pos, "\"host_independent_work_ns\"");
            r->vortex_timing.total_cycles = (uint64_t)parse_json_int(pos, "\"total_cycles\"");
            r->vortex_timing.total_time_ns = (uint64_t)parse_json_int(pos, "\"total_time_ns\"");
            r->injection_delay_ns = parse_json_int(pos, "\"injection_delay_ns\"");
            r->latency_hidden = parse_json_bool(pos, "\"latency_hidden\"");
            r->optimal_prefetch_depth = (uint32_t)parse_json_int(pos, "\"optimal_prefetch_depth\"");

            region_idx++;
            pos++;
        }
    }

    free(content);
    printf("[TwoPass] Loaded profile from %s (%u regions)\n", input_path, ctx->num_regions);
    return 0;
}

void twopass_destroy(twopass_context_t* ctx) {
    if (!ctx) return;

    if (ctx->verilator_sim) {
        vortex_sim_destroy(ctx->verilator_sim);
    }

    if (ctx->regions) {
        for (uint32_t i = 0; i < ctx->num_regions; i++) {
            if (ctx->regions[i].region_name) {
                free((void*)ctx->regions[i].region_name);
            }
        }
        free(ctx->regions);
    }

    pthread_mutex_destroy(&ctx->sim_mutex);
    pthread_cond_destroy(&ctx->sim_cond);
}

//===----------------------------------------------------------------------===//
// Dominator Tree Integration
//===----------------------------------------------------------------------===//

int twopass_compute_prefetch_placement(twopass_context_t* ctx,
                                        uint32_t region_id,
                                        uint32_t dominator_block,
                                        uint32_t user_block,
                                        prefetch_placement_t* placement) {
    if (!ctx || !placement || region_id >= ctx->num_regions) return -1;

    offload_region_profile_t* region = &ctx->regions[region_id];
    memset(placement, 0, sizeof(prefetch_placement_t));

    placement->region_id = region_id;
    placement->dominator_block_id = dominator_block;
    placement->user_block_id = user_block;

    // Available slack = T_host (time between dominator and user)
    placement->available_slack_ns = region->host_independent_work_ns;

    // Required prefetch time = memory stall portion of Vortex execution
    placement->required_prefetch_ns = region->vortex_timing.memory_stall_time_ns;

    // Can we hide the latency?
    placement->can_hide_latency = (placement->available_slack_ns >= placement->required_prefetch_ns);

    // Recommended prefetch depth
    placement->recommended_depth = region->optimal_prefetch_depth;

    // H2D/D2H hoisting/sinking recommendations
    // Hoist H2D if there's enough slack before the loop
    placement->should_hoist_h2d = (placement->available_slack_ns > ctx->cxl_latency_ns * 10);

    // Sink D2H if results aren't needed until after the loop
    placement->should_sink_d2h = true; // Conservative default

    return 0;
}

int twopass_annotate_dominator_tree(twopass_context_t* ctx,
                                     const char* annotation_output_path) {
    if (!ctx || !annotation_output_path) return -1;

    FILE* f = fopen(annotation_output_path, "w");
    if (!f) return -1;

    fprintf(f, "// Auto-generated CIRA timing annotations\n");
    fprintf(f, "// Feed back to compiler for optimized code generation\n\n");

    fprintf(f, "#ifndef CIRA_TIMING_ANNOTATIONS_H\n");
    fprintf(f, "#define CIRA_TIMING_ANNOTATIONS_H\n\n");

    fprintf(f, "#include <stdint.h>\n\n");

    fprintf(f, "typedef struct {\n");
    fprintf(f, "    uint32_t region_id;\n");
    fprintf(f, "    int64_t injection_delay_ns;\n");
    fprintf(f, "    uint32_t optimal_prefetch_depth;\n");
    fprintf(f, "    bool can_hide_latency;\n");
    fprintf(f, "    bool should_hoist_h2d;\n");
    fprintf(f, "    bool should_sink_d2h;\n");
    fprintf(f, "} cira_region_annotation_t;\n\n");

    fprintf(f, "static const cira_region_annotation_t cira_annotations[] = {\n");

    for (uint32_t i = 0; i < ctx->num_regions; i++) {
        offload_region_profile_t* r = &ctx->regions[i];

        // Compute placement recommendations
        prefetch_placement_t placement;
        twopass_compute_prefetch_placement(ctx, i, 0, 0, &placement);

        fprintf(f, "    { %u, %ld, %u, %s, %s, %s },\n",
                r->region_id,
                r->injection_delay_ns,
                r->optimal_prefetch_depth,
                placement.can_hide_latency ? "true" : "false",
                placement.should_hoist_h2d ? "true" : "false",
                placement.should_sink_d2h ? "true" : "false");
    }

    fprintf(f, "};\n\n");
    fprintf(f, "#define CIRA_NUM_ANNOTATIONS %u\n\n", ctx->num_regions);
    fprintf(f, "#endif // CIRA_TIMING_ANNOTATIONS_H\n");

    fclose(f);
    printf("[TwoPass] Generated annotations at %s\n", annotation_output_path);
    return 0;
}

//===----------------------------------------------------------------------===//
// CXL Memory Latency Model
//===----------------------------------------------------------------------===//

void cxl_latency_model_init(cxl_latency_model_t* model) {
    if (!model) return;

    model->base_latency_ns = CXL_BASE_LATENCY_NS;
    model->protocol_overhead_ns = 20;  // CXL.mem protocol overhead
    model->queue_delay_ns = 10;        // Memory controller queuing
    model->contention_factor = 1;      // No contention initially
}

uint64_t cxl_latency_model_calculate(const cxl_latency_model_t* model,
                                      uint64_t access_size,
                                      bool is_sequential) {
    if (!model) return CXL_BASE_LATENCY_NS;

    uint64_t latency = model->base_latency_ns +
                       model->protocol_overhead_ns +
                       model->queue_delay_ns;

    // Sequential accesses benefit from pipelining
    if (is_sequential && access_size > 64) {
        // Amortize overhead over cache lines
        uint64_t num_lines = (access_size + 63) / 64;
        latency = model->base_latency_ns + (num_lines - 1) * 10 +
                  model->protocol_overhead_ns;
    }

    // Apply contention factor
    latency *= model->contention_factor;

    return latency;
}

void cxl_latency_model_update(cxl_latency_model_t* model,
                               uint64_t observed_latency_ns,
                               uint64_t access_size) {
    if (!model) return;

    // Simple exponential moving average update
    // alpha = 0.1 for slow adaptation
    double alpha = 0.1;
    double expected = (double)cxl_latency_model_calculate(model, access_size, false);
    double observed = (double)observed_latency_ns;

    // Update base latency based on observation
    if (observed > expected * 1.1) {
        // Observed latency higher than expected - increase contention factor
        model->contention_factor = (uint64_t)(model->contention_factor * (1 + alpha));
    } else if (observed < expected * 0.9) {
        // Observed latency lower - decrease contention
        if (model->contention_factor > 1) {
            model->contention_factor = (uint64_t)(model->contention_factor * (1 - alpha));
        }
    }
}
