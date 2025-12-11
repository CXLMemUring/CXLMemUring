// Vortex Verilator Simulation Interface
// Implements cycle-accurate simulation with CXL memory latency modeling
// for the two-pass execution methodology described in the paper
//
// Two-Pass Approach:
// 1. Profiling Pass: Collect T_host (wall-clock) and T_vortex (simulated cycles)
// 2. Timing Injection Pass: Inject usleep delays when T_vortex > T_host

#ifndef VORTEX_VERILATOR_SIM_H
#define VORTEX_VERILATOR_SIM_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <pthread.h>

// Vortex SDK paths (update as needed)
#ifndef VORTEX_HOME
#define VORTEX_HOME "/home/victoryang00/vortex"
#endif

#ifndef VORTEX_BUILD_PATH
#define VORTEX_BUILD_PATH VORTEX_HOME "/build"
#endif

// Libraries available for simulation
// - librtlsim.so: RTL-level simulation (most accurate, slowest)
// - libsimx.so: Software simulation (faster, less accurate)
// - libvortex-simx.so: Full Vortex driver with simx backend
#define VORTEX_RTLSIM_PATH VORTEX_BUILD_PATH "/runtime/librtlsim.so"
#define VORTEX_SIMX_PATH VORTEX_BUILD_PATH "/runtime/libsimx.so"
#define VORTEX_DRIVER_PATH VORTEX_BUILD_PATH "/runtime/libvortex-simx.so"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Configuration Constants
//===----------------------------------------------------------------------===//

// Vortex FPGA target configuration (Agilex 7)
#define VORTEX_CLOCK_FREQ_MHZ       200     // 200MHz FPGA clock
#define VORTEX_NUM_CORES            4
#define VORTEX_WARPS_PER_CORE       8
#define VORTEX_THREADS_PER_WARP     32

// CXL Memory Latency Model
#define CXL_BASE_LATENCY_NS         165     // Base CXL latency from device side
#define CXL_BANDWIDTH_GBPS          32.0    // PCIe Gen5 x8 bandwidth
#define LLC_HIT_LATENCY_NS          15      // L3 cache hit latency
#define DRAM_LATENCY_NS             80      // Local DRAM latency

// Synchronization overhead (ring buffer communication)
#define SYNC_OVERHEAD_NS            50      // Host-accelerator sync cost

//===----------------------------------------------------------------------===//
// Simulation State Structures
//===----------------------------------------------------------------------===//

// Memory access event for latency modeling
typedef struct {
    uint64_t address;
    uint64_t timestamp_cycles;
    uint32_t size_bytes;
    bool is_write;
    bool is_prefetch;
} vortex_mem_event_t;

// Simulation timing results
typedef struct {
    // Raw cycle counts from Verilator
    uint64_t total_cycles;
    uint64_t compute_cycles;
    uint64_t memory_stall_cycles;
    uint64_t sync_cycles;

    // Converted to nanoseconds (at VORTEX_CLOCK_FREQ_MHZ)
    uint64_t total_time_ns;
    uint64_t compute_time_ns;
    uint64_t memory_stall_time_ns;

    // Memory statistics
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t memory_requests;
    uint64_t prefetch_hits;
    uint64_t prefetch_misses;

    // Warp utilization
    double avg_warp_occupancy;
    uint64_t divergent_branches;
} vortex_sim_timing_t;

// Per-offload region profiling data
typedef struct {
    // Identification
    uint32_t region_id;
    const char* region_name;

    // Host-side timing (wall-clock during profiling pass)
    uint64_t host_independent_work_ns;  // T_host: time for independent x86 work

    // Vortex-side timing (from simulation)
    vortex_sim_timing_t vortex_timing;  // T_vortex: simulated execution time

    // Computed injection delay
    int64_t injection_delay_ns;         // T_vortex - T_host (if positive)
    bool latency_hidden;                // True if T_host >= T_vortex

    // Prefetch effectiveness
    uint64_t prefetch_slack_ns;         // Available time for prefetching
    uint32_t optimal_prefetch_depth;    // Recommended prefetch lookahead
} offload_region_profile_t;

// Two-pass execution context
typedef struct {
    // Execution mode
    enum {
        TWOPASS_MODE_DISABLED = 0,
        TWOPASS_MODE_PROFILING = 1,     // Pass 1: collect timing
        TWOPASS_MODE_INJECTION = 2      // Pass 2: inject delays
    } mode;

    // Profile data storage
    offload_region_profile_t* regions;
    uint32_t num_regions;
    uint32_t regions_capacity;

    // Current execution state
    uint32_t current_region_id;
    struct timespec host_work_start;
    bool host_work_active;

    // Simulation handle
    void* verilator_sim;                // Opaque handle to Verilator context

    // Threading for async simulation
    pthread_t sim_thread;
    pthread_mutex_t sim_mutex;
    pthread_cond_t sim_cond;
    bool sim_running;
    bool sim_complete;

    // Configuration
    uint64_t cxl_latency_ns;           // Configurable CXL latency
    double clock_freq_mhz;              // Vortex clock frequency
    const char* profile_output_path;    // JSON output file path
} twopass_context_t;

//===----------------------------------------------------------------------===//
// Verilator Simulation Interface
//===----------------------------------------------------------------------===//

// Initialize Verilator simulation context
// sim_path: path to compiled Vortex RTL model
int vortex_sim_init(void** sim_handle, const char* sim_path);

// Configure CXL memory latency model
int vortex_sim_set_memory_latency(void* sim_handle, uint64_t latency_ns);

// Load kernel binary into simulated memory
int vortex_sim_load_kernel(void* sim_handle, const void* binary, size_t size);

// Upload data to simulated device memory
int vortex_sim_upload_data(void* sim_handle, uint64_t dev_addr,
                           const void* host_data, size_t size);

// Download data from simulated device memory
int vortex_sim_download_data(void* sim_handle, void* host_data,
                             uint64_t dev_addr, size_t size);

// Run simulation and collect timing
// Returns simulated cycle count, fills timing structure
int vortex_sim_run(void* sim_handle, vortex_sim_timing_t* timing);

// Run simulation asynchronously (for profiling pass)
int vortex_sim_run_async(void* sim_handle, pthread_t* thread);

// Wait for async simulation to complete
int vortex_sim_wait(void* sim_handle, pthread_t thread,
                    vortex_sim_timing_t* timing);

// Reset simulation state
int vortex_sim_reset(void* sim_handle);

// Cleanup simulation context
void vortex_sim_destroy(void* sim_handle);

//===----------------------------------------------------------------------===//
// Cycle-to-Time Conversion
//===----------------------------------------------------------------------===//

// Convert simulated cycles to nanoseconds
static inline uint64_t cycles_to_ns(uint64_t cycles, double clock_freq_mhz) {
    // ns = cycles * (1000 / freq_mhz)
    return (uint64_t)(cycles * 1000.0 / clock_freq_mhz);
}

// Convert nanoseconds to cycles
static inline uint64_t ns_to_cycles(uint64_t ns, double clock_freq_mhz) {
    return (uint64_t)(ns * clock_freq_mhz / 1000.0);
}

//===----------------------------------------------------------------------===//
// Two-Pass Execution Runtime
//===----------------------------------------------------------------------===//

// Initialize two-pass execution context
int twopass_init(twopass_context_t* ctx, const char* sim_path);

// Configure for profiling pass (Pass 1)
void twopass_set_profiling_mode(twopass_context_t* ctx);

// Configure for injection pass (Pass 2)
// Loads previously collected profile data
int twopass_set_injection_mode(twopass_context_t* ctx,
                                const char* profile_path);

// Register an offload region for tracking
uint32_t twopass_register_region(twopass_context_t* ctx,
                                  const char* region_name);

// Mark start of host independent work (called at dominator point)
void twopass_host_work_start(twopass_context_t* ctx, uint32_t region_id);

// Mark end of host independent work (called before data consumption)
void twopass_host_work_end(twopass_context_t* ctx, uint32_t region_id);

// Launch Vortex kernel (profiling mode: runs simulation)
int twopass_launch_kernel(twopass_context_t* ctx, uint32_t region_id,
                          const void* kernel_binary, size_t binary_size,
                          const void* args, size_t args_size);

// Synchronization point - inject delay if needed (injection mode)
void twopass_sync_point(twopass_context_t* ctx, uint32_t region_id);

// Save profiling data to JSON
int twopass_save_profile(twopass_context_t* ctx, const char* output_path);

// Load profiling data from JSON
int twopass_load_profile(twopass_context_t* ctx, const char* input_path);

// Cleanup
void twopass_destroy(twopass_context_t* ctx);

//===----------------------------------------------------------------------===//
// Dominator Tree Integration
//===----------------------------------------------------------------------===//

// Information about prefetch placement from dominator analysis
typedef struct {
    uint32_t region_id;

    // Dominator tree information
    uint32_t dominator_block_id;        // Earliest legal prefetch point
    uint32_t user_block_id;             // Point where data is consumed

    // Computed slack (from profiling)
    uint64_t available_slack_ns;        // Time between dominator and user
    uint64_t required_prefetch_ns;      // Time needed for prefetch

    // Recommendations
    bool can_hide_latency;              // True if slack >= prefetch time
    uint32_t recommended_depth;         // Prefetch lookahead depth
    bool should_hoist_h2d;              // Hoist H2D transfer out of loop
    bool should_sink_d2h;               // Sink D2H transfer after loop
} prefetch_placement_t;

// Compute prefetch placement based on profiling data
int twopass_compute_prefetch_placement(twopass_context_t* ctx,
                                        uint32_t region_id,
                                        uint32_t dominator_block,
                                        uint32_t user_block,
                                        prefetch_placement_t* placement);

// Update dominator tree annotations with timing data
// This generates attributes for the compiler's second pass
int twopass_annotate_dominator_tree(twopass_context_t* ctx,
                                     const char* annotation_output_path);

//===----------------------------------------------------------------------===//
// Memory Latency Model
//===----------------------------------------------------------------------===//

// CXL memory latency calculation
typedef struct {
    uint64_t base_latency_ns;           // Base device-side latency
    uint64_t protocol_overhead_ns;      // CXL.mem protocol overhead
    uint64_t queue_delay_ns;            // Memory controller queuing
    uint64_t contention_factor;         // Multi-tenant contention multiplier
} cxl_latency_model_t;

// Initialize latency model with defaults
void cxl_latency_model_init(cxl_latency_model_t* model);

// Calculate total access latency
uint64_t cxl_latency_model_calculate(const cxl_latency_model_t* model,
                                      uint64_t access_size,
                                      bool is_sequential);

// Update model based on profiling observations
void cxl_latency_model_update(cxl_latency_model_t* model,
                               uint64_t observed_latency_ns,
                               uint64_t access_size);

//===----------------------------------------------------------------------===//
// Profile Data Serialization
//===----------------------------------------------------------------------===//

// JSON output structure for compiler consumption
typedef struct {
    // Overall statistics
    uint64_t total_profiled_regions;
    uint64_t total_simulation_time_ns;

    // Per-region data
    offload_region_profile_t* regions;
    uint32_t num_regions;

    // Memory model parameters
    cxl_latency_model_t latency_model;

    // Recommendations for next compilation
    struct {
        uint32_t region_id;
        const char* region_name;
        int64_t injection_delay_ns;
        bool latency_hidden;
        uint32_t optimal_prefetch_depth;
        bool hoist_h2d;
        bool sink_d2h;
    }* recommendations;
    uint32_t num_recommendations;
} twopass_profile_data_t;

// Serialize profile data to JSON
int twopass_profile_to_json(const twopass_profile_data_t* data,
                            const char* output_path);

// Parse profile data from JSON
int twopass_profile_from_json(twopass_profile_data_t* data,
                              const char* input_path);

// Free profile data
void twopass_profile_free(twopass_profile_data_t* data);

#ifdef __cplusplus
}
#endif

#endif // VORTEX_VERILATOR_SIM_H
