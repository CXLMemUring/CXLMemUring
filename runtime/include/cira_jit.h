// CIRA JIT — runtime cost-model that picks four offload knobs:
//   1. batch_size           : descriptors per dispatch (amortizes sync overhead)
//   2. traversal_depth      : pointer hops before yielding back to host
//   3. pipeline_distance    : prefetch lookahead (software pipelining stage count)
//   4. host_device_split    : fraction of region work that stays on the host (0..1)
//
// Inputs come from the two-pass profiler in vortex_verilator_sim.h
// (T_host, T_vortex, memory-stall cycles, cache miss rate, etc.).
// Outputs are written into offload_region_profile_t and consumed by the
// MLIR rewriter on the next compilation pass.

#ifndef CIRA_JIT_H
#define CIRA_JIT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Hardware ceilings the JIT must not exceed.
// Defaults match the Agilex 7 + Vortex configuration in vortex_verilator_sim.h.
typedef struct {
    uint32_t max_batch_size;            // BAR0 window / descriptor size
    uint32_t min_batch_size;            // smallest useful batch
    uint32_t max_traversal_depth;       // Vortex warp/register pressure
    uint32_t max_pipeline_distance;     // cache-line staging buffer depth
    uint32_t cache_line_bytes;
    double   sync_overhead_ns;          // host<->device round-trip
    double   cxl_latency_ns;            // remote line fetch
    double   llc_hit_latency_ns;        // local hit cost
    double   clock_freq_mhz;            // device clock (Vortex)
} cira_hw_limits_t;

// Sensible defaults pulled from VORTEX_*/CXL_* macros and the paper.
static inline cira_hw_limits_t cira_jit_default_limits(void) {
    cira_hw_limits_t l;
    l.max_batch_size        = 256;     // 2 MB BAR0 / 8 KB descriptor pool
    l.min_batch_size        = 4;
    l.max_traversal_depth   = 64;      // matches existing prefetch clamp
    l.max_pipeline_distance = 64;
    l.cache_line_bytes      = 64;
    l.sync_overhead_ns      = 1050.0;  // \syncOverhead in eval-commands.tex
    l.cxl_latency_ns        = 165.0;   // \cxlLatency
    l.llc_hit_latency_ns    = 15.0;    // \llcLatency
    l.clock_freq_mhz        = 200.0;   // \vortexFreqMhz
    return l;
}

// Per-region inputs — one snapshot of profile data per call.
typedef struct {
    // Wall-clock host work in this region during the profiling pass.
    uint64_t host_independent_work_ns;
    // Simulated/measured device-side time for the same region.
    uint64_t vortex_total_time_ns;
    // Cycle-level breakdown from the device timing model.
    uint64_t vortex_compute_cycles;
    uint64_t vortex_memory_stall_cycles;
    uint64_t vortex_total_cycles;
    // Cache behaviour over the region.
    uint64_t cache_hits;
    uint64_t cache_misses;
    // Number of independent elements processed by the region (for batching).
    uint64_t num_elements;
    // Per-element synchronous host cost when the work runs on the host
    // (used to bound the host/device split). 0 = unknown, fall back to T_host/N.
    double   host_per_elem_ns;
} cira_jit_workload_t;

// JIT output. Consumed by both the MLIR rewriter (compile-time) and the
// runtime dispatch path (which can re-tune between iterations).
typedef struct {
    uint32_t batch_size;
    uint32_t traversal_depth;
    uint32_t pipeline_distance;
    float    host_device_split;     // 0.0 = fully on device, 1.0 = fully on host
    bool     latency_hidden;        // T_host >= T_vortex after the split
    bool     should_offload;        // false ⇒ keep region on host entirely
    float    expected_speedup;      // vs. host-only execution
    // Per-knob rationale codes (for debug/JSON), see CIRA_JIT_REASON_*.
    uint32_t reason_bits;
} cira_jit_decision_t;

// Reason bits set on cira_jit_decision_t::reason_bits.
#define CIRA_JIT_REASON_SMALL_REGION    (1u << 0)  // region < sync overhead
#define CIRA_JIT_REASON_COMPUTE_BOUND   (1u << 1)  // device wouldn't help
#define CIRA_JIT_REASON_BATCH_AMORT     (1u << 2)  // batch sized to hide sync
#define CIRA_JIT_REASON_DEEP_CHAIN      (1u << 3)  // long dependent-load chain
#define CIRA_JIT_REASON_LATENCY_HIDDEN  (1u << 4)  // T_host >= T_vortex
#define CIRA_JIT_REASON_SPLIT_REBALANCE (1u << 5)  // host shoulder added
#define CIRA_JIT_REASON_HW_CLAMPED      (1u << 6)  // any knob hit a hardware limit

// Pure cost-model entry point. No I/O, no allocation, safe to call on the
// hot path between iterations of an offloaded loop.
void cira_jit_decide(const cira_jit_workload_t* in,
                     const cira_hw_limits_t*    hw,
                     cira_jit_decision_t*       out);

// Clamp a decision in place to the supplied hardware limits.
// Sets CIRA_JIT_REASON_HW_CLAMPED if any field was modified.
void cira_jit_clamp_to_hw(cira_jit_decision_t*    d,
                          const cira_hw_limits_t* hw);

// Human-readable one-line summary for logs / JSON. Writes up to buf_size
// bytes including the trailing NUL. Returns bytes written (excluding NUL),
// or the number that would have been written if buf were large enough.
int cira_jit_format(const cira_jit_decision_t* d, char* buf, size_t buf_size);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CIRA_JIT_H
