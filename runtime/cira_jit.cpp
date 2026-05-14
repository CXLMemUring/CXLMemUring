// CIRA JIT cost-model implementation. See include/cira_jit.h.
//
// All math here is intentionally cheap and side-effect-free so it can run
// on the dispatch path between iterations of an offloaded loop.

#include "cira_jit.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {

inline double safe_div(double a, double b, double fallback = 0.0) {
    return (b > 0.0) ? (a / b) : fallback;
}

inline uint32_t clamp_u32(double v, uint32_t lo, uint32_t hi) {
    if (!(v == v))      return lo;            // NaN guard
    if (v < (double)lo) return lo;
    if (v > (double)hi) return hi;
    return (uint32_t)(v + 0.5);
}

inline double cycles_to_ns(uint64_t cycles, double freq_mhz) {
    return (freq_mhz > 0.0) ? (double)cycles * 1000.0 / freq_mhz : 0.0;
}

// Round up to next power-of-two — descriptor pools and prefetch buffers
// are typically sized in pow2 increments.
inline uint32_t next_pow2(uint32_t v) {
    if (v <= 1) return 1;
    --v;
    v |= v >> 1;  v |= v >> 2;  v |= v >> 4;
    v |= v >> 8;  v |= v >> 16;
    return v + 1;
}

} // namespace

extern "C" void cira_jit_decide(const cira_jit_workload_t* in,
                                const cira_hw_limits_t*    hw,
                                cira_jit_decision_t*       out) {
    if (!in || !hw || !out) return;

    cira_hw_limits_t L = *hw;
    cira_jit_decision_t d{};
    d.reason_bits = 0;

    const double t_host_ns   = (double)in->host_independent_work_ns;
    const double t_vortex_ns = (double)in->vortex_total_time_ns;
    const double stall_ns    = cycles_to_ns(in->vortex_memory_stall_cycles,
                                            L.clock_freq_mhz);
    const double compute_ns  = cycles_to_ns(in->vortex_compute_cycles,
                                            L.clock_freq_mhz);

    // Memory-stall fraction (Top-Down "Memory Bound" proxy).
    const double stall_frac = safe_div(stall_ns, stall_ns + compute_ns, 0.5);

    // Per-element costs (used by both batch sizing and the host/device split).
    const double n_elems = (double)(in->num_elements ? in->num_elements : 1);
    const double per_elem_dev_ns  = safe_div(t_vortex_ns, n_elems, 1.0);
    const double per_elem_host_ns = (in->host_per_elem_ns > 0.0)
                                    ? in->host_per_elem_ns
                                    : safe_div(t_host_ns, n_elems, per_elem_dev_ns);

    // ── (A) Should we offload at all? ───────────────────────────────────────
    // If the whole region is faster than a single round-trip, keep it on host.
    if (t_vortex_ns + L.sync_overhead_ns >= t_host_ns &&
        t_host_ns < L.sync_overhead_ns * 2.0) {
        d.should_offload    = false;
        d.host_device_split = 1.0f;          // entirely on host
        d.batch_size        = L.min_batch_size;
        d.traversal_depth   = 1;
        d.pipeline_distance = 0;
        d.latency_hidden    = true;
        d.expected_speedup  = 1.0f;
        d.reason_bits      |= CIRA_JIT_REASON_SMALL_REGION;
        cira_jit_clamp_to_hw(&d, &L);
        *out = d;
        return;
    }
    if (stall_frac < 0.10 && t_vortex_ns > t_host_ns * 1.5) {
        // Compute-bound and slower on the device — don't bother.
        d.should_offload    = false;
        d.host_device_split = 1.0f;
        d.batch_size        = L.min_batch_size;
        d.traversal_depth   = 1;
        d.pipeline_distance = 0;
        d.latency_hidden    = (t_host_ns >= t_vortex_ns);
        d.expected_speedup  = (float)safe_div(t_vortex_ns, t_host_ns, 1.0);
        d.reason_bits      |= CIRA_JIT_REASON_COMPUTE_BOUND;
        cira_jit_clamp_to_hw(&d, &L);
        *out = d;
        return;
    }
    d.should_offload = true;

    // ── (B) batch_size — amortize sync overhead ────────────────────────────
    // Choose batch so that sync_overhead is at most ~25% of per-batch dev time.
    // batch * per_elem_dev_ns >= 4 * sync_overhead_ns
    double batch_d = safe_div(4.0 * L.sync_overhead_ns, per_elem_dev_ns,
                              (double)L.min_batch_size);
    uint32_t batch = next_pow2(clamp_u32(batch_d, L.min_batch_size, L.max_batch_size));
    d.batch_size = std::min(batch, L.max_batch_size);
    d.reason_bits |= CIRA_JIT_REASON_BATCH_AMORT;

    // ── (C) traversal_depth — how many dependent hops per yield ─────────────
    // A high stall:compute ratio means the chain is long; raising depth lets
    // the device walk further before the host has to re-engage.
    // depth ≈ 1 + 7 * stall_frac, snapped to the [1, max] window.
    double depth_d = 1.0 + 7.0 * stall_frac;
    d.traversal_depth = clamp_u32(depth_d, 1u, L.max_traversal_depth);
    if (d.traversal_depth >= 4) d.reason_bits |= CIRA_JIT_REASON_DEEP_CHAIN;

    // ── (D) pipeline_distance — software-pipelining stages ──────────────────
    // Number of in-flight prefetches to fully hide memory latency.
    // Each prefetched line saves (cxl_latency - llc_latency) ns.
    // distance ≈ stall_ns / max(gain_per_step, 1)
    double gain_per_step = std::max(1.0, L.cxl_latency_ns - L.llc_hit_latency_ns);
    double pipe_d        = safe_div(stall_ns, gain_per_step, 4.0);
    d.pipeline_distance  = clamp_u32(pipe_d, 4u, L.max_pipeline_distance);

    // ── (E) host_device_split — load-balance to hide remaining latency ──────
    // If T_vortex <= T_host: device finishes first; split = 0 (all on device).
    // Else: shift work back to host until they finish together.
    // total_work_ns ≈ T_host + T_vortex (independent slices)
    // split = 1 - T_host / total  if device is the long pole; else 0.
    double split = 0.0;
    if (t_vortex_ns > t_host_ns && (t_host_ns + t_vortex_ns) > 0.0) {
        // Fraction that should stay on host so that
        //   split * per_elem_host == (1-split) * per_elem_dev
        double r = safe_div(per_elem_dev_ns, per_elem_host_ns + per_elem_dev_ns, 0.5);
        split    = std::max(0.0, std::min(1.0, r));
        d.reason_bits |= CIRA_JIT_REASON_SPLIT_REBALANCE;
    }
    d.host_device_split = (float)split;

    // ── (F) Outcome metrics ────────────────────────────────────────────────
    double overlapped_ns = std::max(t_host_ns * (1.0 - split),
                                    t_vortex_ns * (1.0 - split));
    if (split > 0.0) {
        overlapped_ns = std::max(t_host_ns + per_elem_host_ns * split * n_elems * 0.0,
                                 (1.0 - split) * t_vortex_ns +
                                 split * per_elem_host_ns * n_elems);
    }
    overlapped_ns += L.sync_overhead_ns;
    d.latency_hidden   = (t_host_ns >= t_vortex_ns);
    d.expected_speedup = (float)safe_div(t_host_ns + L.sync_overhead_ns,
                                         std::max(overlapped_ns, 1.0), 1.0);
    if (d.latency_hidden) d.reason_bits |= CIRA_JIT_REASON_LATENCY_HIDDEN;

    cira_jit_clamp_to_hw(&d, &L);
    *out = d;
}

extern "C" void cira_jit_clamp_to_hw(cira_jit_decision_t*    d,
                                     const cira_hw_limits_t* hw) {
    if (!d || !hw) return;
    bool clamped = false;

    if (d->batch_size < hw->min_batch_size) {
        d->batch_size = hw->min_batch_size; clamped = true;
    }
    if (d->batch_size > hw->max_batch_size) {
        d->batch_size = hw->max_batch_size; clamped = true;
    }
    if (d->traversal_depth < 1) {
        d->traversal_depth = 1; clamped = true;
    }
    if (d->traversal_depth > hw->max_traversal_depth) {
        d->traversal_depth = hw->max_traversal_depth; clamped = true;
    }
    if (d->pipeline_distance > hw->max_pipeline_distance) {
        d->pipeline_distance = hw->max_pipeline_distance; clamped = true;
    }
    if (d->host_device_split < 0.0f) { d->host_device_split = 0.0f; clamped = true; }
    if (d->host_device_split > 1.0f) { d->host_device_split = 1.0f; clamped = true; }

    if (clamped) d->reason_bits |= CIRA_JIT_REASON_HW_CLAMPED;
}

extern "C" int cira_jit_format(const cira_jit_decision_t* d,
                               char* buf, size_t buf_size) {
    if (!d || !buf || buf_size == 0) return 0;
    return std::snprintf(buf, buf_size,
        "batch=%u depth=%u pipe=%u split=%.2f offload=%d hidden=%d "
        "speedup=%.2fx reasons=0x%x",
        d->batch_size, d->traversal_depth, d->pipeline_distance,
        (double)d->host_device_split,
        d->should_offload ? 1 : 0, d->latency_hidden ? 1 : 0,
        (double)d->expected_speedup, d->reason_bits);
}
