//===- TwoPassTimingAnalysis.h - Two-Pass Timing Integration ------*- C++ -*-===//
//
// This file declares passes for integrating two-pass timing analysis
// with the CIRA compiler infrastructure.
//
//===----------------------------------------------------------------------===//

#ifndef CIRA_DIALECT_TWOPASSTIMINGANALYSIS_H
#define CIRA_DIALECT_TWOPASSTIMINGANALYSIS_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace cira {

//===----------------------------------------------------------------------===//
// Pass Creation Functions
//===----------------------------------------------------------------------===//

/// Creates a pass that loads timing profile data and annotates offload regions
/// with timing information from the profiling pass.
///
/// Options:
///   --profile=<path>: Path to JSON timing profile
///   --annotate-only: Only add annotations, don't modify prefetch depths
std::unique_ptr<OperationPass<ModuleOp>> createTwoPassTimingInjectionPass();

/// Creates a pass that optimizes prefetch operations based on profile-guided
/// timing data. Uses dominator tree analysis to determine optimal placement.
std::unique_ptr<OperationPass<func::FuncOp>> createProfileGuidedPrefetchPass();

/// Creates a pass that inserts runtime calls for timing injection during
/// the second execution pass.
std::unique_ptr<OperationPass<ModuleOp>> createTimingInjectionCallsPass();

/// Register all two-pass related passes with the MLIR pass infrastructure.
void registerTwoPassPasses();

//===----------------------------------------------------------------------===//
// Timing Annotation Attributes
//===----------------------------------------------------------------------===//

// Attribute names used for timing annotations:
//
// twopass.injection_delay_ns (i64)
//   The delay in nanoseconds that should be injected at sync points
//   when T_vortex > T_host. Negative or zero means latency is hidden.
//
// twopass.latency_hidden (bool)
//   True if the host's independent work time (T_host) is sufficient to
//   hide the Vortex execution latency (T_vortex).
//
// twopass.optimal_prefetch_depth (i32)
//   The recommended prefetch lookahead depth based on memory stall analysis.
//   Deeper prefetch for longer memory stalls.
//
// twopass.should_hoist_h2d (bool)
//   True if H2D transfers should be hoisted out of loops based on
//   dominator tree analysis.
//
// twopass.should_sink_d2h (bool)
//   True if D2H transfers should be sunk after loops based on
//   post-dominator analysis.
//
// twopass.vortex_cycles (i64)
//   The total simulated cycle count from Verilator.
//
// twopass.cache_hit_rate (f64)
//   The observed cache hit rate during simulation.
//
// twopass.region_id (i32)
//   The unique identifier for this offload region, used for runtime
//   timing injection calls.

//===----------------------------------------------------------------------===//
// Integration with Cost Model
//===----------------------------------------------------------------------===//

/// Cost model parameters (from paper Section 4.4)
///
/// The gain from offloading is computed as:
///   Gain = Σ(L_CXL - L_LLC) - (C_sync + C_vortex_busy)
///
/// Where:
///   L_CXL = CXL memory access latency (165ns typical)
///   L_LLC = LLC hit latency (15ns typical)
///   C_sync = Synchronization overhead via ring buffer (50ns typical)
///   C_vortex_busy = Time Vortex core is busy with other work
///
/// Offloading is only beneficial when:
///   dependency_chain_depth × (L_CXL - L_LLC) > C_sync + C_vortex_busy

struct OffloadCostModelParams {
  static constexpr double L_CXL_NS = 165.0;      // CXL memory latency
  static constexpr double L_LLC_NS = 15.0;       // LLC hit latency
  static constexpr double C_SYNC_NS = 50.0;      // Ring buffer sync cost
  static constexpr int MIN_CHAIN_DEPTH = 4;      // Minimum profitable depth

  /// Compute expected gain from offloading
  static double computeGain(int64_t chainDepth, double vortexBusyNs = 0.0) {
    double latencySaving = chainDepth * (L_CXL_NS - L_LLC_NS);
    double overhead = C_SYNC_NS + vortexBusyNs;
    return latencySaving - overhead;
  }

  /// Check if offloading is profitable
  static bool shouldOffload(int64_t chainDepth, double vortexBusyNs = 0.0) {
    return computeGain(chainDepth, vortexBusyNs) > 0 &&
           chainDepth >= MIN_CHAIN_DEPTH;
  }
};

//===----------------------------------------------------------------------===//
// Profile Data Format
//===----------------------------------------------------------------------===//

/// JSON profile format expected by the timing injection pass:
///
/// {
///   "num_regions": <int>,
///   "clock_freq_mhz": <float>,
///   "cxl_latency_ns": <int>,
///   "regions": [
///     {
///       "region_id": <int>,
///       "region_name": "<string>",
///       "host_independent_work_ns": <int>,
///       "vortex_timing": {
///         "total_cycles": <int>,
///         "total_time_ns": <int>,
///         "compute_cycles": <int>,
///         "memory_stall_cycles": <int>,
///         "cache_hits": <int>,
///         "cache_misses": <int>
///       },
///       "injection_delay_ns": <int>,
///       "latency_hidden": <bool>,
///       "optimal_prefetch_depth": <int>
///     },
///     ...
///   ]
/// }

} // namespace cira
} // namespace mlir

#endif // CIRA_DIALECT_TWOPASSTIMINGANALYSIS_H
