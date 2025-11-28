#!/usr/bin/env python3
"""
Profile Feedback Compiler for Heterogeneous Offload

This script reads profiled data from heterogeneous execution and feeds it back
to the overhead analysis to regenerate optimized offload decisions.

Usage:
    python3 profile_feedback_compiler.py \
        --profile profile_results/per_offload_point_profile.json \
        --experiment profile_results/experiment_results.json \
        --output bin/mcf_heterogeneous/mcf_offload_annotations.h
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class OverheadAnalysis:
    """Result of overhead analysis for an offload point"""
    kernel_cycles: int
    kernel_cycles_per_element: int
    h2d_latency_ns: int
    h2d_transfer_ns: int
    d2h_latency_ns: int
    d2h_transfer_ns: int
    total_overhead_ns: int
    expected_elements: int
    scaled_kernel_cycles: int

@dataclass
class DominatorAnalysis:
    """Result of dominator tree analysis"""
    can_hoist_h2d: bool
    can_sink_d2h: bool
    optimal_h2d_placement: str
    optimal_d2h_placement: str
    reason: str

@dataclass
class LivenessAnalysis:
    """Result of liveness analysis"""
    h2d_total_bytes: int
    d2h_total_bytes: int
    live_in: List[Dict]
    live_out: List[Dict]

@dataclass
class OffloadHeuristics:
    """Heuristics for offload decision"""
    speedup: float
    crossover_point_elements: int
    transfer_dominance_ratio: float
    decision: str  # "gpu_always", "conditional_offload", "cpu_only"
    reason: str

@dataclass
class OffloadPoint:
    """Complete analysis for a single offload point"""
    id: str
    function: str
    location: str
    description: str
    loop_type: str
    parallelism: str
    overhead: OverheadAnalysis
    dominator: DominatorAnalysis
    liveness: LivenessAnalysis
    heuristics: OffloadHeuristics

class ProfileFeedbackCompiler:
    """Compiler that uses profiled data to generate optimal offload decisions"""

    def __init__(self, profile_path: str, experiment_path: str):
        self.profile_data = self._load_json(profile_path)
        self.experiment_data = self._load_json(experiment_path)
        self.offload_points: Dict[str, OffloadPoint] = {}

    def _load_json(self, path: str) -> dict:
        with open(path, 'r') as f:
            return json.load(f)

    def parse_profile(self):
        """Parse the per-offload-point profile into structured data"""
        for point_data in self.profile_data.get('offload_points', []):
            point = self._parse_offload_point(point_data)
            self.offload_points[point.id] = point

    def _parse_offload_point(self, data: dict) -> OffloadPoint:
        """Parse a single offload point from JSON"""
        overhead_data = data.get('overhead_analysis', {})
        overhead = OverheadAnalysis(
            kernel_cycles=overhead_data.get('kernel_cycles', 0),
            kernel_cycles_per_element=overhead_data.get('kernel_cycles_per_element', 0),
            h2d_latency_ns=overhead_data.get('h2d_latency_ns', 0),
            h2d_transfer_ns=overhead_data.get('h2d_transfer_ns', 0),
            d2h_latency_ns=overhead_data.get('d2h_latency_ns', 0),
            d2h_transfer_ns=overhead_data.get('d2h_transfer_ns', 0),
            total_overhead_ns=overhead_data.get('total_overhead_ns', 0),
            expected_elements=overhead_data.get('expected_elements', 0),
            scaled_kernel_cycles=overhead_data.get('scaled_kernel_cycles', 0)
        )

        dom_data = data.get('dominator_analysis', {})
        dominator = DominatorAnalysis(
            can_hoist_h2d=dom_data.get('can_hoist_h2d', False),
            can_sink_d2h=dom_data.get('can_sink_d2h', False),
            optimal_h2d_placement=dom_data.get('optimal_h2d_placement', ''),
            optimal_d2h_placement=dom_data.get('optimal_d2h_placement', ''),
            reason=dom_data.get('reason', '')
        )

        liveness_data = data.get('liveness_analysis', {})
        liveness = LivenessAnalysis(
            h2d_total_bytes=liveness_data.get('h2d_total_bytes', 0),
            d2h_total_bytes=liveness_data.get('d2h_total_bytes', 0),
            live_in=liveness_data.get('live_in', []),
            live_out=liveness_data.get('live_out', [])
        )

        heuristics_data = data.get('heuristics', {})
        # Get the speedup value - try different keys
        speedup = heuristics_data.get('speedup_at_64_elements',
                  heuristics_data.get('speedup_at_50_elements',
                  heuristics_data.get('speedup_at_1000_elements', 1.0)))

        heuristics = OffloadHeuristics(
            speedup=speedup,
            crossover_point_elements=heuristics_data.get('crossover_point_elements', 0),
            transfer_dominance_ratio=heuristics_data.get('transfer_dominance_ratio', 0.0),
            decision=heuristics_data.get('decision', 'cpu_only'),
            reason=heuristics_data.get('reason', '')
        )

        return OffloadPoint(
            id=data.get('id', ''),
            function=data.get('function', ''),
            location=data.get('location', ''),
            description=data.get('description', ''),
            loop_type=data.get('loop_type', ''),
            parallelism=data.get('parallelism', ''),
            overhead=overhead,
            dominator=dominator,
            liveness=liveness,
            heuristics=heuristics
        )

    def update_with_runtime_data(self):
        """Update analysis with actual runtime experiment data"""
        exp = self.experiment_data

        # Get actual kernel performance from experiment
        vortex = exp.get('vortex_execution', {})
        x86 = exp.get('x86_baseline', {})
        analysis = exp.get('analysis', {})

        actual_kernel_cycles = vortex.get('kernel_cycles', 6652)
        actual_h2d_latency = vortex.get('h2d_latency_ns', 1000000)
        actual_d2h_latency = vortex.get('d2h_latency_ns', 500000)
        actual_speedup = analysis.get('kernel_speedup', 1.0)
        total_speedup = analysis.get('total_speedup_with_overhead', 1.0)

        # Update pricing_kernel with actual runtime data
        if 'pricing_kernel' in self.offload_points:
            pk = self.offload_points['pricing_kernel']
            pk.overhead.kernel_cycles = actual_kernel_cycles
            pk.overhead.h2d_latency_ns = actual_h2d_latency
            pk.overhead.d2h_latency_ns = actual_d2h_latency
            pk.heuristics.speedup = actual_speedup

            # Recompute crossover point with actual data
            # crossover = transfer_overhead / (cpu_time_per_elem - kernel_time_per_elem)
            transfer_overhead = actual_h2d_latency + actual_d2h_latency
            x86_total_ns = x86.get('total_execution_ns', 6842852)
            expected_elements = pk.overhead.expected_elements or 64
            cpu_time_per_elem = x86_total_ns / expected_elements if expected_elements > 0 else 1
            kernel_time_per_elem = actual_kernel_cycles  # cycles ~ ns at 1GHz

            if cpu_time_per_elem > kernel_time_per_elem:
                crossover = int(transfer_overhead / (cpu_time_per_elem - kernel_time_per_elem))
                pk.heuristics.crossover_point_elements = max(100, min(crossover, 10000))

            print(f"[FEEDBACK] Updated pricing_kernel with actual runtime:")
            print(f"           Kernel cycles: {actual_kernel_cycles}")
            print(f"           H2D latency: {actual_h2d_latency} ns")
            print(f"           D2H latency: {actual_d2h_latency} ns")
            print(f"           Crossover point: {pk.heuristics.crossover_point_elements} elements")

    def run_overhead_analysis(self, speedup_threshold: float = 1.5):
        """Run overhead analysis to determine offload decisions"""
        print("\n[OVERHEAD ANALYSIS] Running with profiled data...")

        for point_id, point in self.offload_points.items():
            print(f"\n  Analyzing: {point_id}")
            print(f"    Function: {point.function}")
            print(f"    Parallelism: {point.parallelism}")

            # Compute effective speedup considering transfer overhead
            kernel_cycles = point.overhead.kernel_cycles
            h2d_ns = point.overhead.h2d_latency_ns + point.overhead.h2d_transfer_ns
            d2h_ns = point.overhead.d2h_latency_ns + point.overhead.d2h_transfer_ns
            total_overhead_ns = h2d_ns + d2h_ns

            # If dominator analysis allows hoisting/sinking, reduce overhead
            effective_overhead = total_overhead_ns
            if point.dominator.can_hoist_h2d:
                # H2D overhead amortized over multiple iterations
                effective_overhead -= point.overhead.h2d_latency_ns * 0.5
                print(f"    [DOM] Can hoist H2D - reduced overhead")
            if point.dominator.can_sink_d2h:
                # D2H overhead amortized
                effective_overhead -= point.overhead.d2h_latency_ns * 0.5
                print(f"    [DOM] Can sink D2H - reduced overhead")

            # Make decision based on speedup and overhead
            expected_speedup = point.heuristics.speedup
            transfer_dominance = point.heuristics.transfer_dominance_ratio

            if point.parallelism == 'embarrassingly_parallel' and expected_speedup > speedup_threshold:
                if transfer_dominance < 0.8:
                    point.heuristics.decision = 'gpu_always'
                    print(f"    [DECISION] GPU Always (speedup={expected_speedup:.2f}x)")
                else:
                    point.heuristics.decision = 'conditional_offload'
                    print(f"    [DECISION] Conditional (speedup={expected_speedup:.2f}x, transfer_dom={transfer_dominance:.2f})")
            elif point.parallelism == 'tree_traversal' or point.parallelism == 'sequential':
                point.heuristics.decision = 'cpu_only'
                print(f"    [DECISION] CPU Only (parallelism={point.parallelism})")
            elif expected_speedup < 1.0:
                point.heuristics.decision = 'cpu_only'
                print(f"    [DECISION] CPU Only (speedup={expected_speedup:.2f}x < 1.0)")
            else:
                point.heuristics.decision = 'conditional_offload'
                point.heuristics.crossover_point_elements = max(100, point.heuristics.crossover_point_elements)
                print(f"    [DECISION] Conditional (crossover={point.heuristics.crossover_point_elements})")

    def generate_annotations(self, output_path: str):
        """Generate the offload annotations header file"""
        print(f"\n[CODEGEN] Generating annotations to {output_path}")

        lines = [
            "// Auto-generated offload annotations from profile-guided analysis",
            "// Generated by profile_feedback_compiler.py",
            "// DO NOT EDIT - Regenerate using: python3 scripts/profile_feedback_compiler.py",
            "",
            "#ifndef MCF_OFFLOAD_ANNOTATIONS_H",
            "#define MCF_OFFLOAD_ANNOTATIONS_H",
            "",
            "// Offload decision values:",
            "//   0 = CPU only (never offload)",
            "//   1 = GPU always (always offload)",
            "//   2 = Conditional (offload if elements > MIN_ELEMENTS)",
            "",
        ]

        for point_id, point in self.offload_points.items():
            macro_prefix = point_id.upper()

            # Map decision to numeric value
            decision_value = {
                'cpu_only': 0,
                'gpu_always': 1,
                'conditional_offload': 2
            }.get(point.heuristics.decision, 0)

            decision_comment = {
                0: "CPU only",
                1: "GPU always",
                2: "Conditional"
            }.get(decision_value, "Unknown")

            lines.append(f"// {point_id}: {point.description}")
            lines.append(f"#define OFFLOAD_{macro_prefix} {decision_value}  // {decision_comment}")
            lines.append(f"#define {macro_prefix}_MIN_ELEMENTS {point.heuristics.crossover_point_elements}")
            lines.append(f"#define {macro_prefix}_CAN_HOIST_H2D {1 if point.dominator.can_hoist_h2d else 0}")
            lines.append(f"#define {macro_prefix}_CAN_SINK_D2H {1 if point.dominator.can_sink_d2h else 0}")
            lines.append(f"#define {macro_prefix}_H2D_BYTES {point.liveness.h2d_total_bytes}")
            lines.append(f"#define {macro_prefix}_D2H_BYTES {point.liveness.d2h_total_bytes}")
            lines.append(f"#define {macro_prefix}_KERNEL_CYCLES {point.overhead.kernel_cycles}")
            lines.append(f"#define {macro_prefix}_SPEEDUP {point.heuristics.speedup:.2f}")
            lines.append("")

        # Add compiler optimization hints
        lines.extend([
            "// Compiler optimization hints from liveness analysis",
        ])

        hints = self.profile_data.get('compiler_hints', {})
        for point_id, hint_data in hints.items():
            macro_prefix = point_id.upper()
            prefetch = hint_data.get('prefetch_distance', 64)
            coarsening = hint_data.get('thread_coarsening', 4)
            lines.append(f"#define {macro_prefix}_PREFETCH_DISTANCE {prefetch}")
            lines.append(f"#define {macro_prefix}_THREAD_COARSENING {coarsening}")

        lines.extend([
            "",
            "// Global analysis summary",
            f"#define TOTAL_OFFLOAD_POINTS {len(self.offload_points)}",
            f"#define RECOMMENDED_OFFLOADS {sum(1 for p in self.offload_points.values() if p.heuristics.decision != 'cpu_only')}",
            "",
            "#endif // MCF_OFFLOAD_ANNOTATIONS_H",
        ])

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        print(f"  Generated {len(self.offload_points)} offload point annotations")

    def export_compiler_profile(self, output_path: str):
        """Export profile in compiler-compatible format"""
        compiler_profile = {
            "profile_type": "compiler_offload_profile",
            "version": "2.0",
            "generated_by": "profile_feedback_compiler.py",
            "offload_decisions": {},
            "timing_data": {}
        }

        for point_id, point in self.offload_points.items():
            compiler_profile["offload_decisions"][point_id] = {
                "decision": point.heuristics.decision,
                "min_elements": point.heuristics.crossover_point_elements,
                "can_hoist_h2d": point.dominator.can_hoist_h2d,
                "can_sink_d2h": point.dominator.can_sink_d2h,
                "expected_speedup": point.heuristics.speedup
            }

            compiler_profile["timing_data"][point_id] = {
                "kernel_cycles": point.overhead.kernel_cycles,
                "h2d_latency_ns": point.overhead.h2d_latency_ns,
                "d2h_latency_ns": point.overhead.d2h_latency_ns,
                "h2d_bytes": point.liveness.h2d_total_bytes,
                "d2h_bytes": point.liveness.d2h_total_bytes
            }

        with open(output_path, 'w') as f:
            json.dump(compiler_profile, f, indent=2)

        print(f"[EXPORT] Compiler profile saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Profile Feedback Compiler for Heterogeneous Offload')
    parser.add_argument('--profile', required=True,
                       help='Path to per-offload-point profile JSON')
    parser.add_argument('--experiment', required=True,
                       help='Path to experiment results JSON')
    parser.add_argument('--output', required=True,
                       help='Output path for generated annotations header')
    parser.add_argument('--compiler-profile', default=None,
                       help='Output path for compiler profile JSON')
    parser.add_argument('--speedup-threshold', type=float, default=1.5,
                       help='Minimum speedup threshold for offloading')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')

    args = parser.parse_args()

    print("=" * 60)
    print("Profile Feedback Compiler for Heterogeneous Offload")
    print("=" * 60)

    # Initialize compiler with profile data
    compiler = ProfileFeedbackCompiler(args.profile, args.experiment)

    # Step 1: Parse the profile
    print("\n[STEP 1] Parsing profile data...")
    compiler.parse_profile()
    print(f"  Loaded {len(compiler.offload_points)} offload points")

    # Step 2: Update with runtime data
    print("\n[STEP 2] Updating with runtime experiment data...")
    compiler.update_with_runtime_data()

    # Step 3: Run overhead analysis
    print("\n[STEP 3] Running overhead analysis...")
    compiler.run_overhead_analysis(speedup_threshold=args.speedup_threshold)

    # Step 4: Generate annotations
    print("\n[STEP 4] Generating offload annotations...")
    compiler.generate_annotations(args.output)

    # Step 5: Export compiler profile (optional)
    if args.compiler_profile:
        compiler.export_compiler_profile(args.compiler_profile)

    print("\n" + "=" * 60)
    print("Profile feedback compilation complete!")
    print("=" * 60)

    # Summary
    print("\nOffload Decision Summary:")
    for point_id, point in compiler.offload_points.items():
        decision = point.heuristics.decision
        symbol = {'gpu_always': '+', 'conditional_offload': '?', 'cpu_only': '-'}[decision]
        print(f"  [{symbol}] {point_id}: {decision}")
        if decision == 'conditional_offload':
            print(f"      Min elements: {point.heuristics.crossover_point_elements}")

if __name__ == '__main__':
    main()
