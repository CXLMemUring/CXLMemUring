#!/usr/bin/env bash
# Export experiment results to compiler-compatible profile format
#
# This script converts the experiment_results.json from RTL simulation
# into a format that can be used by the profile-guided offload pass.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
PROFILE_DIR="${REPO_ROOT}/profile_results"

# Input experiment results
EXPERIMENT_RESULTS="${PROFILE_DIR}/experiment_results.json"

# Output compiler profile
COMPILER_PROFILE="${PROFILE_DIR}/compiler_offload_profile.json"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }

#==============================================================================
# Parse experiment results and generate compiler profile
#==============================================================================

generate_compiler_profile() {
    log_info "Generating compiler profile from experiment results..."

    if [ ! -f "${EXPERIMENT_RESULTS}" ]; then
        echo "Error: Experiment results not found: ${EXPERIMENT_RESULTS}"
        exit 1
    fi

    # Read values from experiment results using Python
    python3 << EOF
import json
import sys

# Load experiment results
with open("${EXPERIMENT_RESULTS}", 'r') as f:
    results = json.load(f)

# Extract values
x86 = results.get('x86_baseline', {})
vortex = results.get('vortex_execution', {})
analysis = results.get('analysis', {})
hints = results.get('offload_hints', {})

# Generate compiler profile
compiler_profile = {
    "profile_type": "compiler_offload_profile",
    "version": "2.0",
    "source": "experiment_results",

    # Baseline timing
    "baseline": {
        "profile_type": "x86_baseline",
        "target": "x86_64",
        "total_execution_ns": x86.get('total_execution_ns', 0),
        "functions": {
            "primal_bea_mpp": {
                "calls": 1,
                "total_ns": x86.get('total_execution_ns', 0),
                "avg_ns": x86.get('total_execution_ns', 0),
                "arcs_processed": 64,
                "offload_candidate": True,
                "parallelism": "embarrassingly_parallel"
            }
        },
        "offload_hints": {
            "primary_target": hints.get('primary_target', 'primal_bea_mpp'),
            "expected_speedup_pricing": analysis.get('kernel_speedup', 1.0),
            "min_arcs_for_offload": hints.get('min_arcs_for_offload', 1000),
            "data_transfer_cost_factor": 0.1
        }
    },

    # Vortex timing
    "timing": {
        "kernel_cycles": vortex.get('kernel_cycles', 0),
        "kernel_latency_ns": vortex.get('kernel_latency_ns', 0),
        "h2d_latency_ns": vortex.get('h2d_latency_ns', 1000000),
        "d2h_latency_ns": vortex.get('d2h_latency_ns', 500000),
        "total_offload_ns": vortex.get('total_offload_ns', 0)
    },

    # Bandwidth
    "bandwidth": {
        "h2d_gbps": 10.0,
        "d2h_gbps": 10.0
    },

    # Analysis results
    "overhead_analysis": {
        "h2d_bytes": 1536,  # 64 arcs * 6 arrays * 4 bytes
        "d2h_bytes": 512,   # 64 arcs * 2 arrays * 4 bytes
        "kernel_speedup": analysis.get('kernel_speedup', 1.0),
        "total_speedup": analysis.get('total_speedup_with_overhead', 1.0),
        "should_offload": analysis.get('offload_beneficial', False)
    },

    # Offload decision
    "offload_decision": {
        "expected_speedup": analysis.get('kernel_speedup', 1.0),
        "total_speedup_with_overhead": analysis.get('total_speedup_with_overhead', 1.0),
        "should_offload": analysis.get('offload_beneficial', False),
        "reason": analysis.get('note', 'Profile-guided decision'),
        "target_device": "vortex"
    },

    # Heuristics for compiler
    "heuristics": {
        "min_elements_for_offload": hints.get('min_arcs_for_offload', 1000),
        "speedup_threshold": 1.2,
        "transfer_dominance_threshold": 0.1,
        "parallelism_speedup": {
            "embarrassingly_parallel": analysis.get('kernel_speedup', 10.0),
            "reduction": 5.0,
            "data_dependent": 3.0,
            "tree_traversal": 1.5
        }
    }
}

# Write output
with open("${COMPILER_PROFILE}", 'w') as f:
    json.dump(compiler_profile, f, indent=2)

print(f"Generated compiler profile: ${COMPILER_PROFILE}")

# Print summary
print("\n=== Compiler Profile Summary ===")
print(f"  Kernel speedup:     {analysis.get('kernel_speedup', 1.0):.2f}x")
print(f"  Total speedup:      {analysis.get('total_speedup_with_overhead', 1.0):.2f}x")
print(f"  Should offload:     {analysis.get('offload_beneficial', False)}")
print(f"  H2D latency:        {vortex.get('h2d_latency_ns', 0) / 1000000:.2f} ms")
print(f"  D2H latency:        {vortex.get('d2h_latency_ns', 0) / 1000000:.2f} ms")
print(f"  Kernel cycles:      {vortex.get('kernel_cycles', 0)}")
EOF

    log_info "Compiler profile generated: ${COMPILER_PROFILE}"
}

#==============================================================================
# Generate usage instructions
#==============================================================================

print_usage_instructions() {
    log_info ""
    log_info "=== Usage Instructions ==="
    log_info ""
    log_info "To use this profile with the CIR compiler:"
    log_info ""
    log_info "1. Profile-guided offload pass:"
    log_info "   cira-opt --profile-guided-offload \\"
    log_info "       --offload-profile=${COMPILER_PROFILE} \\"
    log_info "       --speedup-threshold=1.2 \\"
    log_info "       input.mlir -o output.mlir"
    log_info ""
    log_info "2. Enhanced overhead analysis pass:"
    log_info "   cira-opt --enhanced-offload-analysis \\"
    log_info "       --experiment-results=${EXPERIMENT_RESULTS} \\"
    log_info "       --speedup-threshold=1.2 \\"
    log_info "       --verbose \\"
    log_info "       input.mlir -o output.mlir"
    log_info ""
    log_info "3. Full compilation pipeline:"
    log_info "   See scripts/compile_with_profile.sh"
    log_info ""
}

#==============================================================================
# Main
#==============================================================================

main() {
    log_info "=== Export Profile to Compiler Format ==="
    log_info ""

    generate_compiler_profile
    print_usage_instructions

    log_info "Export complete!"
}

main "$@"
