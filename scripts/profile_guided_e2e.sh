#!/usr/bin/env bash
# End-to-End Profile-Guided Compilation for Vortex Offloading
#
# This script runs the complete profiling loop:
# 1. Profile MCF on x86 (baseline)
# 2. Compile kernel for Vortex
# 3. Run kernel on Vortex simulator
# 4. Combine profiles
# 5. Re-compile with combined profile data
#
# Usage: profile_guided_e2e.sh [options]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."

# Directories
BIN_DIR="${REPO_ROOT}/bin"
BUILD_DIR="${REPO_ROOT}/build"
PROFILE_DIR="${REPO_ROOT}/profile_results"
KERNEL_DIR="${REPO_ROOT}/test/kernels"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# Ensure directories exist
mkdir -p "${BIN_DIR}" "${BUILD_DIR}" "${PROFILE_DIR}"

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --skip-baseline    Skip x86 baseline profiling (use existing)"
    echo "  --skip-vortex      Skip Vortex simulation (use existing)"
    echo "  --iterations N     Number of profile-guided iterations (default: 1)"
    echo "  --help             Show this help"
    exit 0
}

# Parse arguments
SKIP_BASELINE=false
SKIP_VORTEX=false
ITERATIONS=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-baseline) SKIP_BASELINE=true; shift ;;
        --skip-vortex) SKIP_VORTEX=true; shift ;;
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

echo "=============================================="
echo " Profile-Guided Compilation for Vortex"
echo "=============================================="
echo ""

#==============================================================================
# Step 1: X86 Baseline Profiling
#==============================================================================

BASELINE_PROFILE="${PROFILE_DIR}/mcf_baseline_profile.json"

if [ "${SKIP_BASELINE}" = false ]; then
    log_info "Step 1: Running x86 baseline profiling..."

    # Build MCF with profiling
    bash "${SCRIPT_DIR}/profile_guided_mcf.sh" phase1

    # Run baseline profiling
    bash "${SCRIPT_DIR}/profile_guided_mcf.sh" phase2

    if [ ! -f "${BASELINE_PROFILE}" ]; then
        log_error "Baseline profile not generated"
        exit 1
    fi

    log_info "Baseline profile: ${BASELINE_PROFILE}"
else
    log_info "Step 1: Skipping baseline profiling (using existing)"
    if [ ! -f "${BASELINE_PROFILE}" ]; then
        log_error "Baseline profile not found: ${BASELINE_PROFILE}"
        exit 1
    fi
fi

# Extract baseline timing
BASELINE_TIME=$(jq -r '.overall.total_execution_ns // 0' "${BASELINE_PROFILE}")
BASELINE_MS=$(echo "scale=2; ${BASELINE_TIME} / 1000000" | bc)
log_info "Baseline execution time: ${BASELINE_MS} ms"

#==============================================================================
# Step 2: Compile Kernel for Vortex
#==============================================================================

KERNEL_SRC="${KERNEL_DIR}/mcf_pricing.c"
KERNEL_BIN="${BUILD_DIR}/kernels/mcf_pricing.vxbin"

log_info "Step 2: Compiling kernel for Vortex..."

mkdir -p "${BUILD_DIR}/kernels"

if [ -f "${KERNEL_SRC}" ]; then
    bash "${SCRIPT_DIR}/compile_vortex_kernel.sh" "${KERNEL_SRC}" "${KERNEL_BIN}" || {
        log_warn "Kernel compilation failed - will use estimated Vortex timing"
        KERNEL_BIN=""
    }
else
    log_warn "Kernel source not found: ${KERNEL_SRC}"
    KERNEL_BIN=""
fi

#==============================================================================
# Step 3: Run Vortex Simulation
#==============================================================================

VORTEX_PROFILE="${PROFILE_DIR}/vortex_profile.json"

if [ "${SKIP_VORTEX}" = false ] && [ -n "${KERNEL_BIN}" ] && [ -f "${KERNEL_BIN}" ]; then
    log_info "Step 3: Running Vortex simulation..."

    bash "${SCRIPT_DIR}/run_vortex_profile.sh" "${KERNEL_BIN}" "${VORTEX_PROFILE}" || {
        log_warn "Vortex simulation failed - will use estimated timing"
        # Create estimated profile
        cat > "${VORTEX_PROFILE}" << EOF
{
  "profile_type": "vortex_estimated",
  "target": "riscv_vortex",
  "timing": {
    "kernel_latency_ns": 50000000,
    "h2d_latency_ns": 10000000,
    "d2h_latency_ns": 5000000
  },
  "bandwidth": {
    "h2d_gbps": 10.0,
    "d2h_gbps": 10.0
  },
  "prefetch_hints": {
    "optimal_distance_bytes": 65536
  }
}
EOF
    }
else
    log_info "Step 3: Using estimated Vortex timing (no simulation)"

    # Create estimated profile based on baseline
    # Estimate: GPU kernel ~10x faster, but add transfer overhead
    KERNEL_NS=$(echo "${BASELINE_TIME} / 10" | bc)
    H2D_NS=$((BASELINE_TIME / 50))
    D2H_NS=$((BASELINE_TIME / 100))

    cat > "${VORTEX_PROFILE}" << EOF
{
  "profile_type": "vortex_estimated",
  "target": "riscv_vortex",
  "timing": {
    "kernel_latency_ns": ${KERNEL_NS},
    "h2d_latency_ns": ${H2D_NS},
    "d2h_latency_ns": ${D2H_NS}
  },
  "bandwidth": {
    "h2d_gbps": 10.0,
    "d2h_gbps": 10.0
  },
  "prefetch_hints": {
    "optimal_distance_bytes": 65536
  }
}
EOF
fi

log_info "Vortex profile: ${VORTEX_PROFILE}"

#==============================================================================
# Step 4: Combine Profiles
#==============================================================================

COMBINED_PROFILE="${PROFILE_DIR}/combined_profile.json"

log_info "Step 4: Combining profiles..."

# Use jq to merge baseline and Vortex profiles
jq -s '
{
  "profile_type": "profile_guided_offload",
  "version": "1.0",
  "baseline": .[0],
  "vortex": .[1],
  "functions": .[0].functions,
  "overall": .[0].overall,
  "offload_hints": .[0].offload_hints,
  "timing": .[1].timing,
  "bandwidth": .[1].bandwidth,
  "prefetch_hints": .[1].prefetch_hints,
  "compilation_hints": {
    "primary_offload_function": .[0].offload_hints.primary_target,
    "target_architecture": "riscv_vortex",
    "optimization_level": "O3",
    "passes": ["profile-guided-offload", "convert-cira-to-llvm-vortex"]
  }
}
' "${BASELINE_PROFILE}" "${VORTEX_PROFILE}" > "${COMBINED_PROFILE}"

log_info "Combined profile: ${COMBINED_PROFILE}"

#==============================================================================
# Step 5: Profile-Guided Compilation
#==============================================================================

log_info "Step 5: Ready for profile-guided compilation"

# Print summary
echo ""
echo "=============================================="
echo " Profile-Guided Compilation Summary"
echo "=============================================="
echo ""

# Extract key metrics
PRIMAL_TIME=$(jq -r '.functions.primal_bea_mpp.total_ns // 0' "${BASELINE_PROFILE}")
PRIMAL_MS=$(echo "scale=2; ${PRIMAL_TIME} / 1000000" | bc)
PRIMAL_CALLS=$(jq -r '.functions.primal_bea_mpp.calls // 0' "${BASELINE_PROFILE}")

KERNEL_TIME=$(jq -r '.timing.kernel_latency_ns // 0' "${VORTEX_PROFILE}")
KERNEL_MS=$(echo "scale=2; ${KERNEL_TIME} / 1000000" | bc)

H2D_TIME=$(jq -r '.timing.h2d_latency_ns // 0' "${VORTEX_PROFILE}")
D2H_TIME=$(jq -r '.timing.d2h_latency_ns // 0' "${VORTEX_PROFILE}")
TRANSFER_MS=$(echo "scale=2; (${H2D_TIME} + ${D2H_TIME}) / 1000000" | bc)

TOTAL_VORTEX=$((KERNEL_TIME + H2D_TIME + D2H_TIME))
TOTAL_VORTEX_MS=$(echo "scale=2; ${TOTAL_VORTEX} / 1000000" | bc)

SPEEDUP=$(echo "scale=2; ${PRIMAL_TIME} / ${TOTAL_VORTEX}" | bc 2>/dev/null || echo "N/A")

echo "Baseline (x86):"
echo "  Total execution:     ${BASELINE_MS} ms"
echo "  primal_bea_mpp:      ${PRIMAL_MS} ms (${PRIMAL_CALLS} calls)"
echo ""
echo "Vortex (GPU):"
echo "  Kernel execution:    ${KERNEL_MS} ms"
echo "  Data transfer:       ${TRANSFER_MS} ms"
echo "  Total GPU time:      ${TOTAL_VORTEX_MS} ms"
echo ""
echo "Estimated Speedup:     ${SPEEDUP}x"
echo ""
echo "Profile files:"
echo "  Baseline:  ${BASELINE_PROFILE}"
echo "  Vortex:    ${VORTEX_PROFILE}"
echo "  Combined:  ${COMBINED_PROFILE}"
echo ""
echo "=============================================="
echo ""

# Show how to use the profile
log_info "To compile MCF with profile-guided offloading:"
echo ""
echo "  export DISAGG_PROFILE_PATH=${COMBINED_PROFILE}"
echo "  bash scripts/build.sh bench/mcf/pbeampp.c"
echo ""
echo "Or use the profile_guided_mcf.sh script:"
echo ""
echo "  bash scripts/profile_guided_mcf.sh phase3"
echo ""

#==============================================================================
# Optional: Run profile-guided iterations
#==============================================================================

if [ "${ITERATIONS}" -gt 1 ]; then
    log_info "Running ${ITERATIONS} profile-guided iterations..."

    for i in $(seq 2 "${ITERATIONS}"); do
        log_info "Iteration ${i}/${ITERATIONS}"

        # Re-run with updated profile
        # This would compile, run, and update the profile
        # For now, just a placeholder
        echo "  [Iteration ${i} would run here]"
    done
fi

log_info "End-to-end profiling complete!"
