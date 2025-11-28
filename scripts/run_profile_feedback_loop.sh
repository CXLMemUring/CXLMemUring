#!/bin/bash
#
# Profile Feedback Loop for Heterogeneous MCF
#
# This script orchestrates the profile-guided recompilation:
# 1. Run overhead analysis with profiled data
# 2. Regenerate offload annotations
# 3. Rebuild the heterogeneous binary
# 4. Optionally run the new binary to verify
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
step() { echo -e "${BLUE}[STEP]${NC} $1"; }

# Default paths
PROFILE_JSON="${PROJECT_ROOT}/profile_results/per_offload_point_profile.json"
EXPERIMENT_JSON="${PROJECT_ROOT}/profile_results/experiment_results.json"
ANNOTATIONS_H="${PROJECT_ROOT}/bin/mcf_heterogeneous/mcf_offload_annotations.h"
COMPILER_PROFILE="${PROJECT_ROOT}/profile_results/compiler_offload_profile.json"

# Options
SPEEDUP_THRESHOLD=1.5
RUN_VERIFICATION=false
VERBOSE=false

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --profile PATH      Path to per-offload-point profile JSON"
    echo "  --experiment PATH   Path to experiment results JSON"
    echo "  --output PATH       Output path for annotations header"
    echo "  --threshold FLOAT   Speedup threshold for offloading (default: 1.5)"
    echo "  --verify            Run verification after rebuild"
    echo "  --verbose           Enable verbose output"
    echo "  -h, --help          Show this help"
    echo ""
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --profile)
            PROFILE_JSON="$2"
            shift 2
            ;;
        --experiment)
            EXPERIMENT_JSON="$2"
            shift 2
            ;;
        --output)
            ANNOTATIONS_H="$2"
            shift 2
            ;;
        --threshold)
            SPEEDUP_THRESHOLD="$2"
            shift 2
            ;;
        --verify)
            RUN_VERIFICATION=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

echo "============================================================"
echo "Profile Feedback Loop for Heterogeneous MCF"
echo "============================================================"
echo ""

# Check input files exist
if [[ ! -f "$PROFILE_JSON" ]]; then
    error "Profile file not found: $PROFILE_JSON"
fi

if [[ ! -f "$EXPERIMENT_JSON" ]]; then
    error "Experiment file not found: $EXPERIMENT_JSON"
fi

# Step 1: Show current profile data
step "1/4: Loading profile data"
info "Profile: $PROFILE_JSON"
info "Experiment: $EXPERIMENT_JSON"

if [[ "$VERBOSE" == "true" ]]; then
    echo ""
    echo "Current experiment results:"
    cat "$EXPERIMENT_JSON" | python3 -m json.tool 2>/dev/null || cat "$EXPERIMENT_JSON"
    echo ""
fi

# Step 2: Run the profile feedback compiler
step "2/4: Running overhead analysis and generating annotations"

COMPILER_ARGS=(
    --profile "$PROFILE_JSON"
    --experiment "$EXPERIMENT_JSON"
    --output "$ANNOTATIONS_H"
    --compiler-profile "$COMPILER_PROFILE"
    --speedup-threshold "$SPEEDUP_THRESHOLD"
)

if [[ "$VERBOSE" == "true" ]]; then
    COMPILER_ARGS+=(--verbose)
fi

python3 "${SCRIPT_DIR}/profile_feedback_compiler.py" "${COMPILER_ARGS[@]}"

if [[ $? -ne 0 ]]; then
    error "Profile feedback compiler failed"
fi

info "Generated annotations: $ANNOTATIONS_H"
info "Generated compiler profile: $COMPILER_PROFILE"

# Show generated annotations
if [[ "$VERBOSE" == "true" ]]; then
    echo ""
    echo "Generated annotations:"
    cat "$ANNOTATIONS_H"
    echo ""
fi

# Step 3: Rebuild the heterogeneous binary
step "3/4: Rebuilding heterogeneous MCF binary"

bash "${SCRIPT_DIR}/build_mcf_heterogeneous.sh" cpu

if [[ $? -ne 0 ]]; then
    error "Build failed"
fi

BINARY="${PROJECT_ROOT}/bin/mcf_heterogeneous/build/mcf_heterogeneous"
if [[ ! -f "$BINARY" ]]; then
    error "Binary not found: $BINARY"
fi

info "Binary built: $BINARY"

# Step 4: Optionally run verification
if [[ "$RUN_VERIFICATION" == "true" ]]; then
    step "4/4: Running verification"

    WORKLOAD="${PROJECT_ROOT}/workloads/mcf/inp.in"
    if [[ ! -f "$WORKLOAD" ]]; then
        warn "Workload not found: $WORKLOAD"
    else
        info "Running: $BINARY $WORKLOAD"
        echo ""

        START_TIME=$(date +%s%N)
        "$BINARY" "$WORKLOAD"
        EXIT_CODE=$?
        END_TIME=$(date +%s%N)

        ELAPSED_MS=$(( (END_TIME - START_TIME) / 1000000 ))

        echo ""
        if [[ $EXIT_CODE -eq 0 ]]; then
            info "Verification PASSED (exit code: $EXIT_CODE)"
            info "Execution time: ${ELAPSED_MS} ms"
        else
            warn "Verification completed with exit code: $EXIT_CODE"
        fi
    fi
else
    step "4/4: Skipping verification (use --verify to enable)"
fi

echo ""
echo "============================================================"
echo "Profile Feedback Loop Complete!"
echo "============================================================"
echo ""
echo "Summary:"
echo "  - Annotations: $ANNOTATIONS_H"
echo "  - Compiler profile: $COMPILER_PROFILE"
echo "  - Binary: $BINARY"
echo ""
echo "Next steps:"
echo "  1. Run the binary: $BINARY <workload>"
echo "  2. Profile the execution to collect new timing data"
echo "  3. Re-run this script with updated profile data"
echo ""
