#!/bin/bash

#=============================================================================
# GPU Offloading Compiler for All CXLMemUring Workloads
#
# This script automatically applies GPU offloading passes to all benchmark
# workloads with workload-specific configuration parameters.
#
# Usage:
#   ./scripts/gpu_offload_all_workloads.sh [--dry-run] [--verbose]
#=============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CXLMEM_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$CXLMEM_DIR/build"
MLIR_OPT="$BUILD_DIR/bin/cira"

# Options
DRY_RUN=false
VERBOSE=false
PARALLEL=4

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --parallel)
            PARALLEL=$2
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

#=============================================================================
# Logging Functions
#=============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_workload() {
    echo -e "${BLUE}===================================================${NC}"
    echo -e "${BLUE}Workload: $1${NC}"
    echo -e "${BLUE}===================================================${NC}"
}

#=============================================================================
# Check Prerequisites
#=============================================================================

check_prerequisites() {
    log_info "Checking prerequisites..."

    if [ ! -f "$MLIR_OPT" ]; then
        log_error "cira not found at $MLIR_OPT"
        echo ""
        echo "Please build CXLMemUring first:"
        echo "  cd $CXLMEM_DIR"
        echo "  mkdir -p build && cd build"
        echo "  cmake .. && make -j\$(nproc)"
        exit 1
    fi

    log_success "cira found: $MLIR_OPT"
    echo ""
}

#=============================================================================
# Workload Configurations
#=============================================================================

declare -A WORKLOAD_INTENSITY
declare -A WORKLOAD_SPEEDUP
declare -A WORKLOAD_MEMORY
declare -A WORKLOAD_BUDGET
declare -A WORKLOAD_EXPECTED_SPEEDUP

# DataFrame - Data aggregation and filtering
WORKLOAD_INTENSITY[dataframe]=8.0
WORKLOAD_SPEEDUP[dataframe]=1.15
WORKLOAD_MEMORY[dataframe]=1073741824
WORKLOAD_BUDGET[dataframe]=536870912
WORKLOAD_EXPECTED_SPEEDUP[dataframe]="1.2-1.6x"

# GAPBS - Graph algorithms
WORKLOAD_INTENSITY[gapbs]=12.0
WORKLOAD_SPEEDUP[gapbs]=1.2
WORKLOAD_MEMORY[gapbs]=2147483648
WORKLOAD_BUDGET[gapbs]=1073741824
WORKLOAD_EXPECTED_SPEEDUP[gapbs]="2.0-3.5x"

# LLaMA - Language models (HIGHEST SPEEDUP)
WORKLOAD_INTENSITY[llama]=20.0
WORKLOAD_SPEEDUP[llama]=1.5
WORKLOAD_MEMORY[llama]=4294967296
WORKLOAD_BUDGET[llama]=2147483648
WORKLOAD_EXPECTED_SPEEDUP[llama]="3.0-5.0x"

# MCF - Optimization
WORKLOAD_INTENSITY[mcf]=10.0
WORKLOAD_SPEEDUP[mcf]=1.2
WORKLOAD_MEMORY[mcf]=536870912
WORKLOAD_BUDGET[mcf]=268435456
WORKLOAD_EXPECTED_SPEEDUP[mcf]="1.3-1.8x"

# MonetDB - Database
WORKLOAD_INTENSITY[monetdb]=9.0
WORKLOAD_SPEEDUP[monetdb]=1.15
WORKLOAD_MEMORY[monetdb]=2147483648
WORKLOAD_BUDGET[monetdb]=1073741824
WORKLOAD_EXPECTED_SPEEDUP[monetdb]="1.5-2.2x"

# NPB - Numerical benchmarks
WORKLOAD_INTENSITY[npb]=15.0
WORKLOAD_SPEEDUP[npb]=1.3
WORKLOAD_MEMORY[npb]=1073741824
WORKLOAD_BUDGET[npb]=536870912
WORKLOAD_EXPECTED_SPEEDUP[npb]="2.5-4.0x"

# Hash Join
WORKLOAD_INTENSITY[hashjoin]=10.0
WORKLOAD_SPEEDUP[hashjoin]=1.2
WORKLOAD_MEMORY[hashjoin]=1073741824
WORKLOAD_BUDGET[hashjoin]=536870912
WORKLOAD_EXPECTED_SPEEDUP[hashjoin]="1.8-2.5x"

# Spatter - Memory patterns (LOW SPEEDUP)
WORKLOAD_INTENSITY[spatter]=2.0
WORKLOAD_SPEEDUP[spatter]=1.05
WORKLOAD_MEMORY[spatter]=2147483648
WORKLOAD_BUDGET[spatter]=1073741824
WORKLOAD_EXPECTED_SPEEDUP[spatter]="0.8-1.2x"

# UME - Mesh operations
WORKLOAD_INTENSITY[ume]=11.0
WORKLOAD_SPEEDUP[ume]=1.2
WORKLOAD_MEMORY[ume]=1610612736
WORKLOAD_BUDGET[ume]=805306368
WORKLOAD_EXPECTED_SPEEDUP[ume]="1.5-2.3x"

#=============================================================================
# Workload Compilation Function
#=============================================================================

compile_workload() {
    local workload_name=$1
    local mlir_file=$2

    # Get configuration
    local intensity=${WORKLOAD_INTENSITY[$workload_name]}
    local speedup=${WORKLOAD_SPEEDUP[$workload_name]}
    local memory=${WORKLOAD_MEMORY[$workload_name]}
    local budget=${WORKLOAD_BUDGET[$workload_name]}
    local expected=${WORKLOAD_EXPECTED_SPEEDUP[$workload_name]}

    # Check if file exists
    if [ ! -f "$mlir_file" ]; then
        log_warning "File not found: $mlir_file"
        return 1
    fi

    log_workload "$workload_name"

    echo "Configuration:"
    echo "  Input: $mlir_file"
    echo "  Compute intensity: $intensity FLOPs/byte"
    echo "  Min speedup: ${speedup}x"
    echo "  Max memory: $((memory / 1024 / 1024 / 1024)) GB"
    echo "  GPU budget: $((budget / 1024 / 1024 / 1024)) GB"
    echo "  Expected speedup: $expected"
    echo ""

    local output_file="${mlir_file%.mlir}_gpu.mlir"
    local report_file="${mlir_file%.mlir}_gpu_analysis.txt"

    # Build pass pipeline
    local pipeline="builtin.module(
        gpu-offload-decision{
            min-compute-intensity=$intensity
            min-speedup=$speedup
            max-memory-bytes=$memory
            use-coherent-memory=true
        },
        gpu-kernel-gen{gpu-memory-budget=$budget},
        gpu-memory-opt
    )"

    if [ "$DRY_RUN" = true ]; then
        log_info "DRY RUN: Would execute:"
        echo "  $MLIR_OPT -pass-pipeline='$pipeline' \\
            $mlir_file -o $output_file"
        return 0
    fi

    # Execute compilation
    if [ "$VERBOSE" = true ]; then
        log_info "Executing GPU offloading passes..."
    fi

    if $MLIR_OPT -pass-pipeline="$pipeline" "$mlir_file" \
        -o "$output_file" 2>"$report_file"; then

        log_success "Compiled: $output_file"

        # Show analysis summary
        if [ "$VERBOSE" = true ]; then
            echo ""
            echo "Analysis Report:"
            head -30 "$report_file" | sed 's/^/  /'
            echo ""
        fi

        return 0
    else
        log_error "Compilation failed for $workload_name"
        if [ -f "$report_file" ]; then
            echo "Error details:"
            cat "$report_file" | sed 's/^/  /'
        fi
        return 1
    fi
}

#=============================================================================
# Find and Compile Workloads
#=============================================================================

find_and_compile() {
    log_info "Scanning for workload files..."
    echo ""

    local total=0
    local success=0
    local failed=0

    # Find all MLIR files in bench directory
    while IFS= read -r mlir_file; do
        if [ -f "$mlir_file" ]; then
            total=$((total + 1))

            # Determine workload type from path
            local workload=""
            case "$mlir_file" in
                *DataFrame*) workload="dataframe" ;;
                *gapbs*) workload="gapbs" ;;
                *llama*) workload="llama" ;;
                *mcf*) workload="mcf" ;;
                *MonetDB*) workload="monetdb" ;;
                *npb*) workload="npb" ;;
                *hash-join*|*partitioned*) workload="hashjoin" ;;
                *spatter*) workload="spatter" ;;
                *ume*) workload="ume" ;;
                *) workload="unknown" ;;
            esac

            if [ -n "${WORKLOAD_INTENSITY[$workload]}" ]; then
                if compile_workload "$workload" "$mlir_file"; then
                    success=$((success + 1))
                else
                    failed=$((failed + 1))
                fi
            else
                log_warning "Unknown workload type in: $mlir_file"
            fi
        fi
    done < <(find "$CXLMEM_DIR/bench" -name "*.mlir" -type f 2>/dev/null)

    echo ""
    echo "========================================="
    echo "Compilation Summary"
    echo "========================================="
    echo "Total MLIR files found: $total"
    log_success "Successfully compiled: $success"
    if [ $failed -gt 0 ]; then
        log_error "Failed: $failed"
    fi
    echo ""
}

#=============================================================================
# Generate Report
#=============================================================================

generate_report() {
    log_info "Generating summary report..."

    local report_file="$CXLMEM_DIR/GPU_OFFLOADING_REPORT.txt"

    cat > "$report_file" << 'EOF'
================================================================
CXLMemUring GPU Offloading Compilation Report
================================================================

Generated: $(date)
Build Directory: $BUILD_DIR

Workload Status:
================================================================
EOF

    for workload in dataframe gapbs llama mcf monetdb npb hashjoin spatter ume; do
        echo "" >> "$report_file"
        echo "Workload: $workload" >> "$report_file"
        echo "  Intensity Threshold: ${WORKLOAD_INTENSITY[$workload]}" >> "$report_file"
        echo "  Speedup Threshold: ${WORKLOAD_SPEEDUP[$workload]}x" >> "$report_file"
        echo "  Expected Speedup: ${WORKLOAD_EXPECTED_SPEEDUP[$workload]}" >> "$report_file"
    done

    log_success "Report saved to: $report_file"
}

#=============================================================================
# Main
#=============================================================================

main() {
    echo "========================================="
    echo "GPU Offloading for CXLMemUring Workloads"
    echo "========================================="
    echo ""

    check_prerequisites

    if [ "$DRY_RUN" = true ]; then
        log_warning "Running in DRY RUN mode (no files will be modified)"
        echo ""
    fi

    find_and_compile
    generate_report

    if [ "$DRY_RUN" = false ]; then
        echo "Next steps:"
        echo "1. Review generated MLIR files (*_gpu.mlir)"
        echo "2. Compare performance: baseline vs GPU-optimized"
        echo "3. Validate correctness of results"
        echo "4. Update benchmark suite with optimized versions"
    fi
}

main
