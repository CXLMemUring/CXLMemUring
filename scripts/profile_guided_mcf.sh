#!/bin/bash
#
# Profile-Guided MCF Compilation for Vortex Offloading
#
# This script:
# 1. Builds and runs MCF on x86 to collect baseline profiling
# 2. Generates profiling hints for the compiler
# 3. Rebuilds MCF with profile-guided Vortex offloading
#

set -e

# Configuration
ROOT_DIR="${ROOT_DIR:-/root/CXLMemUring}"
BUILD_DIR="${ROOT_DIR}/build"
BIN_DIR="${ROOT_DIR}/bin"
PROFILE_DIR="${ROOT_DIR}/profile_results"
VORTEX_DIR="${ROOT_DIR}/vortex"

# Tools
CIRA_BIN="${BUILD_DIR}/bin/cira"
CLANGIR_BIN="${CLANGIR_BIN:-clang}"
# Use GCC for baseline profiling (avoids clangir SROA bugs)
GCC_BIN="${GCC_BIN:-gcc}"
GXX_BIN="${GXX_BIN:-g++}"

# MCF Configuration
MCF_DIR="${ROOT_DIR}/bench/mcf"
MCF_INPUT="${ROOT_DIR}/workloads/mcf/inp.in"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create directories
mkdir -p "${BUILD_DIR}" "${BIN_DIR}" "${PROFILE_DIR}"

#==============================================================================
# Phase 1: Build MCF with profiling instrumentation
#==============================================================================
phase1_build_profiled() {
    log_info "Phase 1: Building MCF with profiling instrumentation"

    # Compile MCF sources with profiling
    PROFILED_SOURCES=(
        "${MCF_DIR}/mcf.c"
        "${MCF_DIR}/psimplex.c"
        "${MCF_DIR}/pbeampp.c"
        "${MCF_DIR}/implicit.c"
        "${MCF_DIR}/mcfutil.c"
        "${MCF_DIR}/output.c"
        "${MCF_DIR}/readmin.c"
        "${MCF_DIR}/treeup.c"
        "${MCF_DIR}/pstart.c"
        "${MCF_DIR}/pflowup.c"
        "${MCF_DIR}/pbla.c"
    )

    # Create stub for 'remote' function (this will be replaced with Vortex offload)
    cat > "${BUILD_DIR}/mcf_remote_stub.c" << 'EOF'
#include "defines.h"
#include <stdio.h>

// BASKET struct from pbeampp.c
typedef struct basket {
    arc_t *a;
    cost_t cost;
    cost_t abs_cost;
} BASKET;

// Check if arc is dual infeasible
static int bea_is_dual_infeasible_stub(arc_t *arc, cost_t red_cost) {
    return (((red_cost < 0) & (arc->ident == AT_LOWER)) |
            ((red_cost > 0) & (arc->ident == AT_UPPER)));
}

static int debug_count = 0;
static int candidates_found = 0;

// Stub implementation of remote() for baseline profiling
// This does the same work as the commented-out code in pbeampp.c
void remote(arc_t *arc, long *basket_size, BASKET *perm[]) {
    debug_count++;
    cost_t red_cost;
    if (arc->ident > BASIC) {
        red_cost = arc->cost - arc->tail->potential + arc->head->potential;
        if (bea_is_dual_infeasible_stub(arc, red_cost)) {
            (*basket_size)++;
            perm[*basket_size]->a = arc;
            perm[*basket_size]->cost = red_cost;
            perm[*basket_size]->abs_cost = (red_cost < 0) ? -red_cost : red_cost;
            candidates_found++;
        }
    }
    if (debug_count == 1000) {
        printf("remote() called 1000 times, candidates found: %d, basket_size: %ld\n",
               candidates_found, *basket_size);
        fflush(stdout);
    }
}
EOF

    # Use GCC for baseline profiling build (avoids clangir SROA bugs)
    # The clangir compiler has bugs in the SROA pass that crash on MCF
    CFLAGS="-O2 -fno-strict-aliasing -DMCF_PROFILING -I${ROOT_DIR}/runtime -I${MCF_DIR}"
    CXXFLAGS="${CFLAGS}"

    # Create profiling wrapper
    cat > "${BUILD_DIR}/mcf_profiled_main.cpp" << 'EOF'
#include "mcf_profiler.h"
#include <cstring>
#include <cstdio>

// External MCF entry point
extern "C" int main_ptr();

// Global profiler
mcf_profile_result_t g_mcf_profile;

int main(int argc, char* argv[]) {
    printf("MCF Profiled starting...\n");
    fflush(stdout);

    // Initialize profiling
    mcf_profile_init();

    printf("Running MCF main_ptr()...\n");
    fflush(stdout);

    // Run MCF
    int result = main_ptr();

    printf("MCF completed with result: %d\n", result);
    fflush(stdout);

    // Finish profiling
    mcf_profile_finish(&g_mcf_profile);

    // Print results
    mcf_profile_print(&g_mcf_profile);

    // Output to JSON
    const char* output_path = "mcf_baseline_profile.json";
    if (argc > 1) {
        output_path = argv[1];
    }
    mcf_profile_to_json(&g_mcf_profile, output_path);

    return result;
}
EOF

    # Build profiled MCF using GCC (avoids clangir SROA bugs)
    log_info "Compiling MCF with profiling using GCC..."

    OBJECTS=""
    for src in "${PROFILED_SOURCES[@]}"; do
        obj="${BUILD_DIR}/$(basename ${src%.c}).o"
        log_info "  Compiling $(basename ${src})"
        ${GCC_BIN} ${CFLAGS} -c "$src" -o "$obj"
        OBJECTS="${OBJECTS} ${obj}"
    done

    # Compile remote stub
    log_info "  Compiling remote stub"
    ${GCC_BIN} ${CFLAGS} -c "${BUILD_DIR}/mcf_remote_stub.c" -o "${BUILD_DIR}/mcf_remote_stub.o"
    OBJECTS="${OBJECTS} ${BUILD_DIR}/mcf_remote_stub.o"

    # Compile profiled main (C++) - define MCF_PROFILER_MAIN to instantiate global context
    ${GXX_BIN} ${CXXFLAGS} -DMCF_PROFILER_MAIN -c "${BUILD_DIR}/mcf_profiled_main.cpp" -o "${BUILD_DIR}/mcf_profiled_main.o"
    OBJECTS="${OBJECTS} ${BUILD_DIR}/mcf_profiled_main.o"

    # Link with g++
    ${GXX_BIN} ${OBJECTS} -o "${BIN_DIR}/mcf_profiled" -lm

    log_info "Built: ${BIN_DIR}/mcf_profiled"
}

#==============================================================================
# Phase 2: Run baseline profiling
#==============================================================================
phase2_run_profiling() {
    log_info "Phase 2: Running x86 baseline profiling"

    if [ ! -f "${BIN_DIR}/mcf_profiled" ]; then
        log_error "Profiled MCF binary not found. Run phase 1 first."
        exit 1
    fi

    # Run MCF and collect profile
    cd "${ROOT_DIR}"
    "${BIN_DIR}/mcf_profiled" "${PROFILE_DIR}/mcf_baseline_profile.json"

    log_info "Baseline profile saved to: ${PROFILE_DIR}/mcf_baseline_profile.json"

    # Display key metrics
    if [ -f "${PROFILE_DIR}/mcf_baseline_profile.json" ]; then
        log_info "Profile summary:"
        python3 << EOF
import json

with open("${PROFILE_DIR}/mcf_baseline_profile.json") as f:
    profile = json.load(f)

print("\nFunction Time Breakdown:")
funcs = profile.get("functions", {})
total = sum(f.get("total_ns", 0) for f in funcs.values())

for name, data in funcs.items():
    ns = data.get("total_ns", 0)
    pct = (ns / total * 100) if total > 0 else 0
    candidate = "GPU" if data.get("offload_candidate", False) else "CPU"
    print(f"  {name}: {ns/1e6:.2f} ms ({pct:.1f}%) - {candidate}")

hints = profile.get("offload_hints", {})
print(f"\nPrimary offload target: {hints.get('primary_target', 'N/A')}")
print(f"Expected speedup: {hints.get('expected_speedup_pricing', 1.0):.1f}x")
EOF
    fi
}

#==============================================================================
# Phase 3: Build with profile-guided Vortex offloading
#==============================================================================
phase3_build_vortex() {
    log_info "Phase 3: Building MCF with profile-guided Vortex offloading"

    if [ ! -f "${PROFILE_DIR}/mcf_baseline_profile.json" ]; then
        log_error "Baseline profile not found. Run phase 2 first."
        exit 1
    fi

    # Check if CIRA binary exists
    if [ ! -f "${CIRA_BIN}" ]; then
        log_warn "CIRA binary not found at ${CIRA_BIN}"
        log_info "Building CIRA..."

        mkdir -p "${BUILD_DIR}"
        cd "${BUILD_DIR}"
        cmake "${ROOT_DIR}" -DCMAKE_BUILD_TYPE=Release
        make -j$(nproc) cira
    fi

    # Use the build script with profile-guided options
    log_info "Running profile-guided compilation..."

    export DISAGG_PROFILE_PATH="${PROFILE_DIR}/mcf_baseline_profile.json"
    export OFFLOAD_TARGET="vortex"
    export PIPELINE_LLVM_LOWERING="direct"

    # Build MCF through CIRA pipeline
    cd "${ROOT_DIR}"
    bash scripts/build.sh mcf

    log_info "Built profile-guided MCF"
}

#==============================================================================
# Phase 4: Generate Vortex kernel from CIR
#==============================================================================
phase4_generate_kernel() {
    log_info "Phase 4: Generating Vortex kernel from profile-guided CIR"

    # Check for CIR output
    CIR_FILE="${BUILD_DIR}/mcf/pbeampp.cir"

    if [ ! -f "${CIR_FILE}" ]; then
        log_warn "CIR file not found at ${CIR_FILE}"
        log_info "Using pre-defined kernel template..."

        # The CIRA compiler generates kernels automatically
        # For now, we use the kernel we created
        KERNEL_SRC="${ROOT_DIR}/test/kernels/mcf_pricing.c"
        KERNEL_BIN="${BUILD_DIR}/kernels/mcf_pricing.vxbin"

        mkdir -p "${BUILD_DIR}/kernels"

        if [ -f "${ROOT_DIR}/scripts/compile_vortex_kernel.sh" ]; then
            bash "${ROOT_DIR}/scripts/compile_vortex_kernel.sh" \
                "${KERNEL_SRC}" "${KERNEL_BIN}"
            log_info "Kernel compiled: ${KERNEL_BIN}"
        else
            log_warn "Vortex kernel compilation script not found"
        fi
    else
        log_info "Found CIR file: ${CIR_FILE}"

        # Apply profile-guided offload pass
        if [ -f "${CIRA_BIN}" ]; then
            ${CIRA_BIN} "${CIR_FILE}" \
                --profile-guided-offload \
                --offload-profile="${PROFILE_DIR}/mcf_baseline_profile.json" \
                --offload-target=vortex \
                --min-offload-elements=1000 \
                --speedup-threshold=1.5 \
                -o "${BUILD_DIR}/mcf/pbeampp_offload.mlir"

            log_info "Applied profile-guided offload annotations"
        fi
    fi
}

#==============================================================================
# Phase 5: Create combined profile for compiler
#==============================================================================
phase5_create_compiler_profile() {
    log_info "Phase 5: Creating combined compiler profile"

    # Merge baseline profile with offload timing (if available)
    python3 << EOF
import json
import os

profile_dir = "${PROFILE_DIR}"
output_file = os.path.join(profile_dir, "compiler_profile.json")

# Load baseline profile
baseline_file = os.path.join(profile_dir, "mcf_baseline_profile.json")
baseline = {}
if os.path.exists(baseline_file):
    with open(baseline_file) as f:
        baseline = json.load(f)

# Load Vortex timing profile (if available)
vortex_file = os.path.join(profile_dir, "vortex_timing.json")
vortex = {}
if os.path.exists(vortex_file):
    with open(vortex_file) as f:
        vortex = json.load(f)

# Create compiler profile
compiler_profile = {
    "profile_type": "profile_guided_offload",
    "version": "1.0",
    "baseline": baseline,
    "vortex_timing": vortex,
    "compilation_hints": {
        "primary_offload_function": baseline.get("offload_hints", {}).get("primary_target", "primal_bea_mpp"),
        "target_architecture": "riscv_vortex",
        "optimization_level": "O3",
        "passes": [
            "profile-guided-offload",
            "convert-cira-to-llvm-vortex"
        ]
    },
    "prefetch_config": {
        "enabled": True,
        "distance_bytes": vortex.get("prefetch_hints", {}).get("optimal_distance_bytes", 65536),
        "strategy": "distance_based"
    }
}

with open(output_file, 'w') as f:
    json.dump(compiler_profile, f, indent=2)

print(f"Compiler profile written to: {output_file}")
EOF

    log_info "Combined compiler profile created"
}

#==============================================================================
# Main workflow
#==============================================================================
main() {
    log_info "=== Profile-Guided MCF Compilation for Vortex ==="
    echo ""

    case "${1:-all}" in
        phase1|build-profiled)
            phase1_build_profiled
            ;;
        phase2|profile)
            phase2_run_profiling
            ;;
        phase3|build-vortex)
            phase3_build_vortex
            ;;
        phase4|kernel)
            phase4_generate_kernel
            ;;
        phase5|compiler-profile)
            phase5_create_compiler_profile
            ;;
        all)
            phase1_build_profiled
            phase2_run_profiling
            phase5_create_compiler_profile
            phase3_build_vortex
            phase4_generate_kernel
            ;;
        *)
            echo "Usage: $0 [phase1|phase2|phase3|phase4|phase5|all]"
            echo ""
            echo "Phases:"
            echo "  phase1 (build-profiled)    - Build MCF with profiling"
            echo "  phase2 (profile)           - Run baseline profiling"
            echo "  phase3 (build-vortex)      - Build with Vortex offloading"
            echo "  phase4 (kernel)            - Generate Vortex kernel"
            echo "  phase5 (compiler-profile)  - Create compiler profile"
            echo "  all                        - Run all phases"
            exit 1
            ;;
    esac

    echo ""
    log_info "=== Complete ==="
}

main "$@"
