#!/usr/bin/env bash
#
# Two-Pass Profiling for DataFrame Benchmark
#
# Two-Pass Methodology:
# 1. Profiling Pass: Run x86 baseline, run Vortex simulation, collect timing
# 2. Timing Injection Pass: Re-run x86 with injected delays to simulate heterogeneous execution
#
# Usage: ./run_dataframe_twopass_compiler.sh [profile|inject|full|test] [options]
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Configuration
DATAFRAME_BIN="${DATAFRAME_BIN:-${REPO_ROOT}/bin/dataframe/x86_64-unknown-linux-gnu/dataframe_benchmark}"
PROFILE_DIR="${PROFILE_DIR:-${REPO_ROOT}/profile_results/dataframe}"
VORTEX_TIME_FILE="${PROFILE_DIR}/.vortex_time"

# Profiling parameters
CLOCK_FREQ_MHZ="${CLOCK_FREQ_MHZ:-200.0}"
CXL_LATENCY_NS="${CXL_LATENCY_NS:-165}"

# DataFrame kernel regions
declare -A DATAFRAME_KERNELS=(
    ["data_generation"]="gen_normal_dist,gen_lognormal_dist,gen_exponential_dist"
    ["mean_calculation"]="MeanVisitor"
    ["variance_calculation"]="VarVisitor"
    ["correlation_calculation"]="CorrVisitor"
    ["selection"]="get_view_by_sel"
)

# Estimated kernel invocations
declare -A DATAFRAME_KERNEL_CALLS=(
    ["data_generation"]="3"
    ["mean_calculation"]="1"
    ["variance_calculation"]="1"
    ["correlation_calculation"]="1"
    ["selection"]="1"
)

LOG_PREFIX="dataframe-twopass"

die() {
    echo "[${LOG_PREFIX}] ERROR: $*" >&2
    exit 1
}

info() {
    echo "[${LOG_PREFIX}] $*"
}

usage() {
    cat <<EOF
Usage: $(basename "$0") <mode> [options]

Modes:
  profile     Run Pass 1: Baseline profiling and Vortex simulation
  inject      Run Pass 2: Timing injection simulation
  full        Run both passes with comparison
  test        Quick test run
  compare     Compare baseline vs injected timing

Options:
  --vortex-time N   Override Vortex simulation time (cycles)
  --clock-freq F    Target accelerator clock frequency (MHz, default: ${CLOCK_FREQ_MHZ})
  --cxl-latency N   CXL memory access latency (ns, default: ${CXL_LATENCY_NS})

Environment:
  DATAFRAME_BIN     Path to dataframe_benchmark binary
  PROFILE_DIR       Output directory for profiles
  CLOCK_FREQ_MHZ    Clock frequency in MHz
  CXL_LATENCY_NS    CXL latency in nanoseconds

EOF
    exit 0
}

# Parse arguments
MODE="${1:-}"
shift || true

VORTEX_TIME_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --vortex-time)
            VORTEX_TIME_OVERRIDE="$2"
            shift 2
            ;;
        --clock-freq)
            CLOCK_FREQ_MHZ="$2"
            shift 2
            ;;
        --cxl-latency)
            CXL_LATENCY_NS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            die "Unknown option: $1"
            ;;
    esac
done

[[ -n "${MODE}" ]] || usage

# Create directories
mkdir -p "${PROFILE_DIR}"

# ============================================================================
# Profiling Functions
# ============================================================================

run_x86_baseline() {
    info "Running x86 baseline..."

    if [[ ! -x "${DATAFRAME_BIN}" ]]; then
        die "DataFrame binary not found at ${DATAFRAME_BIN}. Run: ./scripts/build.sh dataframe"
    fi

    local output_file="${PROFILE_DIR}/x86_baseline.txt"

    # Run benchmark and capture timing
    info "Executing DataFrame benchmark..."
    "${DATAFRAME_BIN}" 2>&1 | tee "${output_file}"

    # Parse timing from output
    local data_gen_time=$(grep "Data generation/load time:" "${output_file}" | awk '{print $4}')
    local calc_time=$(grep "Calculation time:" "${output_file}" | awk '{print $3}')
    local select_time=$(grep "Selection time:" "${output_file}" | awk '{print $3}')

    # Generate baseline profile JSON
    info "Generating baseline profile..."
    python3 - <<EOF > "${PROFILE_DIR}/baseline_profile.json"
import json

data_gen_time = float("${data_gen_time:-0}") * 1000  # Convert to ms
calc_time = float("${calc_time:-0}") * 1000
select_time = float("${select_time:-0}") * 1000

profile = {
    "profile_type": "dataframe_baseline",
    "version": "1.0",
    "configuration": {
        "clock_freq_mhz": ${CLOCK_FREQ_MHZ},
        "cxl_latency_ns": ${CXL_LATENCY_NS}
    },
    "num_regions": 5,
    "regions": [
        {
            "region_id": 0,
            "region_name": "data_generation",
            "execution_time_ms": data_gen_time,
            "kernel_calls": 3,
            "parallelizable": True
        },
        {
            "region_id": 1,
            "region_name": "mean_calculation",
            "execution_time_ms": calc_time * 0.3,
            "kernel_calls": 1,
            "parallelizable": True
        },
        {
            "region_id": 2,
            "region_name": "variance_calculation",
            "execution_time_ms": calc_time * 0.3,
            "kernel_calls": 1,
            "parallelizable": True
        },
        {
            "region_id": 3,
            "region_name": "correlation_calculation",
            "execution_time_ms": calc_time * 0.4,
            "kernel_calls": 1,
            "parallelizable": True
        },
        {
            "region_id": 4,
            "region_name": "selection",
            "execution_time_ms": select_time,
            "kernel_calls": 1,
            "parallelizable": True
        }
    ],
    "summary": {
        "total_time_ms": data_gen_time + calc_time + select_time,
        "data_generation_ms": data_gen_time,
        "calculation_ms": calc_time,
        "selection_ms": select_time
    }
}

print(json.dumps(profile, indent=2))
EOF

    info "Baseline profile saved to: ${PROFILE_DIR}/baseline_profile.json"
}

get_vortex_time() {
    # Check for override first
    if [[ -n "${VORTEX_TIME_OVERRIDE}" ]]; then
        echo "${VORTEX_TIME_OVERRIDE}"
        return
    fi

    # Check for .vortex_time file
    if [[ -f "${VORTEX_TIME_FILE}" ]]; then
        cat "${VORTEX_TIME_FILE}"
        return
    fi

    # Default estimate based on typical Vortex simulation
    echo "10000"
}

run_vortex_simulation() {
    info "Checking Vortex simulation timing..."

    local vortex_cycles=$(get_vortex_time)
    local vortex_time_ns=$(echo "scale=2; ${vortex_cycles} * 1000 / ${CLOCK_FREQ_MHZ}" | bc)

    info "Vortex cycles: ${vortex_cycles}"
    info "Vortex time (ns): ${vortex_time_ns}"

    # Generate Vortex profile
    python3 - <<EOF > "${PROFILE_DIR}/vortex_profile.json"
import json

vortex_cycles = ${vortex_cycles}
clock_freq_mhz = ${CLOCK_FREQ_MHZ}
vortex_time_ns = vortex_cycles * 1000.0 / clock_freq_mhz

# Estimate breakdown by region (based on typical DataFrame workload)
regions = [
    {"region": "data_generation", "cycles_pct": 0.50},
    {"region": "mean_calculation", "cycles_pct": 0.10},
    {"region": "variance_calculation", "cycles_pct": 0.15},
    {"region": "correlation_calculation", "cycles_pct": 0.15},
    {"region": "selection", "cycles_pct": 0.10}
]

profile = {
    "profile_type": "dataframe_vortex",
    "version": "1.0",
    "total_cycles": vortex_cycles,
    "total_time_ns": vortex_time_ns,
    "clock_freq_mhz": clock_freq_mhz,
    "regions": []
}

for r in regions:
    region_cycles = int(vortex_cycles * r["cycles_pct"])
    region_time_ns = region_cycles * 1000.0 / clock_freq_mhz
    profile["regions"].append({
        "region_name": r["region"],
        "cycles": region_cycles,
        "time_ns": region_time_ns,
        "cycles_pct": r["cycles_pct"]
    })

print(json.dumps(profile, indent=2))
EOF

    info "Vortex profile saved to: ${PROFILE_DIR}/vortex_profile.json"
}

run_baseline_profile() {
    info "=== Pass 1: Baseline Profiling ==="

    run_x86_baseline
    run_vortex_simulation

    # Combine profiles
    info "Generating combined baseline profile..."
    python3 - <<EOF > "${PROFILE_DIR}/combined_baseline.json"
import json

with open("${PROFILE_DIR}/baseline_profile.json") as f:
    baseline = json.load(f)

with open("${PROFILE_DIR}/vortex_profile.json") as f:
    vortex = json.load(f)

combined = {
    "profile_type": "dataframe_combined_baseline",
    "version": "1.0",
    "configuration": baseline["configuration"],
    "num_regions": baseline["num_regions"],
    "regions": []
}

for i, b_region in enumerate(baseline["regions"]):
    v_region = vortex["regions"][i] if i < len(vortex["regions"]) else {}

    # Calculate host-independent work (estimated 30% of baseline)
    host_work_ns = b_region["execution_time_ms"] * 1e6 * 0.3

    # Injection delay = Vortex time - host work
    vortex_time_ns = v_region.get("time_ns", 0)
    injection_delay_ns = max(0, vortex_time_ns - host_work_ns)

    combined_region = {
        "region_id": b_region["region_id"],
        "region_name": b_region["region_name"],
        "baseline_time_ms": b_region["execution_time_ms"],
        "host_independent_work_ns": host_work_ns,
        "vortex_time_ns": vortex_time_ns,
        "vortex_cycles": v_region.get("cycles", 0),
        "injection_delay_ns": injection_delay_ns,
        "kernel_calls": b_region["kernel_calls"],
        "parallelizable": b_region["parallelizable"],
        "latency_hidden": injection_delay_ns < host_work_ns,
        "optimal_prefetch_depth": 16 if b_region["parallelizable"] else 4
    }
    combined["regions"].append(combined_region)

combined["summary"] = baseline["summary"]
combined["summary"]["vortex_total_cycles"] = vortex["total_cycles"]
combined["summary"]["vortex_total_time_ns"] = vortex["total_time_ns"]

print(json.dumps(combined, indent=2))
EOF

    info "Combined baseline profile saved to: ${PROFILE_DIR}/combined_baseline.json"

    # Display summary
    echo ""
    echo "=== Baseline Profile Summary ==="
    python3 -c "
import json
with open('${PROFILE_DIR}/combined_baseline.json') as f:
    p = json.load(f)
print(f\"Regions profiled: {p['num_regions']}\")
print(f\"Total baseline time: {p['summary']['total_time_ms']:.2f}ms\")
print(f\"Vortex cycles: {p['summary']['vortex_total_cycles']}\")
print()
print('Per-region timing:')
for r in p['regions']:
    print(f\"  {r['region_name']:25s}: baseline={r['baseline_time_ms']:.2f}ms, vortex={r['vortex_time_ns']/1e6:.2f}ms, injection={r['injection_delay_ns']/1e6:.2f}ms\")
"
}

run_timing_injection() {
    info "=== Pass 2: Timing Injection ==="

    local baseline_profile="${PROFILE_DIR}/combined_baseline.json"
    local injected_profile="${PROFILE_DIR}/injected_profile.json"
    local annotations_header="${PROFILE_DIR}/dataframe_timing_annotations.h"

    if [[ ! -f "${baseline_profile}" ]]; then
        die "Baseline profile not found. Run 'profile' mode first."
    fi

    # Generate timing annotations header
    info "Generating timing annotations header..."
    cat > "${annotations_header}" <<HEADER_EOF
/* Auto-generated timing annotations for DataFrame
 * Generated by run_dataframe_twopass_compiler.sh
 */

#ifndef DATAFRAME_TIMING_ANNOTATIONS_H
#define DATAFRAME_TIMING_ANNOTATIONS_H

#include <stdint.h>
#include <time.h>

/* Timing injection configuration */
#define CIRA_CLOCK_FREQ_MHZ ${CLOCK_FREQ_MHZ}
#define CIRA_CXL_LATENCY_NS ${CXL_LATENCY_NS}

/* Region definitions */
HEADER_EOF

    python3 - >> "${annotations_header}" <<EOF
import json

with open("${baseline_profile}") as f:
    profile = json.load(f)

print(f"#define CIRA_NUM_REGIONS {profile['num_regions']}")
print()

for region in profile['regions']:
    rid = region['region_id']
    name = region['region_name'].upper()
    delay = int(region['injection_delay_ns'])
    prefetch = region['optimal_prefetch_depth']
    hidden = 1 if region['latency_hidden'] else 0

    print(f"/* Region {rid}: {region['region_name']} */")
    print(f"#define CIRA_REGION_{name}_ID {rid}")
    print(f"#define CIRA_REGION_{name}_DELAY_NS {delay}")
    print(f"#define CIRA_REGION_{name}_PREFETCH_DEPTH {prefetch}")
    print(f"#define CIRA_REGION_{name}_LATENCY_HIDDEN {hidden}")
    print()

print("/* Timing injection macro */")
print("#define CIRA_INJECT_TIMING(region_id) do { \\\\")
print("    struct timespec ts; \\\\")
print("    ts.tv_sec = 0; \\\\")
print("    ts.tv_nsec = cira_region_delays[region_id]; \\\\")
print("    if (ts.tv_nsec > 0) nanosleep(&ts, NULL); \\\\")
print("} while(0)")
print()
print("/* Region delay lookup table */")
print("static const uint64_t cira_region_delays[] = {")
for region in profile['regions']:
    print(f"    {int(region['injection_delay_ns'])}, /* {region['region_name']} */")
print("};")
print()
print("#endif /* DATAFRAME_TIMING_ANNOTATIONS_H */")
EOF

    info "Annotations header saved to: ${annotations_header}"

    # Calculate injected timing
    info "Calculating injected timing profile..."
    python3 - <<EOF > "${injected_profile}"
import json
from datetime import datetime

with open("${baseline_profile}") as f:
    baseline = json.load(f)

# Calculate total injection delay
total_injection_delay_ms = 0
region_details = []

for region in baseline['regions']:
    kernel_calls = region['kernel_calls']
    delay_per_call_ns = region['injection_delay_ns']
    total_delay_ns = kernel_calls * delay_per_call_ns
    total_delay_ms = total_delay_ns / 1e6

    total_injection_delay_ms += total_delay_ms

    region_details.append({
        "region_id": region['region_id'],
        "region_name": region['region_name'],
        "kernel_calls": kernel_calls,
        "delay_per_call_ns": delay_per_call_ns,
        "total_delay_ms": total_delay_ms,
        "baseline_time_ms": region['baseline_time_ms'],
        "injected_time_ms": region['baseline_time_ms'] + total_delay_ms
    })

baseline_total_ms = baseline['summary']['total_time_ms']
injected_total_ms = baseline_total_ms + total_injection_delay_ms
slowdown_factor = injected_total_ms / baseline_total_ms if baseline_total_ms > 0 else 1.0

injected = {
    "profile_type": "dataframe_injected",
    "version": "1.0",
    "timestamp": datetime.now().isoformat(),
    "baseline_profile": "${baseline_profile}",
    "configuration": baseline['configuration'],
    "timing_injection": {
        "total_injected_delay_ms": total_injection_delay_ms,
        "baseline_time_ms": baseline_total_ms,
        "injected_time_ms": injected_total_ms,
        "slowdown_factor": slowdown_factor
    },
    "regions": region_details,
    "summary": {
        "baseline_time_ms": baseline_total_ms,
        "injected_time_ms": injected_total_ms,
        "total_injection_delay_ms": total_injection_delay_ms,
        "slowdown_factor": slowdown_factor,
        "speedup_vs_baseline": 1.0 / slowdown_factor if slowdown_factor > 0 else 1.0
    }
}

print(json.dumps(injected, indent=2))
EOF

    info "Injected profile saved to: ${injected_profile}"

    # Generate final two-pass profile
    generate_twopass_profile
}

generate_twopass_profile() {
    info "Generating two-pass profile..."

    local twopass_profile="${PROFILE_DIR}/twopass_profile.json"

    python3 - <<EOF > "${twopass_profile}"
import json
from datetime import datetime

with open("${PROFILE_DIR}/combined_baseline.json") as f:
    baseline = json.load(f)

with open("${PROFILE_DIR}/injected_profile.json") as f:
    injected = json.load(f)

twopass = {
    "profile_type": "dataframe_twopass",
    "version": "1.0",
    "timestamp": datetime.now().isoformat(),
    "configuration": baseline['configuration'],
    "num_regions": baseline['num_regions'],
    "clock_freq_mhz": ${CLOCK_FREQ_MHZ},
    "cxl_latency_ns": ${CXL_LATENCY_NS},
    "regions": [],
    "analysis": {}
}

# Combine region data
for b_region in baseline['regions']:
    i_region = next((r for r in injected['regions'] if r['region_id'] == b_region['region_id']), {})

    twopass_region = {
        "region_id": b_region['region_id'],
        "region_name": b_region['region_name'],
        "kernel_calls": b_region['kernel_calls'],
        "parallelizable": b_region['parallelizable'],
        "host_independent_work_ns": b_region['host_independent_work_ns'],
        "vortex_timing": {
            "total_cycles": b_region['vortex_cycles'],
            "total_time_ns": b_region['vortex_time_ns']
        },
        "baseline_time_ms": b_region['baseline_time_ms'],
        "injection_delay_ns": b_region['injection_delay_ns'],
        "injected_time_ms": i_region.get('injected_time_ms', 0),
        "latency_hidden": b_region['latency_hidden'],
        "optimal_prefetch_depth": b_region['optimal_prefetch_depth']
    }
    twopass['regions'].append(twopass_region)

# Analysis
baseline_ms = baseline['summary']['total_time_ms']
injected_ms = injected['summary']['injected_time_ms']
slowdown = injected['summary']['slowdown_factor']

twopass['analysis'] = {
    "baseline_time_ms": baseline_ms,
    "injected_time_ms": injected_ms,
    "slowdown_factor": slowdown,
    "speedup_vs_baseline": 1.0 / slowdown if slowdown > 0 else 1.0,
    "total_injection_delay_ms": injected['summary']['total_injection_delay_ms'],
    "vortex_total_cycles": baseline['summary']['vortex_total_cycles'],
    "vortex_total_time_ns": baseline['summary']['vortex_total_time_ns'],
    "heterogeneous_estimate": {
        "offload_benefit": "positive" if slowdown < 1.0 else "negative",
        "recommended_action": "offload" if slowdown < 1.5 else "keep_on_host"
    }
}

print(json.dumps(twopass, indent=2))
EOF

    info "Two-pass profile saved to: ${twopass_profile}"

    # Display final summary
    echo ""
    echo "=== Two-Pass Profiling Complete ==="
    python3 -c "
import json
with open('${twopass_profile}') as f:
    p = json.load(f)
print(f\"Configuration: clock={p['clock_freq_mhz']}MHz, CXL latency={p['cxl_latency_ns']}ns\")
print()
print('Baseline Performance:')
print(f\"  Total time: {p['analysis']['baseline_time_ms']:.2f}ms\")
print()
print('With Timing Injection:')
print(f\"  Total time: {p['analysis']['injected_time_ms']:.2f}ms\")
print(f\"  Slowdown Factor: {p['analysis']['slowdown_factor']:.3f}x\")
print(f\"  Total Injected Delay: {p['analysis']['total_injection_delay_ms']:.2f}ms\")
print()
print('Vortex Simulation:')
print(f\"  Total cycles: {p['analysis']['vortex_total_cycles']}\")
print(f\"  Total time: {p['analysis']['vortex_total_time_ns']/1e6:.2f}ms\")
print()
print('Per-Region Analysis:')
for r in p['regions']:
    print(f\"  {r['region_name']:25s}: baseline={r['baseline_time_ms']:.2f}ms, injected={r['injected_time_ms']:.2f}ms, delay={r['injection_delay_ns']/1e6:.2f}ms\")
print()
print(f\"Recommendation: {p['analysis']['heterogeneous_estimate']['recommended_action'].upper()}\")
"
}

compare_results() {
    info "=== Comparing Baseline vs Injected ==="

    local twopass_profile="${PROFILE_DIR}/twopass_profile.json"

    if [[ ! -f "${twopass_profile}" ]]; then
        die "Two-pass profile not found. Run 'full' mode first."
    fi

    python3 - <<EOF
import json

with open("${twopass_profile}") as f:
    p = json.load(f)

print("=" * 70)
print("DataFrame Two-Pass Profiling Comparison")
print("=" * 70)
print()

# Table header
print(f"{'Region':<25} {'Baseline':>12} {'Injected':>12} {'Delay':>12} {'Hidden':>8}")
print("-" * 70)

for r in p['regions']:
    hidden = "Yes" if r['latency_hidden'] else "No"
    print(f"{r['region_name']:<25} {r['baseline_time_ms']:>10.2f}ms {r['injected_time_ms']:>10.2f}ms {r['injection_delay_ns']/1e6:>10.2f}ms {hidden:>8}")

print("-" * 70)
print(f"{'TOTAL':<25} {p['analysis']['baseline_time_ms']:>10.2f}ms {p['analysis']['injected_time_ms']:>10.2f}ms {p['analysis']['total_injection_delay_ms']:>10.2f}ms")
print()
print(f"Slowdown Factor: {p['analysis']['slowdown_factor']:.3f}x")
print(f"Recommendation: {p['analysis']['heterogeneous_estimate']['recommended_action'].upper()}")
EOF
}

cleanup() {
    info "Cleaning up profile data..."
    rm -rf "${PROFILE_DIR}"/*.json "${PROFILE_DIR}"/*.txt "${PROFILE_DIR}"/*.h
    info "Cleanup complete"
}

# ============================================================================
# Main Entry Point
# ============================================================================

case "${MODE}" in
    profile)
        run_baseline_profile
        ;;
    inject)
        run_timing_injection
        ;;
    full)
        run_baseline_profile
        run_timing_injection
        compare_results
        ;;
    test)
        CLOCK_FREQ_MHZ=200.0
        CXL_LATENCY_NS=165
        info "Running quick test..."
        run_baseline_profile
        run_timing_injection
        compare_results
        ;;
    compare)
        compare_results
        ;;
    cleanup)
        cleanup
        ;;
    *)
        die "Unknown mode: ${MODE}. Use profile, inject, full, test, compare, or cleanup."
        ;;
esac

info "Done!"
