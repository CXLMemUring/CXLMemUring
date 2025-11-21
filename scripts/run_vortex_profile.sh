#!/usr/bin/env bash
# Run kernel on Vortex simulator and capture performance metrics
# Outputs timing data to JSON for profile-guided compilation
#
# Usage: run_vortex_profile.sh <kernel.vxbin> <output_profile.json> [simulator_args...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
VORTEX_HOME="${REPO_ROOT}/vortex"

# Vortex simulator
VORTEX_SIM="${VORTEX_HOME}/sim/simx/simx"
VORTEX_RTL="${VORTEX_HOME}/sim/rtlsim/rtlsim"

# Use SimX by default, can override with VORTEX_SIMULATOR env var
SIMULATOR="${VORTEX_SIMULATOR:-${VORTEX_SIM}}"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <kernel.vxbin> <output_profile.json> [simulator_args...]"
    echo ""
    echo "Runs kernel on Vortex simulator and captures performance metrics"
    echo ""
    echo "Environment variables:"
    echo "  VORTEX_SIMULATOR - Path to simulator (default: simx)"
    echo "  VORTEX_CLUSTERS  - Number of clusters (default: 1)"
    echo "  VORTEX_CORES     - Cores per cluster (default: 1)"
    echo "  VORTEX_WARPS     - Warps per core (default: 4)"
    echo "  VORTEX_THREADS   - Threads per warp (default: 4)"
    exit 1
fi

KERNEL="$1"
OUTPUT_PROFILE="$2"
shift 2

EXTRA_ARGS=("$@")

if [ ! -f "${KERNEL}" ]; then
    echo "Error: Kernel file '${KERNEL}' not found"
    exit 1
fi

if [ ! -x "${SIMULATOR}" ]; then
    echo "Error: Vortex simulator not found at ${SIMULATOR}"
    echo "Build Vortex first: cd ${VORTEX_HOME} && make -C sim/simx"
    exit 1
fi

# Configuration
CLUSTERS="${VORTEX_CLUSTERS:-1}"
CORES="${VORTEX_CORES:-1}"
WARPS="${VORTEX_WARPS:-4}"
THREADS="${VORTEX_THREADS:-4}"

echo "[Vortex Profiling Run]"
echo "  Kernel:    ${KERNEL}"
echo "  Simulator: ${SIMULATOR}"
echo "  Config:    ${CLUSTERS}c x ${CORES}cores x ${WARPS}w x ${THREADS}t"
echo ""

# Create temporary file for simulator output
TEMP_OUTPUT=$(mktemp)
trap "rm -f ${TEMP_OUTPUT}" EXIT

# Run simulator with performance counters enabled
echo "Running simulation..."
START_TIME=$(date +%s%N)

"${SIMULATOR}" \
    --clusters=${CLUSTERS} \
    --cores=${CORES} \
    --warps=${WARPS} \
    --threads=${THREADS} \
    --perf=3 \
    "${EXTRA_ARGS[@]}" \
    "${KERNEL}" 2>&1 | tee "${TEMP_OUTPUT}"

END_TIME=$(date +%s%N)
WALL_TIME_NS=$((END_TIME - START_TIME))

echo ""
echo "Extracting performance metrics..."

# Parse simulator output for performance counters
# SimX outputs performance data in a specific format

# Extract cycle count
CYCLES=$(grep -oP 'PERF: cycles=\K\d+' "${TEMP_OUTPUT}" || echo "0")

# Extract instruction count
INSTRS=$(grep -oP 'PERF: instrs=\K\d+' "${TEMP_OUTPUT}" || echo "0")

# Extract IPC
IPC=$(grep -oP 'PERF: IPC=\K[\d.]+' "${TEMP_OUTPUT}" || echo "0")

# Extract memory stats
ICACHE_HITS=$(grep -oP 'PERF: icache hits=\K\d+' "${TEMP_OUTPUT}" || echo "0")
ICACHE_MISSES=$(grep -oP 'PERF: icache misses=\K\d+' "${TEMP_OUTPUT}" || echo "0")
DCACHE_HITS=$(grep -oP 'PERF: dcache hits=\K\d+' "${TEMP_OUTPUT}" || echo "0")
DCACHE_MISSES=$(grep -oP 'PERF: dcache misses=\K\d+' "${TEMP_OUTPUT}" || echo "0")

# Extract memory bandwidth (if available)
MEM_READS=$(grep -oP 'PERF: mem reads=\K\d+' "${TEMP_OUTPUT}" || echo "0")
MEM_WRITES=$(grep -oP 'PERF: mem writes=\K\d+' "${TEMP_OUTPUT}" || echo "0")

# Extract stall cycles
STALLS=$(grep -oP 'PERF: stalls=\K\d+' "${TEMP_OUTPUT}" || echo "0")

# Calculate derived metrics
# Assume 1GHz clock for timing estimates (adjust based on actual Vortex config)
CLOCK_FREQ_GHZ=1.0
KERNEL_TIME_NS=$(echo "${CYCLES} / ${CLOCK_FREQ_GHZ}" | bc)

# Estimate H2D and D2H based on memory operations
# These are rough estimates - actual values depend on memory subsystem
BYTES_PER_ACCESS=4
H2D_BYTES=$((MEM_READS * BYTES_PER_ACCESS))
D2H_BYTES=$((MEM_WRITES * BYTES_PER_ACCESS))

# Assume memory bandwidth of 10 GB/s for estimates
MEM_BW_GBPS=10.0
H2D_TIME_NS=$(echo "${H2D_BYTES} / ${MEM_BW_GBPS}" | bc)
D2H_TIME_NS=$(echo "${D2H_BYTES} / ${MEM_BW_GBPS}" | bc)

# Generate JSON profile
echo "Writing profile to ${OUTPUT_PROFILE}..."

cat > "${OUTPUT_PROFILE}" << EOF
{
  "profile_type": "vortex_execution",
  "target": "riscv_vortex",
  "kernel": "$(basename ${KERNEL})",
  "configuration": {
    "clusters": ${CLUSTERS},
    "cores": ${CORES},
    "warps": ${WARPS},
    "threads": ${THREADS},
    "total_threads": $((CLUSTERS * CORES * WARPS * THREADS))
  },
  "timing": {
    "wall_time_ns": ${WALL_TIME_NS},
    "kernel_cycles": ${CYCLES},
    "kernel_latency_ns": ${KERNEL_TIME_NS},
    "h2d_latency_ns": ${H2D_TIME_NS},
    "d2h_latency_ns": ${D2H_TIME_NS},
    "total_latency_ns": $((KERNEL_TIME_NS + H2D_TIME_NS + D2H_TIME_NS))
  },
  "performance": {
    "instructions": ${INSTRS},
    "cycles": ${CYCLES},
    "ipc": ${IPC},
    "stall_cycles": ${STALLS}
  },
  "memory": {
    "icache_hits": ${ICACHE_HITS},
    "icache_misses": ${ICACHE_MISSES},
    "dcache_hits": ${DCACHE_HITS},
    "dcache_misses": ${DCACHE_MISSES},
    "mem_reads": ${MEM_READS},
    "mem_writes": ${MEM_WRITES},
    "h2d_bytes": ${H2D_BYTES},
    "d2h_bytes": ${D2H_BYTES}
  },
  "bandwidth": {
    "h2d_gbps": $(echo "scale=2; ${H2D_BYTES} / ${H2D_TIME_NS:-1}" | bc),
    "d2h_gbps": $(echo "scale=2; ${D2H_BYTES} / ${D2H_TIME_NS:-1}" | bc)
  },
  "prefetch_hints": {
    "optimal_distance_bytes": 65536,
    "strategy": "distance_based"
  }
}
EOF

echo ""
echo "=== Vortex Profiling Summary ==="
echo "  Cycles:       ${CYCLES}"
echo "  Instructions: ${INSTRS}"
echo "  IPC:          ${IPC}"
echo "  Kernel time:  ${KERNEL_TIME_NS} ns"
echo "  H2D time:     ${H2D_TIME_NS} ns"
echo "  D2H time:     ${D2H_TIME_NS} ns"
echo "  Profile:      ${OUTPUT_PROFILE}"
echo "================================"
