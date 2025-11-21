#!/bin/bash
# Run profiling sweep across multiple data sizes
# Generates aggregated profile data for compiler prefetch tuning

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
E2E_BIN="${REPO_ROOT}/runtime/e2e_profiled"
KERNEL="${REPO_ROOT}/test/kernels/simple_add.vxbin"
OUTPUT_DIR="${REPO_ROOT}/profile_results"

export LD_LIBRARY_PATH="${REPO_ROOT}/vortex/runtime:${LD_LIBRARY_PATH}"

# Data sizes to profile
SIZES=(64 128 256 512 1024)

mkdir -p "${OUTPUT_DIR}"

echo "=== Vortex Offload Profiling Sweep ==="
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Run each size
for size in "${SIZES[@]}"; do
    echo "Running with ${size} elements..."
    output_file="${OUTPUT_DIR}/profile_${size}.json"
    timeout 180 "${E2E_BIN}" "${size}" "${KERNEL}" "${output_file}" 2>&1 | grep -E "Latencies|H2D|D2H|Kernel|Bandwidth|Prefetch|throughput"
    echo ""
done

# Aggregate results into single JSON for compiler
echo "Generating aggregated profile..."
cat > "${OUTPUT_DIR}/compiler_profile.json" << 'HEADER'
{
  "profile_type": "offload_timing",
  "target": "vortex_rv32imf",
  "runs": [
HEADER

first=true
for size in "${SIZES[@]}"; do
    profile_file="${OUTPUT_DIR}/profile_${size}.json"
    if [ -f "${profile_file}" ]; then
        if [ "$first" = true ]; then
            first=false
        else
            echo "," >> "${OUTPUT_DIR}/compiler_profile.json"
        fi
        # Extract and add size info
        echo "    {\"elements\": ${size}, \"data\": $(cat ${profile_file})}" >> "${OUTPUT_DIR}/compiler_profile.json"
    fi
done

cat >> "${OUTPUT_DIR}/compiler_profile.json" << 'FOOTER'
  ],
  "optimization_hints": {
    "prefetch_strategy": "distance_based",
    "notes": "Use optimal_distance_bytes scaled by transfer size ratio"
  }
}
FOOTER

echo ""
echo "=== Profile Sweep Complete ==="
echo "Individual profiles: ${OUTPUT_DIR}/profile_*.json"
echo "Compiler profile:    ${OUTPUT_DIR}/compiler_profile.json"
