#!/usr/bin/env bash
# Collect paired VTune profiles for GAPBS on local DDR and CXL memory nodes.
#
# Example:
#   OMP_NUM_THREADS=64 bash scripts/vtune_gapbs_cxl_compare.sh \
#     --local-node 0 --cxl-node 2 --graph-args "-g 20" \
#     --kernels "bfs pr bc cc_sv pr_spmv"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
GAPBS_DIR="${REPO_ROOT}/bench/gapbs"

RESULT_DIR="${REPO_ROOT}/profile_results/vtune_gapbs_cxl"
LOCAL_NODE="${LOCAL_NODE:-0}"
CXL_NODE="${CXL_NODE:-2}"
CPU_NODE="${CPU_NODE:-0}"
TRIALS="${TRIALS:-1}"
ANALYSIS="${ANALYSIS:-uarch-exploration}"
GRAPH_ARGS="${GRAPH_ARGS:--g 20}"
KERNELS="${KERNELS:-bfs pr bc cc_sv pr_spmv}"
FORCE="${FORCE:-false}"

usage() {
    cat <<'EOF'
Usage: vtune_gapbs_cxl_compare.sh [options]

Options:
  --result-dir DIR      Output directory [profile_results/vtune_gapbs_cxl]
  --local-node N        DDR/local memory NUMA node [0]
  --cxl-node N          CXL memory NUMA node [2]
  --cpu-node N          CPU NUMA node for both runs [0]
  --trials N            Trials per kernel/config [1]
  --analysis NAME       VTune analysis [uarch-exploration]
  --graph-args "ARGS"  GAPBS graph args, e.g. "-g 20" or "-f graph.sg"
  --kernels "LIST"     Space-separated kernels [bfs pr bc cc_sv pr_spmv]
  --force              Overwrite existing result directories
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --result-dir) RESULT_DIR="$2"; shift 2 ;;
        --local-node) LOCAL_NODE="$2"; shift 2 ;;
        --cxl-node) CXL_NODE="$2"; shift 2 ;;
        --cpu-node) CPU_NODE="$2"; shift 2 ;;
        --trials) TRIALS="$2"; shift 2 ;;
        --analysis) ANALYSIS="$2"; shift 2 ;;
        --graph-args) GRAPH_ARGS="$2"; shift 2 ;;
        --kernels) KERNELS="$2"; shift 2 ;;
        --force) FORCE=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

command -v vtune >/dev/null 2>&1 || {
    echo "error: vtune not found in PATH. Load Intel VTune first." >&2
    exit 1
}
command -v numactl >/dev/null 2>&1 || {
    echo "error: numactl not found in PATH." >&2
    exit 1
}

mkdir -p "${RESULT_DIR}"

run_one() {
    local kernel="$1"
    local label="$2"
    local mem_node="$3"
    local trial="$4"
    local bin="${GAPBS_DIR}/${kernel}"
    local out="${RESULT_DIR}/${kernel}_${label}_t${trial}_${ANALYSIS}"

    [[ -x "${bin}" ]] || {
        echo "skip: missing executable ${bin}" >&2
        return 0
    }

    if [[ -d "${out}" && "${FORCE}" != "true" ]]; then
        echo "skip: ${out} exists"
        return 0
    fi
    rm -rf "${out}" "${out}.log" "${out}_summary.csv" "${out}_hotspots.csv"

    read -r -a graph_args <<< "${GRAPH_ARGS}"
    local extra_args=(-n 1)
    case "${kernel}" in
        pr|pr_spmv) extra_args+=(-i 20) ;;
        bc) extra_args+=(-i 1) ;;
    esac

    echo "collect: ${kernel} ${label} trial=${trial} mem_node=${mem_node}"
    vtune -collect "${ANALYSIS}" -result-dir "${out}" -- \
        numactl --cpunodebind="${CPU_NODE}" --membind="${mem_node}" \
        "${bin}" "${graph_args[@]}" "${extra_args[@]}" \
        > "${out}.log" 2>&1 || true

    vtune -report summary -result-dir "${out}" -format csv \
        > "${out}_summary.csv" 2>/dev/null || true
    vtune -report hotspots -result-dir "${out}" -format csv -group-by function \
        > "${out}_hotspots.csv" 2>/dev/null || true
}

for kernel in ${KERNELS}; do
    for trial in $(seq 1 "${TRIALS}"); do
        run_one "${kernel}" "noncxl" "${LOCAL_NODE}" "${trial}"
        run_one "${kernel}" "cxl" "${CXL_NODE}" "${trial}"
    done
done

python3 "${SCRIPT_DIR}/summarize_gapbs_vtune_cxl.py" \
    --result-dir "${RESULT_DIR}" \
    --csv-out "${RESULT_DIR}/gapbs_vtune_cxl_summary.csv" \
    --tex-out "${REPO_ROOT}/6472666535e6f359942ddac6/gapbs-vtune-cxl-table.tex"
