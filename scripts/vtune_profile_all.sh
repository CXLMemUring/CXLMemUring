#!/usr/bin/env bash
# VTune profiling script for all benchmarks with and without numactl -m 2
#
# Usage:
#   ./scripts/vtune_profile_all.sh [benchmark] [--numa-only|--baseline-only]
#
# Benchmarks: gapbs, monetdb, ume, spatter, nas, hashjoin, all (default: all)
#
# Prerequisites:
#   spack load intel-oneapi-vtune
#   MonetDB database must be set up for TPC-H (script handles this)
#   UME requires a mesh file: set UME_MESH=/path/to/file.ume

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."

# ─── Configuration ───────────────────────────────────────────────────────────
VTUNE_COLLECT="${VTUNE_COLLECT:-hotspots}"         # hotspots (for function-level), memory-access, uarch-exploration
VTUNE_DURATION="${VTUNE_DURATION:-30}"            # seconds (0 = run to completion)
NUMA_NODE="${NUMA_NODE:-2}"                       # CXL memory node
RESULT_BASE="${REPO_ROOT}/profile_results/vtune"
MONETDB_PORT="${MONETDB_PORT:-54321}"
GAPBS_GRAPH="${GAPBS_GRAPH:-${REPO_ROOT}/bench/gapbs/benchmark/graphs/twitter.sg}"
GAPBS_SCALE="${GAPBS_SCALE:-20}"                  # fallback: generate graph with -g $SCALE
SPATTER_CONFIG="${SPATTER_CONFIG:-${REPO_ROOT}/bench/spatter/standard-suite/app-traces/amg.json}"
UME_MESH="${UME_MESH:-}"                          # must be set externally if running UME
HASHJOIN_WORKDIR="${REPO_ROOT}/bench/partitioned-hash-join/programs/sigmod/workloads"

# Benchmark binaries
GAPBS_DIR="${REPO_ROOT}/bench/gapbs"
NPB_DIR="${REPO_ROOT}/bench/npb/NPB3.4/NPB3.4-OMP/bin"
SPATTER_BIN="${REPO_ROOT}/bench/spatter/build/spatter"
UME_BIN="${REPO_ROOT}/bench/ume/build/src/ume_serial"
HASHJOIN_HARNESS="${REPO_ROOT}/bench/partitioned-hash-join/programs/sigmod/harness"
HASHJOIN_JOINER="${REPO_ROOT}/bench/partitioned-hash-join/programs/sigmod/joiner"
MSERVER5="${REPO_ROOT}/bin/monetdb/x86_64-unknown-linux-gnu/mserver5"
MCLIENT="${REPO_ROOT}/bench/MonetDB/build/clients/mapiclient/mclient"
TPCH_QUERIES="${REPO_ROOT}/bench/MonetDB/sql/benchmarks/tpch/queries"
TPCH_DATA="${REPO_ROOT}/bench/MonetDB/sql/benchmarks/tpch/SF-0.01"
DBFARM="${RESULT_BASE}/dbfarm"

# Parse arguments
BENCHMARK="${1:-all}"
MODE="${2:-}"

# ─── Helpers ─────────────────────────────────────────────────────────────────
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
err()  { echo "[$(date '+%H:%M:%S')] ERROR: $*" >&2; }
die()  { err "$@"; exit 1; }

check_vtune() {
    if ! command -v vtune &>/dev/null; then
        die "vtune not in PATH. Run: spack load intel-oneapi-vtune"
    fi
    log "VTune: $(vtune --version 2>&1 | head -1)"
}

check_numa() {
    if ! command -v numactl &>/dev/null; then
        die "numactl not found"
    fi
    local max_node
    max_node=$(numactl --hardware 2>/dev/null | grep "^available" | awk '{print $2}')
    if (( NUMA_NODE >= max_node )); then
        err "NUMA node ${NUMA_NODE} not available (system has nodes 0-$((max_node-1)))"
        err "Current topology:"
        numactl --hardware 2>/dev/null | head -8
        err ""
        err "If CXL memory is not yet configured as node ${NUMA_NODE}, only baseline runs will execute."
        NUMA_AVAILABLE=false
    else
        NUMA_AVAILABLE=true
        log "NUMA node ${NUMA_NODE}: available"
    fi
}

# Run a benchmark under vtune, with optional numactl prefix
# Usage: run_vtune <result_dir_name> <description> [numactl_args...] -- <command...>
run_vtune() {
    local result_name="$1"; shift
    local desc="$1"; shift

    local numa_args=()
    while [[ $# -gt 0 && "$1" != "--" ]]; do
        numa_args+=("$1"); shift
    done
    [[ "${1:-}" == "--" ]] && shift

    local cmd=("$@")
    local result_dir="${RESULT_BASE}/${result_name}"

    # Skip if result already exists (use --force to override)
    if [[ -d "${result_dir}" && "${FORCE:-}" != "true" ]]; then
        log "  SKIP ${desc} (result exists: ${result_dir})"
        return 0
    fi
    rm -rf "${result_dir}"

    local full_cmd=()
    if [[ ${#numa_args[@]} -gt 0 ]]; then
        full_cmd=(numactl "${numa_args[@]}" "${cmd[@]}")
    else
        full_cmd=("${cmd[@]}")
    fi

    local vtune_args=(
        -collect "${VTUNE_COLLECT}"
        -result-dir "${result_dir}"
    )
    # VTune 2025.8 defaults are fine for both hotspots and memory-access
    if [[ "${VTUNE_DURATION}" != "0" ]]; then
        vtune_args+=(-duration "${VTUNE_DURATION}")
    fi

    log "  RUN  ${desc}"
    log "       vtune ${vtune_args[*]} -- ${full_cmd[*]}"

    if vtune "${vtune_args[@]}" -- "${full_cmd[@]}" 2>&1 | tee "${result_dir}.log"; then
        log "  DONE ${desc} -> ${result_dir}"
        # Generate summary report
        vtune -report summary -result-dir "${result_dir}" -format csv \
            > "${result_dir}_summary.csv" 2>/dev/null || true
        vtune -report hotspots -result-dir "${result_dir}" -format csv \
            > "${result_dir}_hotspots.csv" 2>/dev/null || true
    else
        err "  FAIL ${desc}"
    fi
}

# Wrapper: run baseline (no numactl) and numa variant
run_both() {
    local name="$1"; shift
    local desc="$1"; shift
    local cmd=("$@")

    if [[ "${MODE}" != "--numa-only" ]]; then
        run_vtune "${name}_baseline" "${desc} [baseline]" -- "${cmd[@]}"
    fi
    if [[ "${MODE}" != "--baseline-only" ]]; then
        if [[ "${NUMA_AVAILABLE}" == "true" ]]; then
            run_vtune "${name}_numa${NUMA_NODE}" "${desc} [numactl -m ${NUMA_NODE}]" \
                -m "${NUMA_NODE}" -- "${cmd[@]}"
        else
            log "  SKIP ${desc} [numactl -m ${NUMA_NODE}] (node unavailable)"
        fi
    fi
}

# ─── GAPBS ───────────────────────────────────────────────────────────────────
profile_gapbs() {
    log "=== GAPBS Benchmarks ==="
    local kernels=(bc bfs cc cc_sv pr pr_spmv sssp tc)

    for k in "${kernels[@]}"; do
        local bin="${GAPBS_DIR}/${k}"
        [[ -x "${bin}" ]] || { err "Missing: ${bin}"; continue; }

        local graph_args=()
        if [[ -f "${GAPBS_GRAPH}" ]]; then
            graph_args=(-f "${GAPBS_GRAPH}")
            # tc needs undirected graph
            [[ "${k}" == "tc" ]] && graph_args=(-sf "${GAPBS_GRAPH}")
        else
            graph_args=(-g "${GAPBS_SCALE}")
            [[ "${k}" == "tc" ]] && graph_args=(-u "${GAPBS_SCALE}")
        fi

        local extra_args=(-n 1)
        case "${k}" in
            pr|pr_spmv) extra_args+=(-i 20) ;;
            bc)         extra_args+=(-i 1)  ;;
        esac

        run_both "gapbs/${k}" "GAPBS ${k}" "${bin}" "${graph_args[@]}" "${extra_args[@]}"
    done
}

# ─── MonetDB TPC-H ──────────────────────────────────────────────────────────
monetdb_start() {
    local numa_prefix=("$@")
    mkdir -p "${DBFARM}"

    # Kill any existing mserver5 on our port
    pkill -f "mserver5.*port=${MONETDB_PORT}" 2>/dev/null || true
    sleep 1

    log "  Starting mserver5 on port ${MONETDB_PORT}..."
    "${numa_prefix[@]}" "${MSERVER5}" \
        --dbpath="${DBFARM}/tpch" \
        --set "mapi_port=${MONETDB_PORT}" \
        --set "mapi_listenaddr=localhost" \
        &>/dev/null &
    MSERVER_PID=$!
    sleep 3

    if ! kill -0 "${MSERVER_PID}" 2>/dev/null; then
        die "mserver5 failed to start"
    fi
    log "  mserver5 started (PID=${MSERVER_PID})"
}

monetdb_stop() {
    if [[ -n "${MSERVER_PID:-}" ]]; then
        log "  Stopping mserver5 (PID=${MSERVER_PID})..."
        kill "${MSERVER_PID}" 2>/dev/null || true
        wait "${MSERVER_PID}" 2>/dev/null || true
        MSERVER_PID=""
    fi
    pkill -f "mserver5.*port=${MONETDB_PORT}" 2>/dev/null || true
}

monetdb_setup_tpch() {
    local db_exists=false
    if [[ -d "${DBFARM}/tpch" ]]; then
        # Check if tables are loaded
        local count
        count=$("${MCLIENT}" -p "${MONETDB_PORT}" -d tpch -f csv -s \
            "SELECT count(*) FROM lineitem;" 2>/dev/null | tail -1) || count=0
        if (( count > 0 )); then
            log "  TPC-H database already loaded (lineitem: ${count} rows)"
            db_exists=true
        fi
    fi

    if [[ "${db_exists}" == "false" ]]; then
        log "  Setting up TPC-H database..."
        rm -rf "${DBFARM}/tpch"

        monetdb_stop
        monetdb_start

        # Create schema
        local schema_sql="${TPCH_QUERIES}/schema.sql"
        "${MCLIENT}" -p "${MONETDB_PORT}" < "${schema_sql}"

        # Load data using COPY INTO with absolute paths
        local data_dir
        data_dir="$(cd "${TPCH_DATA}" && pwd)"
        for tbl in region nation supplier customer part partsupp orders lineitem; do
            log "    Loading ${tbl}..."
            "${MCLIENT}" -p "${MONETDB_PORT}" -s \
                "COPY INTO ${tbl} FROM '${data_dir}/${tbl}.tbl' USING DELIMITERS '|', E'|\\\\n';"
        done
        log "  TPC-H data loaded"
    fi
}

profile_monetdb() {
    log "=== MonetDB TPC-H Benchmarks ==="

    [[ -x "${MSERVER5}" ]] || { err "Missing: ${MSERVER5}"; return; }
    [[ -x "${MCLIENT}" ]] || { err "Missing: ${MCLIENT}"; return; }
    [[ -d "${TPCH_QUERIES}" ]] || { err "Missing: ${TPCH_QUERIES}"; return; }

    local queries=(01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22)

    # --- Baseline (no numactl) ---
    if [[ "${MODE}" != "--numa-only" ]]; then
        monetdb_stop
        rm -rf "${DBFARM}/tpch"
        monetdb_start
        monetdb_setup_tpch

        for q in "${queries[@]}"; do
            local qfile="${TPCH_QUERIES}/q${q}.sql"
            [[ -f "${qfile}" ]] || { err "Missing query: ${qfile}"; continue; }

            local result_dir="${RESULT_BASE}/monetdb/q${q}_baseline"
            [[ -d "${result_dir}" && "${FORCE:-}" != "true" ]] && {
                log "  SKIP MonetDB Q${q} baseline (exists)"; continue;
            }
            rm -rf "${result_dir}"

            log "  RUN  MonetDB Q${q} [baseline]"
            vtune -collect "${VTUNE_COLLECT}" \
                -result-dir "${result_dir}" \
                -knob sampling-interval=1 \
                -knob analyze-mem-objects=true \
                -duration "${VTUNE_DURATION}" \
                -target-pid "${MSERVER_PID}" &
            local vtune_pid=$!

            # Run query in a loop for the duration
            local end_time=$((SECONDS + VTUNE_DURATION))
            while (( SECONDS < end_time )); do
                "${MCLIENT}" -p "${MONETDB_PORT}" < "${qfile}" >/dev/null 2>&1 || true
            done

            wait "${vtune_pid}" 2>/dev/null || true
            vtune -report summary -result-dir "${result_dir}" -format csv \
                > "${result_dir}_summary.csv" 2>/dev/null || true
            vtune -report hotspots -result-dir "${result_dir}" -format csv \
                > "${result_dir}_hotspots.csv" 2>/dev/null || true
            log "  DONE MonetDB Q${q} [baseline]"
        done

        monetdb_stop
    fi

    # --- With numactl -m $NUMA_NODE ---
    if [[ "${MODE}" != "--baseline-only" && "${NUMA_AVAILABLE}" == "true" ]]; then
        monetdb_stop
        rm -rf "${DBFARM}/tpch"
        monetdb_start numactl -m "${NUMA_NODE}"
        monetdb_setup_tpch

        for q in "${queries[@]}"; do
            local qfile="${TPCH_QUERIES}/q${q}.sql"
            [[ -f "${qfile}" ]] || continue

            local result_dir="${RESULT_BASE}/monetdb/q${q}_numa${NUMA_NODE}"
            [[ -d "${result_dir}" && "${FORCE:-}" != "true" ]] && {
                log "  SKIP MonetDB Q${q} numa${NUMA_NODE} (exists)"; continue;
            }
            rm -rf "${result_dir}"

            log "  RUN  MonetDB Q${q} [numactl -m ${NUMA_NODE}]"
            vtune -collect "${VTUNE_COLLECT}" \
                -result-dir "${result_dir}" \
                -knob sampling-interval=1 \
                -knob analyze-mem-objects=true \
                -duration "${VTUNE_DURATION}" \
                -target-pid "${MSERVER_PID}" &
            local vtune_pid=$!

            local end_time=$((SECONDS + VTUNE_DURATION))
            while (( SECONDS < end_time )); do
                "${MCLIENT}" -p "${MONETDB_PORT}" < "${qfile}" >/dev/null 2>&1 || true
            done

            wait "${vtune_pid}" 2>/dev/null || true
            vtune -report summary -result-dir "${result_dir}" -format csv \
                > "${result_dir}_summary.csv" 2>/dev/null || true
            vtune -report hotspots -result-dir "${result_dir}" -format csv \
                > "${result_dir}_hotspots.csv" 2>/dev/null || true
            log "  DONE MonetDB Q${q} [numactl -m ${NUMA_NODE}]"
        done

        monetdb_stop
    fi
}

# ─── UME ─────────────────────────────────────────────────────────────────────
profile_ume() {
    log "=== UME Benchmark ==="
    [[ -x "${UME_BIN}" ]] || { err "Missing: ${UME_BIN}"; return; }

    if [[ -z "${UME_MESH}" || ! -f "${UME_MESH}" ]]; then
        err "UME_MESH not set or file not found. Set UME_MESH=/path/to/mesh.ume"
        err "Generate one with: ${REPO_ROOT}/bench/ume/build/src/txt2bin input.umetxt output.ume"
        return
    fi

    run_both "ume/serial" "UME serial" "${UME_BIN}" "${UME_MESH}"
}

# ─── Spatter ─────────────────────────────────────────────────────────────────
profile_spatter() {
    log "=== Spatter Benchmark ==="
    [[ -x "${SPATTER_BIN}" ]] || { err "Missing: ${SPATTER_BIN}"; return; }

    # AMG trace pattern
    if [[ -f "${SPATTER_CONFIG}" ]]; then
        run_both "spatter/amg" "Spatter AMG" \
            "${SPATTER_BIN}" -f "${SPATTER_CONFIG}"
    fi

    # Uniform stride patterns (varying sizes for memory pressure)
    for stride in 1 8 64 512; do
        run_both "spatter/uniform_s${stride}" "Spatter uniform stride=${stride}" \
            "${SPATTER_BIN}" -pUNIFORM:${stride}:1 -l$((2**24))
    done

    # Gather and scatter kernels
    run_both "spatter/gather" "Spatter gather" \
        "${SPATTER_BIN}" -pUNIFORM:8:1 -l$((2**24)) -kGather
    run_both "spatter/scatter" "Spatter scatter" \
        "${SPATTER_BIN}" -pUNIFORM:8:1 -l$((2**24)) -kScatter

    # App trace patterns
    for trace in lulesh pennant nekbone; do
        local tfile="${REPO_ROOT}/bench/spatter/standard-suite/app-traces/${trace}.json"
        if [[ -f "${tfile}" ]]; then
            run_both "spatter/${trace}" "Spatter ${trace}" \
                "${SPATTER_BIN}" -f "${tfile}"
        fi
    done
}

# ─── NAS Parallel Benchmarks ────────────────────────────────────────────────
profile_nas() {
    log "=== NAS Parallel Benchmarks ==="

    # Run all available NPB binaries
    local npb_bins=()
    for f in "${NPB_DIR}"/*.x; do
        [[ -x "${f}" ]] && npb_bins+=("${f}")
    done

    if [[ ${#npb_bins[@]} -eq 0 ]]; then
        err "No NPB binaries found in ${NPB_DIR}"
        return
    fi

    for bin in "${npb_bins[@]}"; do
        local name
        name=$(basename "${bin}" .x)  # e.g., cg.E, ep.E, ft.E
        run_both "nas/${name}" "NPB ${name}" "${bin}"
    done
}

# ─── Hash Join ───────────────────────────────────────────────────────────────
profile_hashjoin() {
    log "=== Partitioned Hash Join ==="
    [[ -x "${HASHJOIN_HARNESS}" ]] || { err "Missing: ${HASHJOIN_HARNESS}"; return; }
    [[ -x "${HASHJOIN_JOINER}" ]] || { err "Missing: ${HASHJOIN_JOINER}"; return; }

    # Small workload via harness
    if [[ -f "${HASHJOIN_WORKDIR}/small.init" ]]; then
        run_both "hashjoin/small" "HashJoin small" \
            "${HASHJOIN_HARNESS}" \
            "${HASHJOIN_WORKDIR}/small.init" \
            "${HASHJOIN_WORKDIR}/small.work" \
            "${HASHJOIN_WORKDIR}/small.result" \
            "${HASHJOIN_JOINER}"
    fi

    # Public workloads (larger)
    if [[ -d "${HASHJOIN_WORKDIR}/public" ]]; then
        for init_file in "${HASHJOIN_WORKDIR}"/public/*.init; do
            [[ -f "${init_file}" ]] || continue
            local wname
            wname=$(basename "${init_file}" .init)
            local work_file="${HASHJOIN_WORKDIR}/public/${wname}.work"
            local result_file="${HASHJOIN_WORKDIR}/public/${wname}.result"
            [[ -f "${work_file}" && -f "${result_file}" ]] || continue

            run_both "hashjoin/${wname}" "HashJoin ${wname}" \
                "${HASHJOIN_HARNESS}" \
                "${init_file}" "${work_file}" "${result_file}" \
                "${HASHJOIN_JOINER}"
        done
    fi
}

# ─── Main ────────────────────────────────────────────────────────────────────
main() {
    log "=================================================="
    log " VTune Profiling: ${BENCHMARK}"
    log " Collection: ${VTUNE_COLLECT}"
    log " Duration: ${VTUNE_DURATION}s per run"
    log " NUMA node: ${NUMA_NODE}"
    log " Results: ${RESULT_BASE}"
    log "=================================================="

    check_vtune
    check_numa
    mkdir -p "${RESULT_BASE}"

    case "${BENCHMARK}" in
        gapbs)    profile_gapbs ;;
        monetdb)  profile_monetdb ;;
        ume)      profile_ume ;;
        spatter)  profile_spatter ;;
        nas)      profile_nas ;;
        hashjoin) profile_hashjoin ;;
        all)
            profile_gapbs
            profile_monetdb
            profile_ume
            profile_spatter
            profile_nas
            profile_hashjoin
            ;;
        *) die "Unknown benchmark: ${BENCHMARK}. Use: gapbs, monetdb, ume, spatter, nas, hashjoin, all" ;;
    esac

    log ""
    log "=================================================="
    log " Profiling complete. Results in: ${RESULT_BASE}"
    log "=================================================="
    log ""
    log "View results with:"
    log "  vtune-gui ${RESULT_BASE}/<benchmark>_baseline"
    log ""
    log "Generate reports:"
    log "  vtune -report summary -result-dir ${RESULT_BASE}/<benchmark>_baseline"
    log "  vtune -report hotspots -result-dir ${RESULT_BASE}/<benchmark>_baseline -format csv"
    log ""

    # Print summary table
    log "Collected results:"
    find "${RESULT_BASE}" -maxdepth 3 -name "*.vtune" -o -name "config" | \
        sed "s|${RESULT_BASE}/||" | sort | while read -r f; do
        echo "  ${f}"
    done
}

trap 'monetdb_stop 2>/dev/null; exit' EXIT INT TERM
main
