#!/usr/bin/env bash
# VTune uarch-exploration profiling for all benchmarks
# Results go to profile_results/vtune/<bench>/<name>_uarch_{baseline,numa0,numa2}
#
# Usage:
#   spack load intel-oneapi-vtune
#   ./scripts/vtune_uarch_exploration.sh [--bench gapbs|nas|spatter|hashjoin|monetdb|all]
#                                        [--duration N] [--force]
#
# Profiles each benchmark three ways:
#   1. baseline (no NUMA binding)
#   2. numactl -m 0 (local DRAM)
#   3. numactl -m 2 (CXL memory)
#
# Workarounds for kernel 6.18.0-rc5:
#   - Uses sampling-interval=10 to reduce NMI frequency (scheduling-while-atomic)
#   - Skips ft.E.x (needs 273GB+ RSS, causes OOM)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."

RESULT_BASE="${REPO_ROOT}/profile_results/vtune"
VTUNE_DURATION="${VTUNE_DURATION:-30}"
FORCE="${FORCE:-false}"

# Reduce NMI frequency to avoid "scheduling while atomic" on 6.18.0-rc5
VTUNE_KNOBS=(-knob sampling-interval=10)

GAPBS_DIR="${REPO_ROOT}/bench/gapbs"
GAPBS_GRAPH="${GAPBS_DIR}/benchmark/graphs/twitter.sg"
NPB_DIR="${REPO_ROOT}/bench/npb/NPB3.4/NPB3.4-OMP/bin"
SPATTER_BIN="${REPO_ROOT}/bench/spatter/build/spatter"
HASHJOIN_HARNESS="${REPO_ROOT}/bench/partitioned-hash-join/programs/sigmod/harness"
HASHJOIN_JOINER="${REPO_ROOT}/bench/partitioned-hash-join/programs/sigmod/joiner"
HASHJOIN_WORKDIR="${REPO_ROOT}/bench/partitioned-hash-join/programs/sigmod/workloads"
MSERVER5="${REPO_ROOT}/bin/monetdb/x86_64-unknown-linux-gnu/mserver5"
MCLIENT="${REPO_ROOT}/bench/MonetDB/build/clients/mapiclient/mclient"
TPCH_QUERIES="${REPO_ROOT}/bench/MonetDB/sql/benchmarks/tpch/queries"
TPCH_DATA="${REPO_ROOT}/bench/MonetDB/sql/benchmarks/tpch/SF-0.01"
TPCH_DATA_SF1="${REPO_ROOT}/bench/MonetDB/sql/benchmarks/tpch/SF-1"
MONETDB_PORT="${MONETDB_PORT:-54323}"

# NAS benchmarks that exceed 16GB (CXL node 2 capacity)
# ft.E=262GB, lu.E=136GB, mg.E=212GB — skip entirely (also OOM risk on baseline)
NAS_SKIP_ALL="ft.E"
# These fit local DRAM but not CXL node 2 (16GB) — run baseline+numa0 only
NAS_SKIP_NUMA2="lu.E mg.E"

BENCHMARK="all"
NUMA_NODES=(0 2)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bench)    BENCHMARK="$2"; shift 2 ;;
        --duration) VTUNE_DURATION="$2"; shift 2 ;;
        --force)    FORCE=true; shift ;;
        *)          echo "Unknown: $1"; exit 1 ;;
    esac
done

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
err()  { echo "[$(date '+%H:%M:%S')] ERROR: $*" >&2; }
die()  { err "$@"; exit 1; }
MSERVER_PID=""

cleanup_monetdb() {
    if [[ -n "${MSERVER_PID}" ]]; then
        kill "${MSERVER_PID}" 2>/dev/null || true
        wait "${MSERVER_PID}" 2>/dev/null || true
        MSERVER_PID=""
    fi
    pkill -f "mserver5.*port=${MONETDB_PORT}" 2>/dev/null || true
}
trap cleanup_monetdb EXIT INT TERM

# Run VTune uarch-exploration with reduced NMI frequency
# Args: <result_dir> <description> [numa_args...] -- <command...>
vtune_uarch() {
    local result_dir="$1"; shift
    local desc="$1"; shift

    local numa_args=()
    while [[ $# -gt 0 && "$1" != "--" ]]; do
        numa_args+=("$1"); shift
    done
    [[ "${1:-}" == "--" ]] && shift
    local cmd=("$@")

    if [[ -d "${result_dir}" && "${FORCE}" != "true" ]]; then
        log "  SKIP ${desc} (exists)"
        return 0
    fi
    rm -rf "${result_dir}"
    mkdir -p "$(dirname "${result_dir}")"

    local full_cmd=()
    if [[ ${#numa_args[@]} -gt 0 ]]; then
        full_cmd=(numactl "${numa_args[@]}" "${cmd[@]}")
    else
        full_cmd=("${cmd[@]}")
    fi

    log "  RUN  ${desc}"
    vtune -collect uarch-exploration \
        "${VTUNE_KNOBS[@]}" \
        -result-dir "${result_dir}" \
        -duration "${VTUNE_DURATION}" \
        -- "${full_cmd[@]}" > "${result_dir}.log" 2>&1 || true

    if [[ -d "${result_dir}" ]]; then
        log "  DONE ${desc}"
        vtune -report summary -result-dir "${result_dir}" -format csv \
            > "${result_dir}_summary.csv" 2>/dev/null || true
        vtune -report hotspots -result-dir "${result_dir}" -format csv \
            -group-by function \
            > "${result_dir}_hotspots.csv" 2>/dev/null || true
    else
        err "  FAIL ${desc} (see ${result_dir}.log)"
    fi
}

# Run VTune uarch-exploration attached to a PID (for MonetDB)
vtune_uarch_pid() {
    local result_dir="$1"; shift
    local desc="$1"; shift
    local pid="$1"; shift

    if [[ -d "${result_dir}" && "${FORCE}" != "true" ]]; then
        log "  SKIP ${desc} (exists)"
        return 0
    fi
    rm -rf "${result_dir}"
    mkdir -p "$(dirname "${result_dir}")"

    log "  RUN  ${desc}"
    vtune -collect uarch-exploration \
        "${VTUNE_KNOBS[@]}" \
        -result-dir "${result_dir}" \
        -duration "${VTUNE_DURATION}" \
        -target-pid "${pid}" \
        &> "${result_dir}.log" &
    echo $!
}

# Profile a benchmark: baseline + numa0 + numa2
run_bench() {
    local bench="$1"; shift
    local name="$1"; shift
    local cmd=("$@")

    # Baseline (no NUMA binding)
    vtune_uarch "${RESULT_BASE}/${bench}/${name}_uarch_baseline" \
        "${bench}/${name} baseline" -- "${cmd[@]}"

    # NUMA-bound runs
    for node in "${NUMA_NODES[@]}"; do
        vtune_uarch "${RESULT_BASE}/${bench}/${name}_uarch_numa${node}" \
            "${bench}/${name} numa${node}" -m "${node}" -- "${cmd[@]}"
    done
}

profile_gapbs() {
    log "============ GAPBS uarch-exploration ============"
    local kernels=(bc bfs cc cc_sv pr pr_spmv sssp tc)
    for k in "${kernels[@]}"; do
        local bin="${GAPBS_DIR}/${k}"
        [[ -x "${bin}" ]] || { err "Missing: ${bin}"; continue; }

        local graph_args=()
        if [[ -f "${GAPBS_GRAPH}" ]]; then
            graph_args=(-f "${GAPBS_GRAPH}")
            [[ "${k}" == "tc" ]] && graph_args=(-sf "${GAPBS_GRAPH}")
        else
            graph_args=(-g 20)
            [[ "${k}" == "tc" ]] && graph_args=(-u 20)
        fi

        local extra_args=(-n 1)
        case "${k}" in
            pr|pr_spmv) extra_args+=(-i 20) ;;
            bc)         extra_args+=(-i 1)  ;;
        esac

        run_bench "gapbs" "${k}" "${bin}" "${graph_args[@]}" "${extra_args[@]}"
    done
}

profile_nas() {
    log "============ NAS Parallel Benchmarks uarch-exploration ============"
    for bin in "${NPB_DIR}"/*.x; do
        [[ -x "${bin}" ]] || continue
        local name
        name=$(basename "${bin}" .x)

        # Skip benchmarks that OOM even on local DRAM
        if [[ "${NAS_SKIP_ALL}" == *"${name}"* ]]; then
            log "  SKIP ${name} (exceeds local DRAM — 262GB RSS)"
            continue
        fi

        # Benchmarks too large for CXL node 2 (16GB): run baseline + numa0 only
        if [[ "${NAS_SKIP_NUMA2}" == *"${name}"* ]]; then
            log "  ${name}: baseline + numa0 only (${name} exceeds 16GB CXL)"
            vtune_uarch "${RESULT_BASE}/nas/${name}_uarch_baseline" \
                "nas/${name} baseline" -- "${bin}"
            vtune_uarch "${RESULT_BASE}/nas/${name}_uarch_numa0" \
                "nas/${name} numa0" -m 0 -- "${bin}"
            continue
        fi

        run_bench "nas" "${name}" "${bin}"
    done
}

profile_spatter() {
    log "============ Spatter uarch-exploration ============"
    [[ -x "${SPATTER_BIN}" ]] || { err "Missing: ${SPATTER_BIN}"; return; }

    local amg="${REPO_ROOT}/bench/spatter/standard-suite/app-traces/amg.json"
    [[ -f "${amg}" ]] && run_bench "spatter" "amg" "${SPATTER_BIN}" -f "${amg}"

    for stride in 1 8 64 512; do
        run_bench "spatter" "uniform_s${stride}" \
            "${SPATTER_BIN}" -pUNIFORM:${stride}:1 -l$((2**24))
    done

    for trace in lulesh pennant nekbone; do
        local tfile="${REPO_ROOT}/bench/spatter/standard-suite/app-traces/${trace}.json"
        [[ -f "${tfile}" ]] && run_bench "spatter" "${trace}" "${SPATTER_BIN}" -f "${tfile}"
    done
}

profile_hashjoin() {
    log "============ Hash Join uarch-exploration ============"
    [[ -x "${HASHJOIN_HARNESS}" ]] || { err "Missing: ${HASHJOIN_HARNESS}"; return; }
    [[ -x "${HASHJOIN_JOINER}" ]] || { err "Missing: ${HASHJOIN_JOINER}"; return; }

    if [[ -f "${HASHJOIN_WORKDIR}/small.init" ]]; then
        pushd "${HASHJOIN_WORKDIR}" > /dev/null
        run_bench "hashjoin" "small" \
            "${HASHJOIN_HARNESS}" \
            small.init small.work small.result \
            "${HASHJOIN_JOINER}"
        popd > /dev/null
    fi
}

profile_monetdb() {
    log "============ MonetDB TPC-H uarch-exploration ============"
    [[ -x "${MSERVER5}" ]] || { err "Missing: ${MSERVER5}"; return; }
    [[ -x "${MCLIENT}" ]] || { err "Missing: ${MCLIENT}"; return; }

    local queries=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22)

    # --- Baseline (no NUMA) ---
    log "  --- MonetDB baseline ---"
    local dbfarm="${RESULT_BASE}/monetdb/dbfarm"
    mkdir -p "${dbfarm}"
    cleanup_monetdb
    sleep 1

    "${MSERVER5}" \
        --dbpath="${dbfarm}/tpch" \
        --set "mapi_port=${MONETDB_PORT}" \
        --set "mapi_listenaddr=localhost" \
        --set "gdk_nr_threads=0" \
        &>/dev/null &
    MSERVER_PID=$!
    sleep 3
    kill -0 "${MSERVER_PID}" 2>/dev/null || { err "mserver5 failed"; return; }
    log "  mserver5 PID=${MSERVER_PID}"

    # Load TPC-H if needed
    local count
    count=$("${MCLIENT}" -p "${MONETDB_PORT}" -f csv -s \
        "SELECT count(*) FROM lineitem;" 2>/dev/null | tail -1) || count=0
    if (( count == 0 )); then
        log "  Loading TPC-H SF-0.01..."
        "${MCLIENT}" -p "${MONETDB_PORT}" < "${TPCH_QUERIES}/schema.sql"
        local data_dir
        data_dir="$(cd "${TPCH_DATA}" && pwd)"
        for tbl in region nation supplier customer part partsupp orders lineitem; do
            "${MCLIENT}" -p "${MONETDB_PORT}" -s \
                "COPY INTO ${tbl} FROM '${data_dir}/${tbl}.tbl' USING DELIMITERS '|', E'\\n', '\"';"
        done
        log "  TPC-H loaded"
    fi

    for q in "${queries[@]}"; do
        local qfile="${TPCH_QUERIES}/q$(printf '%02d' "${q}").sql"
        local result_dir="${RESULT_BASE}/monetdb/q$(printf '%02d' "${q}")_uarch_baseline"
        [[ -f "${qfile}" ]] || continue

        if [[ -d "${result_dir}" && "${FORCE}" != "true" ]]; then
            log "    SKIP Q${q} uarch baseline"
            continue
        fi
        rm -rf "${result_dir}"

        "${MCLIENT}" -p "${MONETDB_PORT}" < "${qfile}" >/dev/null 2>&1 || true

        log "    Q${q} uarch baseline"
        vtune -collect uarch-exploration \
            "${VTUNE_KNOBS[@]}" \
            -result-dir "${result_dir}" \
            -duration "${VTUNE_DURATION}" \
            -target-pid "${MSERVER_PID}" \
            &> "${result_dir}.log" &
        local vtune_pid=$!
        sleep 1

        local end_time=$((SECONDS + VTUNE_DURATION - 2))
        while (( SECONDS < end_time )); do
            "${MCLIENT}" -p "${MONETDB_PORT}" < "${qfile}" >/dev/null 2>&1 || true
        done
        wait "${vtune_pid}" 2>/dev/null || true

        vtune -report summary -result-dir "${result_dir}" -format csv \
            > "${result_dir}_summary.csv" 2>/dev/null || true
        vtune -report hotspots -result-dir "${result_dir}" -format csv \
            -group-by function > "${result_dir}_hotspots.csv" 2>/dev/null || true
        log "    Q${q} done"
    done
    cleanup_monetdb

    # --- NUMA-bound runs ---
    for node in "${NUMA_NODES[@]}"; do
        log "  --- MonetDB numa${node} ---"
        rm -rf "${dbfarm}/tpch"
        cleanup_monetdb
        sleep 1

        numactl -m "${node}" "${MSERVER5}" \
            --dbpath="${dbfarm}/tpch" \
            --set "mapi_port=${MONETDB_PORT}" \
            --set "mapi_listenaddr=localhost" \
            --set "gdk_nr_threads=0" \
            &>/dev/null &
        MSERVER_PID=$!
        sleep 3
        if ! kill -0 "${MSERVER_PID}" 2>/dev/null; then
            err "mserver5 failed on numa${node}"
            continue
        fi
        log "  mserver5 PID=${MSERVER_PID} (numa${node})"

        # Reload TPC-H
        "${MCLIENT}" -p "${MONETDB_PORT}" < "${TPCH_QUERIES}/schema.sql" 2>/dev/null || true
        local data_dir
        data_dir="$(cd "${TPCH_DATA}" && pwd)"
        for tbl in region nation supplier customer part partsupp orders lineitem; do
            "${MCLIENT}" -p "${MONETDB_PORT}" -s \
                "COPY INTO ${tbl} FROM '${data_dir}/${tbl}.tbl' USING DELIMITERS '|', E'\\n', '\"';" 2>/dev/null || true
        done

        for q in "${queries[@]}"; do
            local qfile="${TPCH_QUERIES}/q$(printf '%02d' "${q}").sql"
            local result_dir="${RESULT_BASE}/monetdb/q$(printf '%02d' "${q}")_uarch_numa${node}"
            [[ -f "${qfile}" ]] || continue

            if [[ -d "${result_dir}" && "${FORCE}" != "true" ]]; then
                log "    SKIP Q${q} uarch numa${node}"
                continue
            fi
            rm -rf "${result_dir}"

            "${MCLIENT}" -p "${MONETDB_PORT}" < "${qfile}" >/dev/null 2>&1 || true

            log "    Q${q} uarch numa${node}"
            vtune -collect uarch-exploration \
                "${VTUNE_KNOBS[@]}" \
                -result-dir "${result_dir}" \
                -duration "${VTUNE_DURATION}" \
                -target-pid "${MSERVER_PID}" \
                &> "${result_dir}.log" &
            local vtune_pid=$!
            sleep 1

            local end_time=$((SECONDS + VTUNE_DURATION - 2))
            while (( SECONDS < end_time )); do
                "${MCLIENT}" -p "${MONETDB_PORT}" < "${qfile}" >/dev/null 2>&1 || true
            done
            wait "${vtune_pid}" 2>/dev/null || true

            vtune -report summary -result-dir "${result_dir}" -format csv \
                > "${result_dir}_summary.csv" 2>/dev/null || true
            vtune -report hotspots -result-dir "${result_dir}" -format csv \
                -group-by function > "${result_dir}_hotspots.csv" 2>/dev/null || true
            log "    Q${q} done"
        done
        cleanup_monetdb
    done
}

profile_monetdb_sf1() {
    log "============ MonetDB TPC-H SF-1 uarch-exploration ============"
    [[ -x "${MSERVER5}" ]] || { err "Missing: ${MSERVER5}"; return; }
    [[ -x "${MCLIENT}" ]] || { err "Missing: ${MCLIENT}"; return; }
    [[ -d "${TPCH_DATA_SF1}" ]] || { err "Missing SF-1 data: ${TPCH_DATA_SF1}"; return; }

    local queries=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22)
    local sf1_port=$((MONETDB_PORT + 1))

    # Helper: start mserver5 for SF-1 with optional numactl prefix
    monetdb_sf1_start() {
        local tag="$1"; shift
        local prefix=("$@")
        local dbfarm="${RESULT_BASE}/monetdb_sf1/dbfarm"
        mkdir -p "${dbfarm}"
        rm -rf "${dbfarm}/tpch_sf1"
        cleanup_monetdb
        sleep 1

        "${prefix[@]}" "${MSERVER5}" \
            --dbpath="${dbfarm}/tpch_sf1" \
            --set "mapi_port=${sf1_port}" \
            --set "mapi_listenaddr=localhost" \
            --set "gdk_nr_threads=0" \
            &>/dev/null &
        MSERVER_PID=$!
        sleep 3
        if ! kill -0 "${MSERVER_PID}" 2>/dev/null; then
            err "mserver5 SF-1 failed (${tag})"
            return 1
        fi
        log "  mserver5 SF-1 PID=${MSERVER_PID} (${tag})"

        # Load SF-1 data
        log "  Loading TPC-H SF-1..."
        "${MCLIENT}" -p "${sf1_port}" < "${TPCH_QUERIES}/../c.sql-primary" 2>/dev/null || \
        "${MCLIENT}" -p "${sf1_port}" < "${TPCH_QUERIES}/schema.sql" 2>/dev/null || true
        local data_dir
        data_dir="$(cd "${TPCH_DATA_SF1}" && pwd)"
        for tbl in region nation supplier customer part partsupp orders lineitem; do
            "${MCLIENT}" -p "${sf1_port}" -s \
                "COPY INTO ${tbl} FROM '${data_dir}/${tbl}.tbl' USING DELIMITERS '|', E'\\n', '\"';" 2>/dev/null || true
        done
        local count
        count=$("${MCLIENT}" -p "${sf1_port}" -f csv -s \
            "SELECT count(*) FROM lineitem;" 2>/dev/null | tail -1) || count=0
        log "  SF-1 loaded (lineitem=${count})"
        return 0
    }

    # Helper: profile all queries against running mserver5
    monetdb_sf1_profile_queries() {
        local tag="$1"
        for q in "${queries[@]}"; do
            local qfile="${TPCH_QUERIES}/q$(printf '%02d' "${q}").sql"
            local result_dir="${RESULT_BASE}/monetdb_sf1/q$(printf '%02d' "${q}")_uarch_${tag}"
            [[ -f "${qfile}" ]] || continue

            if [[ -d "${result_dir}" && "${FORCE}" != "true" ]]; then
                log "    SKIP Q${q} uarch ${tag}"
                continue
            fi
            rm -rf "${result_dir}"

            # Warm up
            "${MCLIENT}" -p "${sf1_port}" < "${qfile}" >/dev/null 2>&1 || true

            log "    Q${q} uarch ${tag}"
            vtune -collect uarch-exploration \
                "${VTUNE_KNOBS[@]}" \
                -result-dir "${result_dir}" \
                -duration "${VTUNE_DURATION}" \
                -target-pid "${MSERVER_PID}" \
                &> "${result_dir}.log" &
            local vtune_pid=$!
            sleep 1

            local end_time=$((SECONDS + VTUNE_DURATION - 2))
            while (( SECONDS < end_time )); do
                "${MCLIENT}" -p "${sf1_port}" < "${qfile}" >/dev/null 2>&1 || true
            done
            wait "${vtune_pid}" 2>/dev/null || true

            vtune -report summary -result-dir "${result_dir}" -format csv \
                > "${result_dir}_summary.csv" 2>/dev/null || true
            vtune -report hotspots -result-dir "${result_dir}" -format csv \
                -group-by function > "${result_dir}_hotspots.csv" 2>/dev/null || true
            log "    Q${q} done"
        done
    }

    # --- Baseline ---
    log "  --- MonetDB SF-1 baseline ---"
    if monetdb_sf1_start "baseline"; then
        monetdb_sf1_profile_queries "baseline"
        cleanup_monetdb
    fi

    # --- NUMA-bound runs ---
    for node in "${NUMA_NODES[@]}"; do
        log "  --- MonetDB SF-1 numa${node} ---"
        if monetdb_sf1_start "numa${node}" numactl -m "${node}"; then
            monetdb_sf1_profile_queries "numa${node}"
            cleanup_monetdb
        fi
    done
}

main() {
    command -v vtune &>/dev/null || die "vtune not in PATH. Run: spack load intel-oneapi-vtune"

    log "============================================================"
    log " VTune uarch-exploration Profiling"
    log " Benchmark: ${BENCHMARK}"
    log " Duration:  ${VTUNE_DURATION}s"
    log " NUMA:      baseline + ${NUMA_NODES[*]}"
    log " Knobs:     ${VTUNE_KNOBS[*]}"
    log " NAS skip:  all=${NAS_SKIP_ALL} numa2=${NAS_SKIP_NUMA2}"
    log " Results:   ${RESULT_BASE}"
    log "============================================================"

    mkdir -p "${RESULT_BASE}"

    case "${BENCHMARK}" in
        gapbs)       profile_gapbs ;;
        nas)         profile_nas ;;
        spatter)     profile_spatter ;;
        hashjoin)    profile_hashjoin ;;
        monetdb)     profile_monetdb ;;
        monetdb_sf1) profile_monetdb_sf1 ;;
        all)
            profile_gapbs
            profile_nas
            profile_spatter
            profile_hashjoin
            profile_monetdb
            profile_monetdb_sf1
            ;;
        *) die "Unknown: ${BENCHMARK}" ;;
    esac

    log ""
    log "============================================================"
    log " uarch-exploration profiling complete"
    log " Results: ${RESULT_BASE}/"
    log "============================================================"

    local n_results
    n_results=$(find "${RESULT_BASE}" -name "*uarch*_summary.csv" 2>/dev/null | wc -l)
    log "Total uarch result sets: ${n_results}"
}

main
