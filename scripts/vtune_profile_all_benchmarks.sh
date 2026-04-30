#!/usr/bin/env bash
# Comprehensive VTune Profiling for All Benchmarks
# With and without numactl -m (CXL memory node), with CXLMemUring Type2 GPU
#
# This script profiles:
#   1. GAPBS (bc, bfs, cc, cc_sv, pr, pr_spmv, sssp, tc)
#   2. MonetDB TPC-H Q1-Q22 (per-query hotspot analysis)
#   3. UME (unstructured mesh)
#   4. Spatter (gather/scatter)
#   5. NAS Parallel Benchmarks
#   6. Partitioned Hash Join
#
# Each benchmark is profiled with:
#   - VTune hotspots (function-level)
#   - VTune memory-access (NUMA/CXL bandwidth)
#   - Baseline (no NUMA binding)
#   - numactl -m N (CXL memory node)
#   - Optional: with amem_prefetcher (libhemem.so)
#
# Usage:
#   ./scripts/vtune_profile_all_benchmarks.sh [options]
#
# Options:
#   --bench BENCH     Run only this benchmark (gapbs|monetdb|ume|spatter|nas|hashjoin|all)
#   --numa N          Also profile with numactl -m N
#   --with-hemem      Also profile with amem_prefetcher
#   --collect TYPE    VTune collection (hotspots|memory-access|both) [default: both]
#   --duration N      VTune duration in seconds [default: 30]
#   --force           Overwrite existing results
#   --cxl-gpu-test    Run CXLMemUring Type2 GPU test after profiling
#
# Prerequisites:
#   spack load intel-oneapi-vtune

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."

# ─── Configuration ───────────────────────────────────────────────────────────
RESULT_BASE="${REPO_ROOT}/profile_results/vtune"
VTUNE_DURATION="${VTUNE_DURATION:-30}"
FORCE="${FORCE:-false}"

# Benchmark paths
GAPBS_DIR="${REPO_ROOT}/bench/gapbs"
GAPBS_GRAPH="${GAPBS_DIR}/benchmark/graphs/twitter.sg"
NPB_DIR="${REPO_ROOT}/bench/npb/NPB3.4/NPB3.4-OMP/bin"
SPATTER_BIN="${REPO_ROOT}/bench/spatter/build/spatter"
UME_BIN="${REPO_ROOT}/bench/ume/build/src/ume_serial"
HASHJOIN_HARNESS="${REPO_ROOT}/bench/partitioned-hash-join/programs/sigmod/harness"
HASHJOIN_JOINER="${REPO_ROOT}/bench/partitioned-hash-join/programs/sigmod/joiner"
HASHJOIN_WORKDIR="${REPO_ROOT}/bench/partitioned-hash-join/programs/sigmod/workloads"
MSERVER5="${REPO_ROOT}/bin/monetdb/x86_64-unknown-linux-gnu/mserver5"
MCLIENT="${REPO_ROOT}/bench/MonetDB/build/clients/mapiclient/mclient"
TPCH_QUERIES="${REPO_ROOT}/bench/MonetDB/sql/benchmarks/tpch/queries"
TPCH_DATA="${REPO_ROOT}/bench/MonetDB/sql/benchmarks/tpch/SF-0.01"
MONETDB_PORT="${MONETDB_PORT:-54322}"

# amem_prefetcher
HEMEM_DIR="/home/yuyi/amem_prefetcher"
HEMEM_LIB="${HEMEM_DIR}/src/libhemem.so"

# CXLMemUring
CIRA_BIN="${REPO_ROOT}/build/bin/cira"
CIRA_LIB="${REPO_ROOT}/build/lib/libCXLMemUring.a"
CXL_PCI_DEVICE="0000:3b:00.0"

# Parse arguments
BENCHMARK="all"
NUMA_NODE=""
WITH_HEMEM=false
COLLECT_TYPE="both"
CXL_GPU_TEST=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bench)       BENCHMARK="$2"; shift 2 ;;
        --numa)        NUMA_NODE="$2"; shift 2 ;;
        --with-hemem)  WITH_HEMEM=true; shift ;;
        --collect)     COLLECT_TYPE="$2"; shift 2 ;;
        --duration)    VTUNE_DURATION="$2"; shift 2 ;;
        --force)       FORCE=true; shift ;;
        --cxl-gpu-test) CXL_GPU_TEST=true; shift ;;
        *)             echo "Unknown: $1"; exit 1 ;;
    esac
done

# ─── Helpers ─────────────────────────────────────────────────────────────────
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

# Run VTune with a given collection type
# Args: <collection> <result_dir> <description> [numactl_args...] -- <command...>
vtune_run() {
    local collection="$1"; shift
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

    local vtune_args=(
        -collect "${collection}"
        -result-dir "${result_dir}"
    )
    vtune_args+=(-duration "${VTUNE_DURATION}")

    log "  RUN  ${desc} [${collection}]"

    vtune "${vtune_args[@]}" -- "${full_cmd[@]}" > "${result_dir}.log" 2>&1 || true

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

# Run a benchmark for all requested configurations
# Args: <bench_name> <sub_name> <cmd...>
profile_benchmark() {
    local bench="$1"; shift
    local name="$1"; shift
    local cmd=("$@")

    local collections=()
    case "${COLLECT_TYPE}" in
        hotspots)      collections=(hotspots) ;;
        memory-access) collections=(memory-access) ;;
        both)          collections=(hotspots memory-access) ;;
    esac

    for coll in "${collections[@]}"; do
        local coll_short="${coll//memory-access/memaccess}"

        # Baseline
        vtune_run "${coll}" "${RESULT_BASE}/${bench}/${name}_${coll_short}_baseline" \
            "${bench}/${name} baseline" -- "${cmd[@]}"

        # NUMA
        if [[ -n "${NUMA_NODE}" ]]; then
            vtune_run "${coll}" "${RESULT_BASE}/${bench}/${name}_${coll_short}_numa${NUMA_NODE}" \
                "${bench}/${name} numa${NUMA_NODE}" -m "${NUMA_NODE}" -- "${cmd[@]}"
        fi
    done
}

# ─── GAPBS ───────────────────────────────────────────────────────────────────
profile_gapbs() {
    log "============================================================"
    log " GAPBS Graph Benchmarks"
    log "============================================================"

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

        profile_benchmark "gapbs" "${k}" "${bin}" "${graph_args[@]}" "${extra_args[@]}"
    done
}

# ─── MonetDB TPC-H ──────────────────────────────────────────────────────────
monetdb_start() {
    local prefix=("$@")
    local dbfarm="${RESULT_BASE}/monetdb/dbfarm"
    mkdir -p "${dbfarm}"
    cleanup_monetdb
    sleep 1

    "${prefix[@]}" "${MSERVER5}" \
        --dbpath="${dbfarm}/tpch" \
        --set "mapi_port=${MONETDB_PORT}" \
        --set "mapi_listenaddr=localhost" \
        --set "gdk_nr_threads=0" \
        &>/dev/null &
    MSERVER_PID=$!
    sleep 3

    kill -0 "${MSERVER_PID}" 2>/dev/null || die "mserver5 failed to start"
    log "  mserver5 PID=${MSERVER_PID}"
}

monetdb_load() {
    local count
    count=$("${MCLIENT}" -p "${MONETDB_PORT}" -f csv -s \
        "SELECT count(*) FROM lineitem;" 2>/dev/null | tail -1) || count=0
    if (( count > 0 )); then
        log "  TPC-H loaded (lineitem=${count})"
        return
    fi
    log "  Loading TPC-H SF-0.01..."
    "${MCLIENT}" -p "${MONETDB_PORT}" < "${TPCH_QUERIES}/schema.sql"
    local data_dir
    data_dir="$(cd "${TPCH_DATA}" && pwd)"
    for tbl in region nation supplier customer part partsupp orders lineitem; do
        "${MCLIENT}" -p "${MONETDB_PORT}" -s \
            "COPY INTO ${tbl} FROM '${data_dir}/${tbl}.tbl' USING DELIMITERS '|', E'|\\\\n';"
    done
    log "  TPC-H loaded"
}

# Profile a single TPC-H query by attaching VTune to mserver5
monetdb_profile_query() {
    local qnum="$1"
    local tag="$2"
    local coll="$3"
    local coll_short="${coll//memory-access/memaccess}"
    local qfile="${TPCH_QUERIES}/q$(printf '%02d' "${qnum}").sql"
    local result_dir="${RESULT_BASE}/monetdb/q$(printf '%02d' "${qnum}")_${coll_short}_${tag}"

    [[ -f "${qfile}" ]] || return
    if [[ -d "${result_dir}" && "${FORCE}" != "true" ]]; then
        log "    SKIP Q${qnum} ${tag} [${coll_short}]"
        return
    fi
    rm -rf "${result_dir}"

    # Warm up
    "${MCLIENT}" -p "${MONETDB_PORT}" < "${qfile}" >/dev/null 2>&1 || true

    log "    Q${qnum} ${tag} [${coll_short}]"

    local vtune_args=(-collect "${coll}" -result-dir "${result_dir}" -duration "${VTUNE_DURATION}")
    # no extra knobs - VTune 2025.8 defaults are fine
    vtune_args+=(-target-pid "${MSERVER_PID}")

    vtune "${vtune_args[@]}" &> "${result_dir}.log" &
    local vtune_pid=$!
    sleep 1

    # Run query in loop for sampling duration
    local end_time=$((SECONDS + VTUNE_DURATION - 2))
    local i=0
    while (( SECONDS < end_time )); do
        "${MCLIENT}" -p "${MONETDB_PORT}" < "${qfile}" >/dev/null 2>&1 || true
        ((i++))
    done

    wait "${vtune_pid}" 2>/dev/null || true

    # Reports
    vtune -report summary -result-dir "${result_dir}" -format csv \
        > "${result_dir}_summary.csv" 2>/dev/null || true
    vtune -report hotspots -result-dir "${result_dir}" -format csv \
        -group-by function > "${result_dir}_hotspots.csv" 2>/dev/null || true

    log "    Q${qnum} done (${i} iterations)"
}

profile_monetdb() {
    log "============================================================"
    log " MonetDB TPC-H Per-Query Profiling"
    log "============================================================"

    [[ -x "${MSERVER5}" ]] || { err "Missing: ${MSERVER5}"; return; }
    [[ -x "${MCLIENT}" ]] || { err "Missing: ${MCLIENT}"; return; }

    local queries=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22)
    local collections=()
    case "${COLLECT_TYPE}" in
        hotspots)      collections=(hotspots) ;;
        memory-access) collections=(memory-access) ;;
        both)          collections=(hotspots memory-access) ;;
    esac

    # ── Baseline ──
    log "  --- Baseline ---"
    cleanup_monetdb
    rm -rf "${RESULT_BASE}/monetdb/dbfarm/tpch"
    monetdb_start
    monetdb_load

    for coll in "${collections[@]}"; do
        for q in "${queries[@]}"; do
            monetdb_profile_query "${q}" "baseline" "${coll}"
        done
    done
    cleanup_monetdb

    # ── NUMA ──
    if [[ -n "${NUMA_NODE}" ]]; then
        local max_node
        max_node=$(numactl --hardware 2>/dev/null | grep "^available" | awk '{print $2}')
        if (( NUMA_NODE < max_node )); then
            log "  --- NUMA node ${NUMA_NODE} ---"
            rm -rf "${RESULT_BASE}/monetdb/dbfarm/tpch"
            monetdb_start numactl -m "${NUMA_NODE}"
            monetdb_load

            for coll in "${collections[@]}"; do
                for q in "${queries[@]}"; do
                    monetdb_profile_query "${q}" "numa${NUMA_NODE}" "${coll}"
                done
            done
            cleanup_monetdb
        else
            err "NUMA node ${NUMA_NODE} unavailable (max=${max_node}). Skipping."
        fi
    fi

    # ── With HeMem ──
    if [[ "${WITH_HEMEM}" == "true" && -f "${HEMEM_LIB}" ]]; then
        log "  --- HeMem (amem_prefetcher) ---"
        rm -rf "${RESULT_BASE}/monetdb/dbfarm/tpch"
        monetdb_start env \
            DRAMSIZE=$((2*1024*1024*1024)) \
            NVMSIZE=$((4*1024*1024*1024)) \
            DRAM_NUMA_NODE=0 \
            NVM_NUMA_NODE="${NUMA_NODE:-2}" \
            LD_LIBRARY_PATH="${HEMEM_DIR}/src:${LD_LIBRARY_PATH:-}"
        monetdb_load

        for coll in "${collections[@]}"; do
            for q in "${queries[@]}"; do
                monetdb_profile_query "${q}" "hemem" "${coll}"
            done
        done
        cleanup_monetdb
    fi

    # Generate per-query breakdown
    generate_monetdb_breakdown
}

# ─── Spatter ─────────────────────────────────────────────────────────────────
profile_spatter() {
    log "============================================================"
    log " Spatter Gather/Scatter Benchmarks"
    log "============================================================"
    [[ -x "${SPATTER_BIN}" ]] || { err "Missing: ${SPATTER_BIN}"; return; }

    # AMG application trace
    local amg="${REPO_ROOT}/bench/spatter/standard-suite/app-traces/amg.json"
    [[ -f "${amg}" ]] && profile_benchmark "spatter" "amg" "${SPATTER_BIN}" -f "${amg}"

    # Uniform stride patterns
    for stride in 1 8 64 512; do
        profile_benchmark "spatter" "uniform_s${stride}" \
            "${SPATTER_BIN}" -pUNIFORM:${stride}:1 -l$((2**24))
    done

    # Application traces
    for trace in lulesh pennant nekbone; do
        local tfile="${REPO_ROOT}/bench/spatter/standard-suite/app-traces/${trace}.json"
        [[ -f "${tfile}" ]] && profile_benchmark "spatter" "${trace}" "${SPATTER_BIN}" -f "${tfile}"
    done
}

# ─── NAS Parallel Benchmarks ────────────────────────────────────────────────
profile_nas() {
    log "============================================================"
    log " NAS Parallel Benchmarks"
    log "============================================================"

    for bin in "${NPB_DIR}"/*.x; do
        [[ -x "${bin}" ]] || continue
        local name
        name=$(basename "${bin}" .x)
        profile_benchmark "nas" "${name}" "${bin}"
    done
}

# ─── Hash Join ───────────────────────────────────────────────────────────────
profile_hashjoin() {
    log "============================================================"
    log " Partitioned Hash Join"
    log "============================================================"
    [[ -x "${HASHJOIN_HARNESS}" ]] || { err "Missing: ${HASHJOIN_HARNESS}"; return; }
    [[ -x "${HASHJOIN_JOINER}" ]] || { err "Missing: ${HASHJOIN_JOINER}"; return; }

    # Hash join needs to run from its workload directory (relation files use relative paths)
    if [[ -f "${HASHJOIN_WORKDIR}/small.init" ]]; then
        pushd "${HASHJOIN_WORKDIR}" > /dev/null
        profile_benchmark "hashjoin" "small" \
            "${HASHJOIN_HARNESS}" \
            small.init small.work small.result \
            "${HASHJOIN_JOINER}"
        popd > /dev/null
    fi
}

# ─── UME ─────────────────────────────────────────────────────────────────────
profile_ume() {
    log "============================================================"
    log " UME Unstructured Mesh"
    log "============================================================"
    [[ -x "${UME_BIN}" ]] || { err "Missing: ${UME_BIN}"; return; }

    if [[ -z "${UME_MESH:-}" || ! -f "${UME_MESH:-}" ]]; then
        err "UME_MESH not set. Set UME_MESH=/path/to/mesh.ume to profile UME."
        return
    fi

    profile_benchmark "ume" "serial" "${UME_BIN}" "${UME_MESH}"
}

# ─── CXLMemUring Type2 GPU Test ─────────────────────────────────────────────
test_cxl_gpu() {
    log "============================================================"
    log " CXLMemUring Type2 GPU Device Test (${CXL_PCI_DEVICE})"
    log "============================================================"

    # Check device is present
    if [[ ! -d "/sys/bus/pci/devices/${CXL_PCI_DEVICE}" ]]; then
        err "CXL device ${CXL_PCI_DEVICE} not found in sysfs"
        return
    fi

    local dev_info
    dev_info=$(lspci -nn -s "${CXL_PCI_DEVICE}" 2>/dev/null)
    log "Device: ${dev_info}"

    # Check BAR resources are accessible
    local bar0="/sys/bus/pci/devices/${CXL_PCI_DEVICE}/resource0"
    local bar2="/sys/bus/pci/devices/${CXL_PCI_DEVICE}/resource2"

    if [[ -r "${bar0}" ]]; then
        log "BAR0 (2MB DCOH): accessible"
    else
        err "BAR0 not readable. Run as root or check permissions."
    fi

    if [[ -r "${bar2}" ]]; then
        log "BAR2 (128K CSR): accessible"
    else
        err "BAR2 not readable."
    fi

    # Check CXL capabilities via sysfs
    log ""
    log "CXL device registers:"
    local cxl_devs
    cxl_devs=$(ls /sys/bus/cxl/devices/ 2>/dev/null | tr '\n' ' ')
    log "  CXL bus devices: ${cxl_devs:-none}"

    # Check CXL memory windows
    log ""
    log "CXL Memory Windows (from /proc/iomem):"
    grep "CXL Window" /proc/iomem 2>/dev/null | while read -r line; do
        log "  ${line}"
    done

    # Check NUMA topology
    log ""
    log "NUMA topology:"
    numactl --hardware 2>/dev/null | grep -E "available|node [0-9]+ size" | while read -r line; do
        log "  ${line}"
    done

    # Check if DAX devices exist (CXL memory online)
    local dax_devs
    dax_devs=$(ls /dev/dax* 2>/dev/null | tr '\n' ' ')
    if [[ -n "${dax_devs}" ]]; then
        log "  DAX devices: ${dax_devs}"
    else
        log "  DAX devices: none (CXL memory not onlined)"
        log ""
        log "  To online CXL memory as a NUMA node:"
        log "    cxl create-region -m cxlmem0 -d decoder0.0"
        log "    ndctl create-namespace -m devdax -e namespace0.0"
        log "    daxctl reconfigure-device dax0.0 --mode=system-ram"
    fi

    # Run CXLMemUring test binary if it exists
    local test_bins=(
        "${REPO_ROOT}/build/test/test_type2_gpu"
        "${REPO_ROOT}/build/test/type2_gpu_test"
    )
    for tb in "${test_bins[@]}"; do
        if [[ -x "${tb}" ]]; then
            log ""
            log "Running: ${tb}"
            vtune -collect hotspots \
                -result-dir "${RESULT_BASE}/cxl_gpu/type2_test_hotspots" \
                -duration "${VTUNE_DURATION}" \
                -- "${tb}" 2>&1 | tail -20
            break
        fi
    done

    # Profile BAR0 read latency with a simple test
    log ""
    log "BAR0 CSR read test (checking device responsiveness):"
    python3 - "${bar0}" <<'PYEOF' 2>/dev/null || log "  (requires root for BAR0 mmap)"
import mmap
import struct
import time
import sys

bar0_path = sys.argv[1]
try:
    with open(bar0_path, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 2*1024*1024)
        # Read CSR_BASE + STATUS register (offset 0x180100 + 0x12C = 0x18022C)
        offset = 0x180100 + 0x12C
        status = struct.unpack_from('<I', mm, offset)[0]
        print(f"  GPU Status register: 0x{status:08x}", flush=True)
        if status == 0x00:
            print("  Status: IDLE (device ready)", flush=True)
        elif status == 0x01:
            print("  Status: RUNNING", flush=True)
        elif status == 0x02:
            print("  Status: DONE", flush=True)
        elif status == 0xFF:
            print("  Status: ERROR", flush=True)
        else:
            print(f"  Status: UNKNOWN (0x{status:02x})", flush=True)

        # Latency test: 1000 CSR reads
        start = time.perf_counter_ns()
        for _ in range(1000):
            _ = struct.unpack_from('<I', mm, offset)[0]
        elapsed = time.perf_counter_ns() - start
        print(f"  CSR read latency: {elapsed/1000:.0f} ns/read (avg over 1000 reads)", flush=True)
        mm.close()
except PermissionError:
    print("  Permission denied - run as root for BAR access", flush=True)
except Exception as e:
    print(f"  Error: {e}", flush=True)
PYEOF
}

# ─── MonetDB per-query breakdown analysis ────────────────────────────────────
generate_monetdb_breakdown() {
    log "Generating MonetDB per-query hotspot breakdown..."

    python3 - "${RESULT_BASE}/monetdb" <<'PYEOF'
import csv
import json
import glob
import os
import sys

monetdb_dir = sys.argv[1]
output = os.path.join(monetdb_dir, "query_hotspot_analysis.json")

# Functions to track (from the paper)
target_funcs = {
    "BATgroupavg3": "aggregation", "dosum": "aggregation", "dofsum": "aggregation",
    "BATgroupsum": "aggregation", "BATgroupcount": "aggregation",
    "hashjoin": "join", "mergejoin_int": "join", "mergejoin_lng": "join",
    "densescan_int": "scan_filter", "densescan_lng": "scan_filter",
    "fullscan_int": "scan_filter", "fullscan_lng": "scan_filter",
    "fullscan_str": "scan_filter",
    "__strstr_sse2": "string", "pcmpestri": "string", "STRstr_search": "string",
    "BATsort": "sort", "GDKqsort": "sort",
    "BATproject": "projection",
    "HASHins": "hash", "HASHfind": "hash",
    "runMALsequence": "interpreter", "runMALDataflow": "dataflow",
}

results = {"queries": {}, "configs": set()}

for hotspot_csv in sorted(glob.glob(os.path.join(monetdb_dir, "q*_hotspots_*_hotspots.csv"))):
    basename = os.path.basename(hotspot_csv).replace("_hotspots.csv", "")
    # Parse: q01_hotspots_baseline or q01_hotspots_numa2
    parts = basename.split("_")
    if len(parts) < 3:
        continue
    qname = parts[0].upper()
    config = parts[2] if len(parts) >= 3 else "baseline"
    results["configs"].add(config)

    query_data = {"config": config, "functions": {}, "categories": {}, "top10": []}
    total_time = 0.0

    try:
        with open(hotspot_csv) as f:
            reader = csv.reader(f)
            header = None
            func_col = time_col = -1

            for row in reader:
                if not header:
                    for i, col in enumerate(row):
                        cl = col.strip().lower()
                        if "function" in cl and "call" not in cl:
                            func_col = i
                        if "cpu time" in cl:
                            time_col = i
                    if func_col >= 0 and time_col >= 0:
                        header = row
                    continue

                if func_col >= len(row) or time_col >= len(row):
                    continue
                func = row[func_col].strip()
                try:
                    t = float(row[time_col].strip().replace(",", ""))
                except ValueError:
                    continue

                total_time += t
                query_data["top10"].append({"function": func, "time": t})

                for target, cat in target_funcs.items():
                    if target in func:
                        query_data["functions"][target] = {"time": t, "category": cat}
                        query_data["categories"].setdefault(cat, 0.0)
                        query_data["categories"][cat] += t
                        break
    except Exception:
        continue

    query_data["total_time"] = total_time
    query_data["top10"].sort(key=lambda x: -x["time"])
    query_data["top10"] = query_data["top10"][:10]

    if total_time > 0:
        for f in query_data["functions"].values():
            f["percent"] = round(f["time"] / total_time * 100, 2)
        for cat in query_data["categories"]:
            query_data["categories"][cat] = round(query_data["categories"][cat] / total_time * 100, 2)
        for f in query_data["top10"]:
            f["percent"] = round(f["time"] / total_time * 100, 2)

    results["queries"].setdefault(qname, {})[config] = query_data

results["configs"] = sorted(results["configs"])

# Paper-relevant findings
findings = {
    "aggregation_dominated": [],
    "join_dominated": [],
    "scan_filter_dominated": [],
    "string_dominated": [],
}
for qname, configs in results["queries"].items():
    for config, data in configs.items():
        cats = data.get("categories", {})
        if cats.get("aggregation", 0) > 20:
            findings["aggregation_dominated"].append(f"{qname}({config})")
        if cats.get("join", 0) > 20:
            findings["join_dominated"].append(f"{qname}({config})")
        if cats.get("scan_filter", 0) > 20:
            findings["scan_filter_dominated"].append(f"{qname}({config})")
        if cats.get("string", 0) > 10:
            findings["string_dominated"].append(f"{qname}({config})")

results["paper_findings"] = findings

with open(output, "w") as f:
    json.dump(results, f, indent=2, default=list)

print(f"\n  Analysis: {output}")
print(f"  Queries: {len(results['queries'])}, Configs: {results['configs']}")
print(f"\n  Paper-relevant findings:")
for cat, qs in findings.items():
    if qs:
        print(f"    {cat}: {', '.join(sorted(qs))}")

# Print hotspot shift table
print(f"\n  Per-query dominant hotspot (baseline):")
print(f"  {'Query':<8} {'Top Function':<30} {'Category':<15} {'%':>6}")
print(f"  {'-'*62}")
for qname in sorted(results["queries"].keys()):
    data = results["queries"][qname].get("baseline", results["queries"][qname].get(list(results["queries"][qname].keys())[0], {}))
    top = data.get("top10", [{}])
    if top:
        print(f"  {qname:<8} {top[0].get('function','?')[:28]:<30} {max(data.get('categories',{'?':0}), key=data.get('categories',{'?':0}).get, default='?'):<15} {top[0].get('percent',0):>5.1f}%")
PYEOF
}

# ─── Main ────────────────────────────────────────────────────────────────────
main() {
    command -v vtune &>/dev/null || die "vtune not in PATH. Run: spack load intel-oneapi-vtune"

    log "============================================================"
    log " VTune Comprehensive Benchmark Profiling"
    log " Benchmark: ${BENCHMARK}"
    log " Collect:   ${COLLECT_TYPE}"
    log " Duration:  ${VTUNE_DURATION}s"
    log " NUMA:      ${NUMA_NODE:-none}"
    log " HeMem:     ${WITH_HEMEM}"
    log " CXL GPU:   ${CXL_GPU_TEST}"
    log " Results:   ${RESULT_BASE}"
    log "============================================================"

    mkdir -p "${RESULT_BASE}"

    # Check NUMA
    if [[ -n "${NUMA_NODE}" ]]; then
        local max_node
        max_node=$(numactl --hardware 2>/dev/null | grep "^available" | awk '{print $2}')
        if (( NUMA_NODE >= max_node )); then
            err "NUMA node ${NUMA_NODE} not available (max=${max_node})"
            err "CXL memory may need to be onlined first."
            NUMA_NODE=""
        fi
    fi

    case "${BENCHMARK}" in
        gapbs)    profile_gapbs ;;
        monetdb)  profile_monetdb ;;
        ume)      profile_ume ;;
        spatter)  profile_spatter ;;
        nas)      profile_nas ;;
        hashjoin) profile_hashjoin ;;
        all)
            profile_gapbs
            profile_nas
            profile_spatter
            profile_hashjoin
            profile_ume
            profile_monetdb   # last because it's most complex
            ;;
        *) die "Unknown: ${BENCHMARK}" ;;
    esac

    if [[ "${CXL_GPU_TEST}" == "true" ]]; then
        test_cxl_gpu
    fi

    log ""
    log "============================================================"
    log " All profiling complete"
    log " Results: ${RESULT_BASE}/"
    log "============================================================"
    log ""
    log "View results:"
    log "  vtune-gui <result_dir>"
    log "  vtune -report hotspots -result-dir <result_dir>"
    log ""

    # Count results
    local n_results
    n_results=$(find "${RESULT_BASE}" -name "*_summary.csv" 2>/dev/null | wc -l)
    log "Total result sets: ${n_results}"
}

main
