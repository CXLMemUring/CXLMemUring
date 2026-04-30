#!/usr/bin/env bash
# Per-Query VTune Hotspot Profiling for MonetDB TPC-H
#
# Captures query-dependent hotspot shifts described in the paper:
#   - Q1 (aggregation): BATgroupavg3, dosum
#   - Q5,Q9,Q21 (join): hashjoin, mergejoin_int
#   - Q13 (string): __strstr_sse2, pcmpestri
#   - Q6 (scan/filter): densescan_int, fullscan_lng
#
# Usage:
#   ./scripts/vtune_monetdb_perquery.sh [--numa N] [--queries "1 5 6 9 13 21"] [--sf 0.01]
#   ./scripts/vtune_monetdb_perquery.sh --with-hemem   # Run with amem_prefetcher
#
# Prerequisites:
#   spack load intel-oneapi-vtune

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."

# ─── Configuration ───────────────────────────────────────────────────────────
MONETDB_PORT="${MONETDB_PORT:-54322}"
RESULT_BASE="${REPO_ROOT}/profile_results/vtune/monetdb_perquery"
MSERVER5="${REPO_ROOT}/bin/monetdb/x86_64-unknown-linux-gnu/mserver5"
MCLIENT="${REPO_ROOT}/bench/MonetDB/build/clients/mapiclient/mclient"
TPCH_QUERIES="${REPO_ROOT}/bench/MonetDB/sql/benchmarks/tpch/queries"
TPCH_DATA="${REPO_ROOT}/bench/MonetDB/sql/benchmarks/tpch/SF-0.01"
DBFARM="${RESULT_BASE}/dbfarm"
VTUNE_DURATION="${VTUNE_DURATION:-30}"
QUERY_REPEAT="${QUERY_REPEAT:-50}"        # repeat each query N times for stable sampling

# amem_prefetcher
HEMEM_DIR="/home/yuyi/amem_prefetcher"
HEMEM_LIB="${HEMEM_DIR}/src/libhemem.so"

# Default: run representative queries that show diverse hotspots
DEFAULT_QUERIES="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22"

# Query categorization for the paper
declare -A QUERY_CATEGORY=(
    [1]="aggregation"  [2]="subquery"    [3]="join_agg"    [4]="subquery"
    [5]="join"         [6]="scan_filter" [7]="join"        [8]="join"
    [9]="join"         [10]="join_agg"   [11]="subquery"   [12]="join_filter"
    [13]="string"      [14]="join_agg"   [15]="subquery"   [16]="subquery"
    [17]="subquery"    [18]="join_agg"   [19]="filter"     [20]="subquery"
    [21]="join"        [22]="subquery"
)

# GDK functions to track
HOTSPOT_FUNCTIONS=(
    "BATgroupavg3"
    "dosum"
    "dofsum"
    "BATgroupsum"
    "BATgroupcount"
    "hashjoin"
    "mergejoin_int"
    "mergejoin_lng"
    "BATproject"
    "BATselect"
    "BATthetaselect"
    "densescan_int"
    "densescan_lng"
    "fullscan_int"
    "fullscan_lng"
    "fullscan_str"
    "__strstr_sse2"
    "pcmpestri"
    "STRstr_search"
    "BATsort"
    "GDKqsort"
    "runMALsequence"
    "runMALDataflow"
    "HASHins"
    "HASHfind"
    "BATcalcmuldbl_dbl"
    "BATcalcsubdbl_dbl"
)

# ─── Parse arguments ─────────────────────────────────────────────────────────
NUMA_NODE=""
QUERIES="${DEFAULT_QUERIES}"
WITH_HEMEM=false
HEMEM_PREFETCHER="${HEMEM_PREFETCHER:-7}"
DRAM_SIZE="${DRAM_SIZE:-$((2*1024*1024*1024))}"
NVM_SIZE="${NVM_SIZE:-$((4*1024*1024*1024))}"
DRAM_NUMA_NODE="${DRAM_NUMA_NODE:-0}"
NVM_NUMA_NODE="${NVM_NUMA_NODE:-2}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --numa)       NUMA_NODE="$2"; shift 2 ;;
        --queries)    QUERIES="$2"; shift 2 ;;
        --sf)         shift 2 ;; # scale factor placeholder
        --with-hemem) WITH_HEMEM=true; shift ;;
        --prefetcher) HEMEM_PREFETCHER="$2"; shift 2 ;;
        --duration)   VTUNE_DURATION="$2"; shift 2 ;;
        --repeat)     QUERY_REPEAT="$2"; shift 2 ;;
        --force)      FORCE=true; shift ;;
        *)            echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ─── Helpers ─────────────────────────────────────────────────────────────────
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
err()  { echo "[$(date '+%H:%M:%S')] ERROR: $*" >&2; }
die()  { err "$@"; exit 1; }
MSERVER_PID=""

cleanup() {
    if [[ -n "${MSERVER_PID}" ]]; then
        kill "${MSERVER_PID}" 2>/dev/null || true
        wait "${MSERVER_PID}" 2>/dev/null || true
    fi
    pkill -f "mserver5.*port=${MONETDB_PORT}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

start_mserver() {
    local prefix=("$@")
    mkdir -p "${DBFARM}"
    pkill -f "mserver5.*port=${MONETDB_PORT}" 2>/dev/null || true
    sleep 1

    log "Starting mserver5 (port=${MONETDB_PORT})..."
    if [[ ${#prefix[@]} -gt 0 ]]; then
        log "  prefix: ${prefix[*]}"
    fi

    "${prefix[@]}" "${MSERVER5}" \
        --dbpath="${DBFARM}/tpch" \
        --set "mapi_port=${MONETDB_PORT}" \
        --set "mapi_listenaddr=localhost" \
        --set "gdk_nr_threads=0" \
        &>/dev/null &
    MSERVER_PID=$!
    sleep 3

    if ! kill -0 "${MSERVER_PID}" 2>/dev/null; then
        die "mserver5 failed to start"
    fi
    log "mserver5 started (PID=${MSERVER_PID})"
}

mclient_cmd() {
    "${MCLIENT}" -p "${MONETDB_PORT}" "$@"
}

setup_tpch() {
    local count
    count=$(mclient_cmd -f csv -s "SELECT count(*) FROM lineitem;" 2>/dev/null | tail -1) || count=0
    if (( count > 0 )); then
        log "TPC-H already loaded (lineitem: ${count} rows)"
        return
    fi

    log "Loading TPC-H data..."
    mclient_cmd < "${TPCH_QUERIES}/schema.sql"

    local data_dir
    data_dir="$(cd "${TPCH_DATA}" && pwd)"
    for tbl in region nation supplier customer part partsupp orders lineitem; do
        mclient_cmd -s "COPY INTO ${tbl} FROM '${data_dir}/${tbl}.tbl' USING DELIMITERS '|', E'|\\\\n';"
    done
    log "TPC-H data loaded"
}

# ─── Per-query VTune profiling ───────────────────────────────────────────────
profile_query() {
    local qnum="$1"
    local tag="$2"
    local qfile="${TPCH_QUERIES}/q$(printf '%02d' ${qnum}).sql"

    [[ -f "${qfile}" ]] || { err "Missing: ${qfile}"; return 1; }

    local cat="${QUERY_CATEGORY[$qnum]:-unknown}"
    local result_dir="${RESULT_BASE}/${tag}/q$(printf '%02d' ${qnum})_${cat}"

    if [[ -d "${result_dir}" && "${FORCE:-}" != "true" ]]; then
        log "  SKIP Q${qnum} (${cat}) - exists"
        return 0
    fi
    rm -rf "${result_dir}"

    log "  PROFILE Q${qnum} (${cat}) [${tag}] - repeating ${QUERY_REPEAT}x over ${VTUNE_DURATION}s"

    # Warm up the query once
    mclient_cmd < "${qfile}" >/dev/null 2>&1 || true

    # Start VTune hotspot collection attached to mserver5
    vtune -collect hotspots \
        -result-dir "${result_dir}" \
        -duration "${VTUNE_DURATION}" \
        -target-pid "${MSERVER_PID}" &
    local vtune_pid=$!

    sleep 1  # let vtune attach

    # Run query repeatedly for stable sampling
    local i=0
    local end_time=$((SECONDS + VTUNE_DURATION - 2))
    while (( SECONDS < end_time && i < QUERY_REPEAT )); do
        mclient_cmd < "${qfile}" >/dev/null 2>&1 || true
        ((i++))
    done

    wait "${vtune_pid}" 2>/dev/null || true
    log "    Executed Q${qnum} ${i} times"

    # Extract reports
    extract_reports "${result_dir}" "${qnum}" "${cat}" "${tag}"
}

extract_reports() {
    local result_dir="$1"
    local qnum="$2"
    local cat="$3"
    local tag="$4"

    # Summary report
    vtune -report summary -result-dir "${result_dir}" -format csv \
        > "${result_dir}_summary.csv" 2>/dev/null || true

    # Full hotspot report
    vtune -report hotspots -result-dir "${result_dir}" -format csv \
        -group-by function \
        > "${result_dir}_hotspots.csv" 2>/dev/null || true

    # Top-down microarchitecture report
    vtune -report hotspots -result-dir "${result_dir}" -format csv \
        -group-by function \
        -column "CPU Time:Self,CPU Time:Effective Time:Idle:Self,CPU Time:Effective Time:Poor:Self,CPU Time:Spin Time:Self,Module" \
        > "${result_dir}_topdown.csv" 2>/dev/null || true

    # Extract targeted function percentages
    extract_function_breakdown "${result_dir}" "${qnum}" "${cat}" "${tag}"
}

extract_function_breakdown() {
    local result_dir="$1"
    local qnum="$2"
    local cat="$3"
    local tag="$4"
    local hotspots_csv="${result_dir}_hotspots.csv"
    local output_json="${result_dir}_breakdown.json"

    [[ -f "${hotspots_csv}" ]] || return

    python3 - "${hotspots_csv}" "${output_json}" "${qnum}" "${cat}" "${tag}" <<'PYEOF'
import csv
import json
import sys
import re

hotspots_csv = sys.argv[1]
output_json = sys.argv[2]
qnum = int(sys.argv[3])
category = sys.argv[4]
tag = sys.argv[5]

# Target functions to track
target_funcs = {
    # Aggregation
    "BATgroupavg3": "aggregation",
    "dosum": "aggregation",
    "dofsum": "aggregation",
    "BATgroupsum": "aggregation",
    "BATgroupcount": "aggregation",
    # Join
    "hashjoin": "join",
    "mergejoin_int": "join",
    "mergejoin_lng": "join",
    # Scan/Filter
    "densescan_int": "scan_filter",
    "densescan_lng": "scan_filter",
    "fullscan_int": "scan_filter",
    "fullscan_lng": "scan_filter",
    "fullscan_str": "scan_filter",
    # String
    "__strstr_sse2": "string",
    "pcmpestri": "string",
    "STRstr_search": "string",
    # Sort
    "BATsort": "sort",
    "GDKqsort": "sort",
    # Projection
    "BATproject": "projection",
    # Hash
    "HASHins": "hash",
    "HASHfind": "hash",
    # Execution
    "runMALsequence": "interpreter",
    "runMALDataflow": "dataflow",
    # Arithmetic
    "BATcalcmuldbl_dbl": "arithmetic",
    "BATcalcsubdbl_dbl": "arithmetic",
}

result = {
    "query": f"Q{qnum}",
    "category": category,
    "tag": tag,
    "functions": {},
    "category_breakdown": {},
    "total_cpu_time": 0.0,
    "top_functions": [],
}

try:
    with open(hotspots_csv, "r") as f:
        reader = csv.reader(f)
        header = None
        func_col = -1
        time_col = -1

        for row in reader:
            if not header:
                # Find the header row
                for i, col in enumerate(row):
                    cl = col.strip().lower()
                    if "function" in cl and "call" not in cl:
                        func_col = i
                    if "cpu time" in cl and "self" not in cl and "idle" not in cl:
                        time_col = i
                    if "cpu time" in cl and func_col >= 0 and time_col < 0:
                        time_col = i
                if func_col >= 0 and time_col >= 0:
                    header = row
                    # Try to find better time column
                    for i, col in enumerate(row):
                        cl = col.strip().lower()
                        if cl in ("cpu time", "cpu time:self"):
                            time_col = i
                continue

            if func_col >= len(row) or time_col >= len(row):
                continue

            func_name = row[func_col].strip()
            try:
                cpu_time = float(row[time_col].strip().replace(",", ""))
            except (ValueError, IndexError):
                continue

            result["total_cpu_time"] += cpu_time
            result["top_functions"].append({
                "function": func_name,
                "cpu_time": cpu_time,
            })

            # Check against target functions
            for target, tcat in target_funcs.items():
                if target in func_name:
                    result["functions"][target] = {
                        "cpu_time": cpu_time,
                        "category": tcat,
                    }
                    result["category_breakdown"].setdefault(tcat, 0.0)
                    result["category_breakdown"][tcat] += cpu_time
                    break

    # Sort top functions and keep top 20
    result["top_functions"].sort(key=lambda x: x["cpu_time"], reverse=True)
    result["top_functions"] = result["top_functions"][:20]

    # Calculate percentages
    total = result["total_cpu_time"]
    if total > 0:
        for f in result["functions"].values():
            f["percent"] = round(f["cpu_time"] / total * 100, 2)
        for cat in result["category_breakdown"]:
            result["category_breakdown"][cat] = round(
                result["category_breakdown"][cat] / total * 100, 2
            )
        for f in result["top_functions"]:
            f["percent"] = round(f["cpu_time"] / total * 100, 2)

except Exception as e:
    result["error"] = str(e)

with open(output_json, "w") as f:
    json.dump(result, f, indent=2)

# Print summary
print(f"    Q{qnum} ({category}):")
total = result["total_cpu_time"]
if total > 0:
    for entry in result["top_functions"][:5]:
        pct = entry.get("percent", 0)
        print(f"      {pct:5.1f}%  {entry['function']}")
    if result["category_breakdown"]:
        cats = sorted(result["category_breakdown"].items(), key=lambda x: -x[1])
        cat_str = ", ".join(f"{c}={v:.1f}%" for c, v in cats)
        print(f"      Categories: {cat_str}")
PYEOF
}

# ─── Generate combined analysis ─────────────────────────────────────────────
generate_combined_report() {
    local tag="$1"
    local report_dir="${RESULT_BASE}/${tag}"
    local combined="${report_dir}/combined_analysis.json"

    log "Generating combined analysis for ${tag}..."

    python3 - "${report_dir}" "${combined}" <<'PYEOF'
import json
import glob
import os
import sys

report_dir = sys.argv[1]
combined_path = sys.argv[2]

all_queries = {}
category_totals = {}

for breakdown_file in sorted(glob.glob(os.path.join(report_dir, "*_breakdown.json"))):
    with open(breakdown_file) as f:
        data = json.load(f)

    qname = data.get("query", "?")
    all_queries[qname] = {
        "category": data.get("category", "unknown"),
        "total_cpu_time": data.get("total_cpu_time", 0),
        "top_5_functions": data.get("top_functions", [])[:5],
        "gdk_breakdown": data.get("category_breakdown", {}),
        "tracked_functions": data.get("functions", {}),
    }

    for cat, pct in data.get("category_breakdown", {}).items():
        category_totals.setdefault(cat, []).append({
            "query": qname,
            "percent": pct,
        })

# Identify query clusters by dominant hotspot type
clusters = {}
for qname, qdata in all_queries.items():
    if qdata["gdk_breakdown"]:
        dominant = max(qdata["gdk_breakdown"], key=qdata["gdk_breakdown"].get)
    else:
        dominant = "other"
    clusters.setdefault(dominant, []).append(qname)

combined = {
    "analysis": "TPC-H Per-Query Hotspot Analysis",
    "description": "Demonstrates that database workload hotspots are query-dependent",
    "queries": all_queries,
    "hotspot_clusters": clusters,
    "category_summary": {
        cat: {
            "queries": entries,
            "avg_percent": round(sum(e["percent"] for e in entries) / len(entries), 2),
        }
        for cat, entries in category_totals.items()
    },
    "key_findings": {
        "aggregation_heavy": [q for q, d in all_queries.items()
                              if d["gdk_breakdown"].get("aggregation", 0) > 20],
        "join_heavy": [q for q, d in all_queries.items()
                       if d["gdk_breakdown"].get("join", 0) > 20],
        "scan_filter_heavy": [q for q, d in all_queries.items()
                              if d["gdk_breakdown"].get("scan_filter", 0) > 20],
        "string_heavy": [q for q, d in all_queries.items()
                         if d["gdk_breakdown"].get("string", 0) > 10],
    },
}

with open(combined_path, "w") as f:
    json.dump(combined, f, indent=2)

print(f"\n  Combined analysis: {combined_path}")
print(f"  Queries analyzed: {len(all_queries)}")
print(f"\n  Hotspot Clusters:")
for cat, queries in clusters.items():
    print(f"    {cat}: {', '.join(sorted(queries))}")
PYEOF
}

# ─── MonetDB built-in profiler integration ───────────────────────────────────
run_monetdb_profiler() {
    local qnum="$1"
    local tag="$2"
    local qfile="${TPCH_QUERIES}/q$(printf '%02d' ${qnum}).sql"
    local cat="${QUERY_CATEGORY[$qnum]:-unknown}"
    local output="${RESULT_BASE}/${tag}/q$(printf '%02d' ${qnum})_${cat}_mal_trace.json"

    mkdir -p "$(dirname "${output}")"

    log "  MAL-trace Q${qnum} (${cat})"

    # Enable profiler, run query, collect trace
    mclient_cmd -s "CALL profiler.start();" 2>/dev/null || true
    mclient_cmd < "${qfile}" >/dev/null 2>&1 || true
    mclient_cmd -s "CALL profiler.stop();" 2>/dev/null || true

    # Export trace (MonetDB outputs JSON lines to profiler stream)
    # The trace data from the query with timing is available via stethoscope or
    # the profiler.getTrace() function. We capture what we can via mclient.
    mclient_cmd -f csv -s "
        SELECT
            \"module\",
            \"function\",
            CAST(SUM(\"ticks\") AS BIGINT) as total_ticks,
            COUNT(*) as call_count,
            CAST(AVG(\"ticks\") AS BIGINT) as avg_ticks
        FROM sys.tracelog()
        GROUP BY \"module\", \"function\"
        ORDER BY total_ticks DESC
        LIMIT 30;
    " 2>/dev/null > "${output}" || true

    if [[ -s "${output}" ]]; then
        log "    MAL trace saved: ${output}"
    fi
}

# ─── Main orchestration ─────────────────────────────────────────────────────
main() {
    command -v vtune &>/dev/null || die "vtune not in PATH. Run: spack load intel-oneapi-vtune"
    [[ -x "${MSERVER5}" ]] || die "Missing: ${MSERVER5}"
    [[ -x "${MCLIENT}" ]] || die "Missing: ${MCLIENT}"
    [[ -d "${TPCH_QUERIES}" ]] || die "Missing: ${TPCH_QUERIES}"

    mkdir -p "${RESULT_BASE}"

    log "=================================================="
    log " MonetDB TPC-H Per-Query Hotspot Analysis"
    log " Queries: ${QUERIES}"
    log " Duration: ${VTUNE_DURATION}s per query"
    log " Repeat: ${QUERY_REPEAT}x per query"
    log " NUMA: ${NUMA_NODE:-none}"
    log " HeMem: ${WITH_HEMEM}"
    log " Results: ${RESULT_BASE}"
    log "=================================================="

    # ── Baseline run (no NUMA, no HeMem) ─────────────────────────────────
    local tag="baseline"
    log ""
    log "=== Phase 1: Baseline profiling ==="

    cleanup
    rm -rf "${DBFARM}/tpch"
    start_mserver
    setup_tpch

    for q in ${QUERIES}; do
        profile_query "${q}" "${tag}"
        run_monetdb_profiler "${q}" "${tag}"
    done

    generate_combined_report "${tag}"
    cleanup

    # ── NUMA run (numactl -m N) ──────────────────────────────────────────
    if [[ -n "${NUMA_NODE}" ]]; then
        tag="numa${NUMA_NODE}"
        log ""
        log "=== Phase 2: NUMA node ${NUMA_NODE} profiling ==="

        # Verify node exists
        local max_node
        max_node=$(numactl --hardware 2>/dev/null | grep "^available" | awk '{print $2}')
        if (( NUMA_NODE >= max_node )); then
            err "NUMA node ${NUMA_NODE} not available (max=${max_node}). Skipping."
        else
            rm -rf "${DBFARM}/tpch"
            start_mserver numactl -m "${NUMA_NODE}"
            setup_tpch

            for q in ${QUERIES}; do
                profile_query "${q}" "${tag}"
                run_monetdb_profiler "${q}" "${tag}"
            done

            generate_combined_report "${tag}"
            cleanup
        fi
    fi

    # ── HeMem / amem_prefetcher run ──────────────────────────────────────
    if [[ "${WITH_HEMEM}" == "true" ]]; then
        tag="hemem_pf${HEMEM_PREFETCHER}"
        log ""
        log "=== Phase 3: HeMem/amem_prefetcher (prefetcher=${HEMEM_PREFETCHER}) ==="

        if [[ ! -f "${HEMEM_LIB}" ]]; then
            err "libhemem.so not found at ${HEMEM_LIB}. Build amem_prefetcher first."
        else
            rm -rf "${DBFARM}/tpch"

            # Start mserver5 with hemem interposed
            log "Starting mserver5 with libhemem.so LD_LIBRARY_PATH interposition"
            start_mserver env \
                DRAMSIZE="${DRAM_SIZE}" \
                NVMSIZE="${NVM_SIZE}" \
                DRAM_NUMA_NODE="${DRAM_NUMA_NODE}" \
                NVM_NUMA_NODE="${NVM_NUMA_NODE}" \
                LD_LIBRARY_PATH="${HEMEM_DIR}/src:${LD_LIBRARY_PATH:-}"

            setup_tpch

            for q in ${QUERIES}; do
                profile_query "${q}" "${tag}"
                run_monetdb_profiler "${q}" "${tag}"
            done

            generate_combined_report "${tag}"
            cleanup
        fi
    fi

    # ── Final combined cross-config comparison ───────────────────────────
    log ""
    log "=== Generating cross-configuration comparison ==="
    generate_cross_comparison

    log ""
    log "=================================================="
    log " Done. Results in: ${RESULT_BASE}"
    log "=================================================="
}

generate_cross_comparison() {
    python3 - "${RESULT_BASE}" <<'PYEOF'
import json
import glob
import os
import sys

base_dir = sys.argv[1]
output = os.path.join(base_dir, "cross_comparison.json")

configs = {}
for analysis_file in sorted(glob.glob(os.path.join(base_dir, "*/combined_analysis.json"))):
    config_name = os.path.basename(os.path.dirname(analysis_file))
    with open(analysis_file) as f:
        configs[config_name] = json.load(f)

if not configs:
    print("  No completed configurations to compare.")
    return

comparison = {
    "configurations": list(configs.keys()),
    "per_query_comparison": {},
}

# Get all queries across configs
all_queries = set()
for config_data in configs.values():
    all_queries.update(config_data.get("queries", {}).keys())

for query in sorted(all_queries):
    comparison["per_query_comparison"][query] = {}
    for config_name, config_data in configs.items():
        qdata = config_data.get("queries", {}).get(query, {})
        comparison["per_query_comparison"][query][config_name] = {
            "total_cpu_time": qdata.get("total_cpu_time", 0),
            "gdk_breakdown": qdata.get("gdk_breakdown", {}),
            "top_function": qdata.get("top_5_functions", [{}])[0].get("function", "N/A")
                if qdata.get("top_5_functions") else "N/A",
        }

with open(output, "w") as f:
    json.dump(comparison, f, indent=2)

print(f"  Cross-comparison: {output}")
print(f"  Configurations: {', '.join(configs.keys())}")

# Print comparison table
print(f"\n  {'Query':<8}", end="")
for cfg in configs:
    print(f" {cfg:<30}", end="")
print()
print("  " + "-" * (8 + 30 * len(configs)))

for query in sorted(all_queries):
    print(f"  {query:<8}", end="")
    for cfg in configs:
        qdata = comparison["per_query_comparison"].get(query, {}).get(cfg, {})
        top = qdata.get("top_function", "N/A")[:25]
        cpu = qdata.get("total_cpu_time", 0)
        print(f" {top:<20} {cpu:>7.2f}s ", end="")
    print()
PYEOF
}

main
