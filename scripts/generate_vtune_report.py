#!/usr/bin/env python3
"""
Generate combined VTune profiling analysis report from all benchmark results.

Parses hotspot CSV files from profile_results/vtune/ and produces:
  1. Per-benchmark hotspot comparison (baseline vs numa1)
  2. NUMA impact analysis (CPU time delta)
  3. MonetDB per-query hotspot shift analysis
  4. Combined JSON + human-readable summary

Usage:
    python3 scripts/generate_vtune_report.py [--result-dir profile_results/vtune]
"""

import csv
import json
import glob
import os
import sys
from collections import defaultdict

RESULT_DIR = sys.argv[1] if len(sys.argv) > 1 else "profile_results/vtune"


def parse_hotspot_csv(path):
    """Parse a VTune hotspot CSV and return list of (function, cpu_time) tuples."""
    funcs = []
    if not os.path.exists(path):
        return funcs
    try:
        with open(path) as f:
            reader = csv.reader(f, delimiter='\t')
            header = None
            func_col = time_col = -1
            for row in reader:
                if not header:
                    for i, col in enumerate(row):
                        cl = col.strip().lower()
                        if "function" in cl and "full" not in cl and "call" not in cl:
                            func_col = i
                        if cl == "cpu time":
                            time_col = i
                    if func_col >= 0 and time_col >= 0:
                        header = row
                    continue
                if func_col >= len(row) or time_col >= len(row):
                    continue
                try:
                    funcs.append((row[func_col].strip(), float(row[time_col].strip())))
                except (ValueError, IndexError):
                    continue
    except Exception:
        pass
    return funcs


def parse_summary_csv(path):
    """Parse a VTune summary CSV for key metrics."""
    metrics = {}
    if not os.path.exists(path):
        return metrics
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if '\t' in line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        metrics[parts[0].strip()] = parts[1].strip()
    except Exception:
        pass
    return metrics


def analyze_benchmark(bench_dir, bench_name):
    """Analyze all profiles for a benchmark."""
    results = {"name": bench_name, "configs": {}}

    for csv_path in sorted(glob.glob(os.path.join(bench_dir, "*_hotspots.csv"))):
        config_name = os.path.basename(csv_path).replace("_hotspots.csv", "")
        funcs = parse_hotspot_csv(csv_path)
        if not funcs:
            continue

        total_time = sum(t for _, t in funcs)
        top10 = funcs[:10]

        results["configs"][config_name] = {
            "total_cpu_time": round(total_time, 3),
            "top_functions": [
                {
                    "function": f,
                    "cpu_time": round(t, 3),
                    "percent": round(t / total_time * 100, 1) if total_time > 0 else 0,
                }
                for f, t in top10
            ],
        }

        # Also parse summary if available
        summary_path = csv_path.replace("_hotspots.csv", "_summary.csv")
        summary = parse_summary_csv(summary_path)
        if summary:
            results["configs"][config_name]["summary"] = summary

    return results


def compute_numa_impact(results):
    """Compare baseline vs numa1 for each sub-benchmark."""
    impact = []
    configs = results.get("configs", {})

    # Group by sub-benchmark (remove _baseline/_numa1 suffix)
    groups = defaultdict(dict)
    for config_name, data in configs.items():
        if "_baseline" in config_name:
            base = config_name.replace("_hotspots_baseline", "").replace("_baseline", "")
            groups[base]["baseline"] = data
        elif "_numa1" in config_name:
            base = config_name.replace("_hotspots_numa1", "").replace("_numa1", "")
            groups[base]["numa1"] = data

    for sub, variants in sorted(groups.items()):
        if "baseline" in variants and "numa1" in variants:
            b = variants["baseline"]["total_cpu_time"]
            n = variants["numa1"]["total_cpu_time"]
            slowdown = (n / b - 1) * 100 if b > 0 else 0
            impact.append({
                "benchmark": sub,
                "baseline_cpu_time": b,
                "numa1_cpu_time": n,
                "slowdown_pct": round(slowdown, 1),
                "baseline_top": variants["baseline"]["top_functions"][0]["function"]
                    if variants["baseline"]["top_functions"] else "N/A",
                "numa1_top": variants["numa1"]["top_functions"][0]["function"]
                    if variants["numa1"]["top_functions"] else "N/A",
            })

    return impact


def analyze_monetdb_queries(bench_results):
    """Special analysis for MonetDB per-query hotspot shifts."""
    # Target GDK functions
    gdk_categories = {
        "BATgroupavg": "aggregation", "dosum": "aggregation", "dofsum": "aggregation",
        "BATgroupsum": "aggregation", "BATgroupcount": "aggregation",
        "hashjoin": "join", "mergejoin": "join",
        "densescan": "scan_filter", "fullscan": "scan_filter",
        "__strstr": "string", "pcmpestri": "string", "STRstr": "string",
        "BATsort": "sort", "GDKqsort": "sort",
        "BATproject": "projection",
        "HASHins": "hash", "HASHfind": "hash",
        "runMAL": "interpreter",
        "MT_create_thread": "thread_overhead", "accept4": "io_overhead",
        "__pthread_create": "thread_overhead",
    }

    query_analysis = {}
    for config_name, data in bench_results.get("configs", {}).items():
        if not config_name.startswith("q"):
            continue

        qnum = config_name.split("_")[0]  # e.g., "q01"
        tag = "baseline" if "baseline" in config_name else "numa1"

        categories = defaultdict(float)
        total = data.get("total_cpu_time", 0)

        for entry in data.get("top_functions", []):
            func = entry["function"]
            for pattern, cat in gdk_categories.items():
                if pattern in func:
                    categories[cat] += entry["cpu_time"]
                    break

        dominant = max(categories, key=categories.get) if categories else "unknown"

        query_analysis.setdefault(qnum, {})[tag] = {
            "total_cpu_time": total,
            "dominant_category": dominant,
            "category_breakdown": {
                k: round(v / total * 100, 1) if total > 0 else 0
                for k, v in sorted(categories.items(), key=lambda x: -x[1])
            },
            "top3": [e["function"] for e in data.get("top_functions", [])[:3]],
        }

    return query_analysis


def main():
    report = {
        "title": "VTune Comprehensive Benchmark Profiling Report",
        "result_dir": RESULT_DIR,
        "benchmarks": {},
        "numa_impact": {},
        "monetdb_query_analysis": {},
    }

    # Process each benchmark directory
    for bench_name in ["gapbs", "nas", "spatter", "hashjoin", "monetdb"]:
        bench_dir = os.path.join(RESULT_DIR, bench_name)
        if not os.path.isdir(bench_dir):
            continue

        results = analyze_benchmark(bench_dir, bench_name)
        report["benchmarks"][bench_name] = results

        # Compute NUMA impact
        impact = compute_numa_impact(results)
        if impact:
            report["numa_impact"][bench_name] = impact

        # Special MonetDB analysis
        if bench_name == "monetdb":
            report["monetdb_query_analysis"] = analyze_monetdb_queries(results)

    # Save JSON
    output_json = os.path.join(RESULT_DIR, "combined_analysis.json")
    with open(output_json, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("=" * 70)
    print(" VTune Profiling Analysis Report")
    print("=" * 70)

    total_profiles = sum(
        len(b.get("configs", {})) for b in report["benchmarks"].values()
    )
    print(f"\nTotal profiles: {total_profiles}")
    print(f"Output: {output_json}\n")

    # Per-benchmark summary
    for bench_name, bench_data in report["benchmarks"].items():
        n = len(bench_data.get("configs", {}))
        print(f"--- {bench_name} ({n} profiles) ---")

        impact = report["numa_impact"].get(bench_name, [])
        if impact:
            print(f"  {'Benchmark':<25} {'Baseline':>10} {'NUMA-1':>10} {'Slowdown':>10}  Top Function")
            print(f"  {'-'*80}")
            for entry in impact:
                print(
                    f"  {entry['benchmark']:<25} "
                    f"{entry['baseline_cpu_time']:>9.2f}s "
                    f"{entry['numa1_cpu_time']:>9.2f}s "
                    f"{entry['slowdown_pct']:>+9.1f}%  "
                    f"{entry['baseline_top'][:35]}"
                )
        else:
            for config, data in list(bench_data.get("configs", {}).items())[:5]:
                top = data["top_functions"][0] if data.get("top_functions") else {}
                print(
                    f"  {config:<35} "
                    f"{data.get('total_cpu_time', 0):>8.2f}s  "
                    f"{top.get('function', 'N/A')[:30]}"
                )
        print()

    # MonetDB query analysis
    qa = report.get("monetdb_query_analysis", {})
    if qa:
        print("--- MonetDB TPC-H Per-Query Hotspot Analysis ---")
        print(f"  {'Query':<8} {'Dominant':<20} {'Top Functions'}")
        print(f"  {'-'*65}")
        for qname in sorted(qa.keys()):
            for tag in ["baseline", "numa1"]:
                d = qa[qname].get(tag, {})
                if not d:
                    continue
                top3 = ", ".join(d.get("top3", [])[:2])[:40]
                suffix = f" [{tag}]" if tag != "baseline" else ""
                print(f"  {qname}{suffix:<8} {d.get('dominant_category', '?'):<20} {top3}")
        print()

    print(f"Full report: {output_json}")


if __name__ == "__main__":
    main()
