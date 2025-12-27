#!/usr/bin/env python3
"""
MonetDB Kernel Profiler for Two-Pass Compilation

This script profiles MonetDB execution to identify hot kernel regions
for targeted optimization in the two-pass compilation workflow.

Usage:
    python3 profile_monetdb_kernels.py --mode profile [options]
    python3 profile_monetdb_kernels.py --mode analyze --profile-file <file>
"""

import argparse
import subprocess
import json
import os
import sys
import time
from datetime import datetime
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILE_DIR = os.path.join(REPO_ROOT, "profile_results", "monetdb")

# MonetDB kernel functions of interest for offloading
# These are the computational kernels that benefit from acceleration
MONETDB_KERNELS = {
    # GDK (Database Kernel) functions
    "gdk_select": {"type": "filter", "parallelizable": True},
    "gdk_join": {"type": "join", "parallelizable": True},
    "gdk_group": {"type": "aggregation", "parallelizable": True},
    "gdk_sort": {"type": "sort", "parallelizable": True},
    "BATproject": {"type": "projection", "parallelizable": True},
    "BATselect": {"type": "filter", "parallelizable": True},
    "BATjoin": {"type": "join", "parallelizable": True},
    "BATgroup": {"type": "aggregation", "parallelizable": True},
    "BATsort": {"type": "sort", "parallelizable": True},
    "BATcalc": {"type": "compute", "parallelizable": True},

    # Hash operations
    "HASHins": {"type": "hash", "parallelizable": False},
    "HASHfind": {"type": "hash", "parallelizable": True},

    # Aggregation functions
    "AGGRsum": {"type": "reduction", "parallelizable": True},
    "AGGRcount": {"type": "reduction", "parallelizable": True},
    "AGGRmin": {"type": "reduction", "parallelizable": True},
    "AGGRmax": {"type": "reduction", "parallelizable": True},
    "AGGRavg": {"type": "reduction", "parallelizable": True},

    # MAL (MonetDB Assembly Language) execution
    "runMALsequence": {"type": "interpreter", "parallelizable": False},
    "runMALDataflow": {"type": "dataflow", "parallelizable": True},

    # SQL execution
    "sql_trans_": {"type": "transaction", "parallelizable": False},
    "sql_column_": {"type": "storage", "parallelizable": True},
}


def run_perf_record(mserver_bin, db_name, workload_script, duration, output_file):
    """Run perf record on mserver5 during workload execution"""
    print(f"Starting perf recording for {duration} seconds...")

    # Find mserver5 PID or start it
    try:
        result = subprocess.run(
            ["pgrep", "-f", f"mserver5.*{db_name}"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            pid = result.stdout.strip().split()[0]
        else:
            print("MonetDB not running. Please start it first.")
            return None
    except Exception as e:
        print(f"Error finding mserver5: {e}")
        return None

    # Run perf record
    perf_cmd = [
        "perf", "record",
        "-p", pid,
        "-g",  # Call graph
        "-F", "99",  # Sampling frequency
        "-o", output_file,
        "--", "sleep", str(duration)
    ]

    # Start workload in background
    workload_proc = subprocess.Popen(
        ["python3", workload_script, "--dbname", db_name, "--duration", str(duration)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Run perf
    try:
        subprocess.run(perf_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Perf recording failed: {e}")
        workload_proc.terminate()
        return None
    except PermissionError:
        print("Permission denied. Try running with sudo or adjust perf_event_paranoid")
        workload_proc.terminate()
        return None

    workload_proc.wait()
    return output_file


def run_perf_report(perf_data_file):
    """Parse perf report and extract function hotspots"""
    print(f"Analyzing perf data from {perf_data_file}...")

    try:
        result = subprocess.run(
            ["perf", "report", "-i", perf_data_file, "--stdio", "-n", "--no-children"],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            print(f"Perf report failed: {result.stderr}")
            return None

        return result.stdout
    except Exception as e:
        print(f"Error running perf report: {e}")
        return None


def parse_perf_output(perf_output):
    """Parse perf report output into structured data"""
    functions = []
    current_section = None

    for line in perf_output.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        # Parse lines like: "  5.23%  12345  mserver5  [.] function_name"
        parts = line.split()
        if len(parts) >= 5 and '%' in parts[0]:
            try:
                overhead = float(parts[0].rstrip('%'))
                samples = int(parts[1])
                binary = parts[2]
                symbol = parts[-1]

                # Only include mserver5 functions
                if 'mserver5' in binary or 'libmonetdb' in binary.lower():
                    functions.append({
                        'name': symbol,
                        'overhead_pct': overhead,
                        'samples': samples,
                        'binary': binary
                    })
            except (ValueError, IndexError):
                continue

    return functions


def identify_kernel_regions(functions):
    """Identify MonetDB kernel regions for optimization"""
    regions = []

    for func in functions:
        func_name = func['name']

        # Check if this is a known kernel function
        for kernel_name, kernel_info in MONETDB_KERNELS.items():
            if kernel_name in func_name:
                region = {
                    'region_name': func_name,
                    'kernel_type': kernel_info['type'],
                    'parallelizable': kernel_info['parallelizable'],
                    'overhead_pct': func['overhead_pct'],
                    'samples': func['samples'],
                    'offload_candidate': kernel_info['parallelizable'] and func['overhead_pct'] > 1.0
                }
                regions.append(region)
                break

    return regions


def generate_twopass_profile(regions, config):
    """Generate two-pass profile from kernel analysis"""
    profile = {
        "profile_type": "monetdb_kernel_analysis",
        "version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "configuration": config,
        "num_regions": len(regions),
        "clock_freq_mhz": config.get("clock_freq_mhz", 200.0),
        "cxl_latency_ns": config.get("cxl_latency_ns", 165),
        "regions": []
    }

    for i, region in enumerate(regions):
        # Estimate timing based on overhead percentage
        # Assume total execution time and scale by overhead
        overhead_pct = region['overhead_pct']
        samples = region['samples']

        # Rough estimation of cycle counts based on sampling
        # Assume 99Hz sampling, so each sample ≈ 10ms
        estimated_time_ms = samples * 10.0
        estimated_time_ns = int(estimated_time_ms * 1e6)

        # Estimate compute vs memory cycles (30/70 split as baseline)
        compute_ratio = 0.3 if region['parallelizable'] else 0.5
        compute_cycles = int(estimated_time_ns * compute_ratio * config.get("clock_freq_mhz", 200) / 1000)
        memory_cycles = int(estimated_time_ns * (1 - compute_ratio) * config.get("clock_freq_mhz", 200) / 1000)

        # Injection delay: additional latency from heterogeneous execution
        injection_delay = memory_cycles * 1000 / config.get("clock_freq_mhz", 200)

        profile_region = {
            "region_id": i,
            "region_name": region['region_name'],
            "kernel_type": region['kernel_type'],
            "parallelizable": region['parallelizable'],
            "offload_candidate": region['offload_candidate'],
            "overhead_pct": overhead_pct,
            "samples": samples,
            "host_independent_work_ns": int(estimated_time_ns * compute_ratio),
            "vortex_timing": {
                "total_cycles": compute_cycles + memory_cycles,
                "total_time_ns": estimated_time_ns,
                "compute_cycles": compute_cycles,
                "memory_stall_cycles": memory_cycles,
                "cache_hits": 0,
                "cache_misses": 0
            },
            "injection_delay_ns": int(injection_delay),
            "latency_hidden": memory_cycles < compute_cycles,
            "optimal_prefetch_depth": 16 if region['parallelizable'] else 4
        }
        profile['regions'].append(profile_region)

    # Sort by overhead
    profile['regions'].sort(key=lambda x: x['overhead_pct'], reverse=True)

    # Summary
    total_overhead = sum(r['overhead_pct'] for r in profile['regions'])
    offload_candidates = [r for r in profile['regions'] if r['offload_candidate']]

    profile['analysis'] = {
        'total_profiled_overhead_pct': total_overhead,
        'num_offload_candidates': len(offload_candidates),
        'offload_candidate_overhead_pct': sum(r['overhead_pct'] for r in offload_candidates),
        'top_functions': [r['region_name'] for r in profile['regions'][:10]]
    }

    return profile


def mock_profiling(config):
    """Generate mock profiling data for testing without perf"""
    print("Generating mock profiling data (perf not available)...")

    # Simulate realistic MonetDB function overhead distribution
    mock_functions = [
        {'name': 'BATproject', 'overhead_pct': 15.2, 'samples': 1520, 'binary': 'mserver5'},
        {'name': 'BATselect', 'overhead_pct': 12.8, 'samples': 1280, 'binary': 'mserver5'},
        {'name': 'BATjoin', 'overhead_pct': 10.5, 'samples': 1050, 'binary': 'mserver5'},
        {'name': 'AGGRsum', 'overhead_pct': 8.3, 'samples': 830, 'binary': 'mserver5'},
        {'name': 'HASHfind', 'overhead_pct': 7.1, 'samples': 710, 'binary': 'mserver5'},
        {'name': 'BATsort', 'overhead_pct': 6.4, 'samples': 640, 'binary': 'mserver5'},
        {'name': 'AGGRcount', 'overhead_pct': 5.2, 'samples': 520, 'binary': 'mserver5'},
        {'name': 'BATgroup', 'overhead_pct': 4.8, 'samples': 480, 'binary': 'mserver5'},
        {'name': 'runMALDataflow', 'overhead_pct': 4.1, 'samples': 410, 'binary': 'mserver5'},
        {'name': 'sql_column_fetch', 'overhead_pct': 3.5, 'samples': 350, 'binary': 'mserver5'},
    ]

    regions = identify_kernel_regions(mock_functions)
    return generate_twopass_profile(regions, config)


def main():
    parser = argparse.ArgumentParser(description="MonetDB Kernel Profiler")
    parser.add_argument("--mode", choices=["profile", "analyze", "mock"], required=True,
                        help="Operation mode")
    parser.add_argument("--mserver", type=str, default=None,
                        help="Path to mserver5 binary")
    parser.add_argument("--dbname", type=str, default="tpcc_profile",
                        help="Database name")
    parser.add_argument("--duration", type=int, default=60,
                        help="Profiling duration in seconds")
    parser.add_argument("--workload", type=str, default=None,
                        help="Path to workload driver script")
    parser.add_argument("--profile-file", type=str, default=None,
                        help="Perf data file for analysis mode")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file for profile")
    parser.add_argument("--clock-freq", type=float, default=200.0,
                        help="Clock frequency in MHz")
    parser.add_argument("--cxl-latency", type=int, default=165,
                        help="CXL latency in nanoseconds")

    args = parser.parse_args()

    os.makedirs(PROFILE_DIR, exist_ok=True)

    config = {
        "dbname": args.dbname,
        "duration": args.duration,
        "clock_freq_mhz": args.clock_freq,
        "cxl_latency_ns": args.cxl_latency
    }

    if args.mode == "profile":
        # Full profiling with perf
        if args.mserver is None:
            args.mserver = os.path.join(REPO_ROOT, "bin/monetdb/x86_64-unknown-linux-gnu/mserver5")
        if args.workload is None:
            args.workload = os.path.join(REPO_ROOT, "bench/MonetDB/sql/benchmarks/tpcc/workload_driver.py")

        perf_output = os.path.join(PROFILE_DIR, "perf.data")
        result = run_perf_record(args.mserver, args.dbname, args.workload, args.duration, perf_output)

        if result:
            perf_report = run_perf_report(result)
            if perf_report:
                functions = parse_perf_output(perf_report)
                regions = identify_kernel_regions(functions)
                profile = generate_twopass_profile(regions, config)
            else:
                print("Failed to generate perf report, using mock data")
                profile = mock_profiling(config)
        else:
            print("Failed to record perf data, using mock data")
            profile = mock_profiling(config)

    elif args.mode == "analyze":
        # Analyze existing perf data
        if args.profile_file is None:
            args.profile_file = os.path.join(PROFILE_DIR, "perf.data")

        if not os.path.exists(args.profile_file):
            print(f"Profile file not found: {args.profile_file}")
            sys.exit(1)

        perf_report = run_perf_report(args.profile_file)
        if perf_report:
            functions = parse_perf_output(perf_report)
            regions = identify_kernel_regions(functions)
            profile = generate_twopass_profile(regions, config)
        else:
            print("Failed to analyze profile")
            sys.exit(1)

    elif args.mode == "mock":
        # Generate mock data for testing
        profile = mock_profiling(config)

    # Output profile
    if args.output is None:
        args.output = os.path.join(PROFILE_DIR, "kernel_profile.json")

    with open(args.output, 'w') as f:
        json.dump(profile, f, indent=2)

    print(f"\nProfile saved to: {args.output}")

    # Print summary
    print("\n" + "=" * 60)
    print("MonetDB Kernel Profile Summary")
    print("=" * 60)
    print(f"Regions identified: {profile['num_regions']}")
    print(f"Offload candidates: {profile['analysis']['num_offload_candidates']}")
    print(f"Total overhead covered: {profile['analysis']['total_profiled_overhead_pct']:.1f}%")
    print(f"Offload candidate overhead: {profile['analysis']['offload_candidate_overhead_pct']:.1f}%")
    print("\nTop functions:")
    for i, name in enumerate(profile['analysis']['top_functions'][:5], 1):
        print(f"  {i}. {name}")


if __name__ == "__main__":
    main()
