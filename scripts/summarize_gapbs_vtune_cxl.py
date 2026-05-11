#!/usr/bin/env python3
"""Summarize paired GAPBS VTune runs for local DDR vs CXL placement."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


KERNEL_ORDER = ["bfs", "bc", "pr", "cc_sv", "pr_spmv", "sssp", "tc"]
SUMMARY_SUFFIX = "_summary.csv"
HOTSPOT_SUFFIX = "_hotspots.csv"


@dataclass
class Run:
    kernel: str
    label: str
    trial: int
    summary: dict[str, str]
    top_function: str = "n/a"


def normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def split_row(line: str) -> list[str]:
    if "\t" in line:
        return [field.strip().strip('"') for field in line.rstrip("\n").split("\t")]
    return [field.strip().strip('"') for field in next(csv.reader([line]))]


def parse_summary(path: Path) -> dict[str, str]:
    metrics: dict[str, str] = {}
    if not path.exists():
        return metrics
    for line in path.read_text(errors="ignore").splitlines():
        row = split_row(line)
        if len(row) < 2:
            continue
        key = normalize_key(row[0])
        if key:
            metrics[key] = row[1].strip()
    return metrics


def parse_top_function(path: Path) -> str:
    if not path.exists():
        return "n/a"
    lines = path.read_text(errors="ignore").splitlines()
    header: list[str] | None = None
    func_idx = -1
    time_idx = -1
    rows: list[tuple[float, str]] = []
    for line in lines:
        row = split_row(line)
        if len(row) < 2:
            continue
        if header is None:
            lowered = [normalize_key(col) for col in row]
            for idx, col in enumerate(lowered):
                if "function" in col and "call stack" not in col:
                    func_idx = idx
                if col == "cpu time" or col.endswith("cpu time"):
                    time_idx = idx
            if func_idx >= 0:
                header = row
            continue
        if func_idx >= len(row):
            continue
        score = parse_number(row[time_idx]) if 0 <= time_idx < len(row) else 0.0
        rows.append((score, row[func_idx].strip()))
    if not rows:
        return "n/a"
    rows.sort(reverse=True)
    return rows[0][1] or "n/a"


def parse_number(text: str) -> float | None:
    if text is None:
        return None
    cleaned = text.strip().replace(",", "")
    cleaned = cleaned.replace("%", "")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    return float(match.group(0))


def metric(summary: dict[str, str], names: list[str]) -> float | None:
    for name in names:
        key = normalize_key(name)
        if key in summary:
            return parse_number(summary[key])
    for key, value in summary.items():
        if any(normalize_key(name) in key for name in names):
            parsed = parse_number(value)
            if parsed is not None:
                return parsed
    return None


def fmt(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 100:
        return f"{value:.0f}{suffix}"
    return f"{value:.1f}{suffix}"


def tex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def discover_runs(result_dir: Path) -> list[Run]:
    runs: list[Run] = []
    pattern = str(result_dir / f"*{SUMMARY_SUFFIX}")
    regex = re.compile(r"(?P<kernel>.+)_(?P<label>noncxl|cxl)_t(?P<trial>\d+)_")
    for summary_path in glob.glob(pattern):
        path = Path(summary_path)
        match = regex.search(path.name)
        if not match:
            continue
        kernel = match.group("kernel")
        label = match.group("label")
        trial = int(match.group("trial"))
        hotspot_path = Path(str(path).replace(SUMMARY_SUFFIX, HOTSPOT_SUFFIX))
        runs.append(
            Run(
                kernel=kernel,
                label=label,
                trial=trial,
                summary=parse_summary(path),
                top_function=parse_top_function(hotspot_path),
            )
        )
    return runs


def grouped_metrics(runs: list[Run]) -> list[dict[str, object]]:
    by_key: dict[tuple[str, str], list[Run]] = {}
    for run in runs:
        by_key.setdefault((run.kernel, run.label), []).append(run)

    kernels = sorted({run.kernel for run in runs}, key=lambda k: (KERNEL_ORDER.index(k) if k in KERNEL_ORDER else 99, k))
    rows: list[dict[str, object]] = []
    for kernel in kernels:
        non = by_key.get((kernel, "noncxl"), [])
        cxl = by_key.get((kernel, "cxl"), [])
        non_elapsed = average_metric(non, ["elapsed time", "execution time", "total time"])
        cxl_elapsed = average_metric(cxl, ["elapsed time", "execution time", "total time"])
        non_mem = average_metric(non, ["memory bound"])
        cxl_mem = average_metric(cxl, ["memory bound"])
        non_dram = average_metric(non, ["dram bound"])
        cxl_dram = average_metric(cxl, ["dram bound"])
        slowdown = None
        if non_elapsed and cxl_elapsed:
            slowdown = (cxl_elapsed / non_elapsed - 1.0) * 100.0
        rows.append(
            {
                "kernel": kernel,
                "non_elapsed": non_elapsed,
                "cxl_elapsed": cxl_elapsed,
                "slowdown_pct": slowdown,
                "non_mem_bound": non_mem,
                "cxl_mem_bound": cxl_mem,
                "non_dram_bound": non_dram,
                "cxl_dram_bound": cxl_dram,
                "non_top": non[0].top_function if non else "n/a",
                "cxl_top": cxl[0].top_function if cxl else "n/a",
            }
        )
    return rows


def average_metric(runs: list[Run], names: list[str]) -> float | None:
    values = [metric(run.summary, names) for run in runs]
    present = [value for value in values if value is not None]
    return mean(present) if present else None


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "kernel",
                "non_elapsed",
                "cxl_elapsed",
                "slowdown_pct",
                "non_mem_bound",
                "cxl_mem_bound",
                "non_dram_bound",
                "cxl_dram_bound",
                "non_top",
                "cxl_top",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_tex(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [
        "% Generated by scripts/summarize_gapbs_vtune_cxl.py.",
        "\\begin{table}[t]",
        "\\caption{VTune CXL vs. non-CXL comparison for GAPBS. Non-CXL binds data to a local DDR NUMA node; CXL binds the same workload to the CXL memory node while keeping CPU placement fixed.}",
        "\\label{tab:gapbs_vtune_cxl}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{@{}lrrrrl@{}}",
        "\\toprule",
        "\\textbf{Kernel} & \\textbf{Non-CXL (s)} & \\textbf{CXL (s)} & \\textbf{Slowdown} & \\textbf{Mem Bound $\\Delta$} & \\textbf{Top CXL Hotspot} \\\\ \\midrule",
    ]
    for row in rows:
        non_mem = row["non_mem_bound"]
        cxl_mem = row["cxl_mem_bound"]
        mem_delta = None
        if isinstance(non_mem, float) and isinstance(cxl_mem, float):
            mem_delta = cxl_mem - non_mem
        body.append(
            f"{tex_escape(str(row['kernel']).upper())} & "
            f"{fmt(row['non_elapsed'])} & "
            f"{fmt(row['cxl_elapsed'])} & "
            f"{fmt(row['slowdown_pct'], '\\\\%')} & "
            f"{fmt(mem_delta, ' pp')} & "
            f"\\texttt{{{tex_escape(str(row['cxl_top']))}}} \\\\"
        )
    body.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=Path("profile_results/vtune_gapbs_cxl"))
    parser.add_argument("--csv-out", type=Path, default=Path("profile_results/vtune_gapbs_cxl/gapbs_vtune_cxl_summary.csv"))
    parser.add_argument("--tex-out", type=Path)
    args = parser.parse_args()

    runs = discover_runs(args.result_dir)
    rows = grouped_metrics(runs)
    write_csv(rows, args.csv_out)
    if args.tex_out:
        write_tex(rows, args.tex_out)
    print(f"wrote {len(rows)} GAPBS VTune comparison rows")


if __name__ == "__main__":
    main()
