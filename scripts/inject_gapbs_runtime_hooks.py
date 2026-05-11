#!/usr/bin/env python3
"""Inject CIRA runtime markers into optimized GAPBS LLVM IR.

The normal GAPBS build uses the direct ClangIR-to-LLVM path for C++ support.
That path can still carry concrete runtime calls for the compiler/gem5 backend:
this script inserts a marker at known GAPBS kernel function entries after LLVM
optimization and before llc lowers to an object file.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


HOOKS_BY_FILE = {
    "bc": [
        ("4PBFS", 0, 0),
        ("7Brandes", 0, 1),
    ],
    "sssp": [
        ("9DeltaStep", 1, 0),
        ("10RelaxEdges", 1, 1),
    ],
    "pr": [
        ("14PageRankPullGS", 2, 0),
    ],
}

DECLARE = "declare void @cira_gapbs_region_marker(i32, i32, ptr, i64)"
DEFINE_RE = re.compile(
    r"^(?P<header>define\b.*\s@(?P<name>(?:\"[^\"]+\"|[^\s(]+))\s*"
    r"\((?P<args>.*)\)\s*(?:#[0-9]+)?\s*\{)\s*$"
)


def source_stem(path: Path) -> str:
    name = path.name
    for suffix in (".opt.ll", ".ll"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def strip_quotes(name: str) -> str:
    if len(name) >= 2 and name[0] == '"' and name[-1] == '"':
        return name[1:-1]
    return name


def hook_for_function(stem: str, func_name: str):
    for needle, benchmark_id, region_id in HOOKS_BY_FILE.get(stem, []):
        if needle in func_name:
            return benchmark_id, region_id
    return None


def first_graph_arg(args: str) -> str:
    # GAPBS kernels all carry the graph reference as their first lowered
    # pointer argument. Keep a null fallback for nonstandard IR spelling.
    first = args.split(",", 1)[0].strip()
    match = re.match(r"ptr(?:\s+[^%\s]+)*\s+(%\w+)$", first)
    if match:
        return f"ptr {match.group(1)}"
    return "ptr null"


def inject(text: str, stem: str) -> tuple[str, int]:
    lines = text.splitlines()
    out: list[str] = []
    injected = 0
    seen_declare = DECLARE in text

    for line in lines:
        out.append(line)
        match = DEFINE_RE.match(line)
        if not match:
            continue

        func_name = strip_quotes(match.group("name"))
        hook = hook_for_function(stem, func_name)
        if not hook:
            continue

        benchmark_id, region_id = hook
        graph_arg = first_graph_arg(match.group("args"))
        out.append(
            "  call void @cira_gapbs_region_marker("
            f"i32 {benchmark_id}, i32 {region_id}, {graph_arg}, i64 0)"
        )
        injected += 1

    if injected and not seen_declare:
        out.append("")
        out.append(DECLARE)

    return "\n".join(out) + ("\n" if text.endswith("\n") else ""), injected


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        print(
            "usage: inject_gapbs_runtime_hooks.py <input.ll> [output.ll]",
            file=sys.stderr,
        )
        return 2

    input_path = Path(argv[1])
    output_path = Path(argv[2]) if len(argv) == 3 else input_path
    stem = source_stem(input_path)

    text = input_path.read_text()
    new_text, injected = inject(text, stem)
    output_path.write_text(new_text)
    print(f"[gapbs-hooks] {input_path}: injected {injected} CIRA marker(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
