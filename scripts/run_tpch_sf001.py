#!/usr/bin/env python3
"""Prepare and run the checked-in MonetDB TPC-H SF=0.01 workload."""

import argparse
import os
import pathlib
import subprocess


EXPECTED_COUNTS = {
    "region": 5,
    "nation": 25,
    "supplier": 100,
    "customer": 1500,
    "part": 2000,
    "partsupp": 8000,
    "orders": 15000,
    "lineitem": 60175,
}


def has_top_level_order_by(sql: str) -> bool:
    words = []
    token = []
    depth = 0
    quote = None
    index = 0
    while index < len(sql):
        char = sql[index]
        if quote:
            if char == quote:
                if index + 1 < len(sql) and sql[index + 1] == quote:
                    index += 2
                    continue
                quote = None
            index += 1
            continue
        if char in "'\"":
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif char.isalnum() or char == "_":
            token.append(char.lower())
        else:
            if token and depth == 0:
                words.append("".join(token))
            token.clear()
        index += 1
    if token and depth == 0:
        words.append("".join(token))
    return any(left == "order" and right == "by" for left, right in zip(words, words[1:]))


def extract_statement(text: str, start_marker: str, end_marker: str) -> str:
    start = text.find(start_marker)
    if start < 0:
        raise ValueError(f"missing TPC-H marker: {start_marker!r}")
    start += len(start_marker)
    end = text.find(end_marker, start)
    if end < 0:
        raise ValueError(f"missing TPC-H marker after {start_marker!r}: {end_marker!r}")
    return text[start:end].strip() + "\n"


def extract_queries(source: pathlib.Path, output: pathlib.Path) -> None:
    text = source.read_text(encoding="utf-8")
    output.mkdir(parents=True, exist_ok=True)
    for query_id in range(1, 23):
        marker = f"#Q{query_id}\n"
        start = text.find(marker)
        if start < 0:
            raise ValueError(f"missing {marker.strip()} in {source}")
        query_header = text.find("query ", start + len(marker))
        query_start = text.find("\n", query_header) + 1
        query_end = text.find("\n----", query_start)
        if query_header < 0 or query_start == 0 or query_end < 0:
            raise ValueError(f"malformed {marker.strip()} in {source}")
        sql = text[query_start:query_end].strip() + "\n"
        stem = f"q{query_id:02d}"
        (output / f"{stem}.sql").write_text(sql, encoding="utf-8")
        (output / f"{stem}.ordered").write_text(
            "1\n" if has_top_level_order_by(sql) else "0\n", encoding="ascii"
        )

    setup = extract_statement(
        text, "statement ok\ncreate view revenue0", "\n\n#Q15"
    )
    cleanup = extract_statement(
        text, "statement ok\ndrop view revenue0", "\n\n#Q16"
    )
    (output / "q15-setup.sql").write_text("create view revenue0" + setup, encoding="utf-8")
    (output / "q15-cleanup.sql").write_text("drop view revenue0" + cleanup, encoding="utf-8")


def validate_data_counts(data: pathlib.Path) -> dict[str, int]:
    actual = {}
    for table, expected in EXPECTED_COUNTS.items():
        path = data / f"{table}.tbl"
        if not path.is_file():
            raise ValueError(f"missing TPC-H data file: {path}")
        with path.open("rb") as stream:
            count = sum(1 for _ in stream)
        if count != expected:
            raise ValueError(f"{path}: expected {expected} rows, found {count}")
        actual[table] = count
    return actual


def main(argv=None) -> int:
    repo = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "forced-debug"), required=True)
    parser.add_argument("--query", default="all")
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument(
        "--runner", "--binary", dest="runner", type=pathlib.Path,
        default=repo / "bin/monetdb/x86_64-unknown-linux-gnu/monetdb-tpch-cira",
    )
    parser.add_argument(
        "--data", type=pathlib.Path,
        default=repo / "bench/MonetDB/sql/benchmarks/tpch/SF-0.01",
    )
    args = parser.parse_args(argv)

    data = args.data.resolve()
    validate_data_counts(data)
    queries = repo / "build/tpch-sf001-queries"
    extract_queries(
        repo / "bench/MonetDB/sql/benchmarks/tpch/Tests/one.test.in", queries
    )
    runner = args.runner.resolve()
    if not runner.is_file() or not os.access(runner, os.X_OK):
        raise ValueError(f"TPC-H runner is not executable: {runner}")
    args.output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    if args.mode == "forced-debug":
        environment.setdefault("CIRA_CXL_ARENA_SIZE", str(512 * 1024 * 1024))
    command = [
        str(runner), "--data", str(data), "--queries", str(queries),
        "--schema", str(repo / "bench/tpch_runner/schema.sql"),
        "--output", str(args.output.resolve()), "--mode", args.mode,
        "--query", args.query,
    ]
    return subprocess.run(command, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
