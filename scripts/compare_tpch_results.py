#!/usr/bin/env python3
"""Compare canonical typed TPC-H result artifacts without numeric tolerance."""

import argparse
import csv
import json
import pathlib
import sys


def load_result(directory: pathlib.Path, stem: str):
    metadata_path = directory / f"{stem}.json"
    values_path = directory / f"{stem}.tsv"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required = {"schema_version", "query_id", "ordered", "row_count", "columns"}
    missing = sorted(required - metadata.keys())
    if missing:
        raise ValueError(f"{directory}/{stem}: missing metadata fields {missing}")
    if not isinstance(metadata["ordered"], bool):
        raise ValueError(f"{directory}/{stem}: ordered must be a boolean")
    if not isinstance(metadata["row_count"], int) or metadata["row_count"] < 0:
        raise ValueError(f"{directory}/{stem}: row_count must be a nonnegative integer")
    if not isinstance(metadata["columns"], list):
        raise ValueError(f"{directory}/{stem}: columns must be a list")
    with values_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream, delimiter="\t"))
    if metadata.get("row_count") != len(rows):
        raise ValueError(
            f"{directory}/{stem}: metadata row_count={metadata.get('row_count')} "
            f"but TSV contains {len(rows)} rows"
        )
    expected_columns = len(metadata.get("columns", []))
    for row_index, row in enumerate(rows, start=1):
        if len(row) != expected_columns:
            raise ValueError(
                f"{directory}/{stem}: row {row_index} has {len(row)} fields, "
                f"expected {expected_columns}"
            )
    return metadata, rows


def comparable_metadata(metadata):
    return {
        "schema_version": metadata.get("schema_version"),
        "query_id": metadata.get("query_id"),
        "ordered": metadata.get("ordered"),
        "row_count": metadata.get("row_count"),
        "columns": metadata.get("columns"),
    }


def compare_query(baseline_dir: pathlib.Path, candidate_dir: pathlib.Path, stem: str):
    baseline_meta, baseline_rows = load_result(baseline_dir, stem)
    candidate_meta, candidate_rows = load_result(candidate_dir, stem)
    if comparable_metadata(baseline_meta) != comparable_metadata(candidate_meta):
        return False, "metadata mismatch"

    if not baseline_meta["ordered"]:
        baseline_rows.sort()
        candidate_rows.sort()
    if baseline_rows != candidate_rows:
        mismatch = next(
            (
                index
                for index, pair in enumerate(zip(baseline_rows, candidate_rows), start=1)
                if pair[0] != pair[1]
            ),
            min(len(baseline_rows), len(candidate_rows)) + 1,
        )
        return False, f"value mismatch at canonical row {mismatch}"
    return True, ""


def result_stems(directory: pathlib.Path):
    return {path.stem for path in directory.glob("q[0-9][0-9].json")}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=pathlib.Path)
    parser.add_argument("candidate", type=pathlib.Path)
    args = parser.parse_args(argv)

    baseline_stems = result_stems(args.baseline)
    candidate_stems = result_stems(args.candidate)
    if not baseline_stems:
        parser.error(f"no qNN.json results in {args.baseline}")
    if baseline_stems != candidate_stems:
        missing = sorted(baseline_stems - candidate_stems)
        extra = sorted(candidate_stems - baseline_stems)
        print(f"result set mismatch: missing={missing} extra={extra}", file=sys.stderr)
        return 1

    failed = False
    for stem in sorted(baseline_stems):
        label = stem.upper()
        try:
            matches, reason = compare_query(args.baseline, args.candidate, stem)
        except (OSError, ValueError, json.JSONDecodeError, csv.Error) as error:
            matches, reason = False, str(error)
        if matches:
            print(f"{label} PASS")
        else:
            failed = True
            print(f"{label} FAIL: {reason}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
