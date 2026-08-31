#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
units_file="${repo_root}/scripts/monetdb_cira_units.txt"
builder="${repo_root}/scripts/build_monetdb_tpch_cira.sh"
smoke="${repo_root}/scripts/smoke_mserver5.py"

expected=(
  "gdk/gdk_select.c q6 scan"
  "gdk/gdk_aggr.c q1 aggregation"
  "gdk/gdk_join.c q9,q11 join"
  "gdk/gdk_cand.c q11 candidate"
  "gdk/gdk_string.c q13 grouping-traversal"
  "monetdb5/modules/mal/pcre.c q13 string-group"
)

mapfile -t actual < <(grep -Ev '^[[:space:]]*(#|$)' "${units_file}")
[[ "${#actual[@]}" -eq "${#expected[@]}" ]]
for index in "${!expected[@]}"; do
  [[ "${actual[index]}" == "${expected[index]}" ]]
done

[[ -x "${builder}" ]]
[[ -x "${smoke}" ]]
contract="$(mktemp)"
trap 'rm -f "${contract}"' EXIT
"${builder}" --contract-only --contract "${contract}"

python3 - "${contract}" "${units_file}" <<'PY'
import json
import pathlib
import sys

contract_path, units_path = map(pathlib.Path, sys.argv[1:])
contract = json.loads(contract_path.read_text(encoding="utf-8"))
expected = [
    line.split()[0]
    for line in units_path.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]
records = contract.get("units")
assert isinstance(records, list), "build contract has no unit records"
assert [record.get("source") for record in records] == expected, "contract source order differs from target list"
required = {"cir-mlir-scf-prepare", "cir-to-mlir", "cir-to-cira", "cira-region-formation",
            "cira-outline-rv64-kernels", "convert-cira-to-llvm-hetero",
            "cira-prune-unreachable-blocks", "convert-to-llvm", "convert-ub-to-llvm",
            "cira-finalize-llvm-materializations"}
for record in records:
    source = record["source"]
    assert record.get("mode") == "cira-hetero", f"{source} is not fail-closed cira-hetero"
    assert record.get("fallback_allowed") is False, f"{source} permits native fallback"
    passes = set(record.get("passes", []))
    missing = required - passes
    assert not missing, f"{source} omits passes: {sorted(missing)}"
    assert record.get("object"), f"{source} has no deterministic object path"
    assert isinstance(record.get("kernel_ids"), list), f"{source} has no kernel-ID field"
PY

if [[ "${CIRA_RUN_MONETDB_TRANSLATION_CHECK:-0}" == "1" ]]; then
  "${builder}" --check-units --contract "${contract}"
fi

echo "MonetDB CIRA build contract checks passed"
