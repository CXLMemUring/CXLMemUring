#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
units_file="${CIRA_MONETDB_UNITS_FILE:-${repo_root}/scripts/monetdb_cira_units.txt}"
contract="${repo_root}/build/monetdb_pipeline/build-contract.json"
mode=""

while (( $# )); do
  case "$1" in
    --contract-only|--check-units|--server-only)
      [[ -z "${mode}" ]] || { echo "only one build mode may be selected" >&2; exit 2; }
      mode="$1"
      shift
      ;;
    --contract)
      [[ $# -ge 2 ]] || { echo "--contract requires a path" >&2; exit 2; }
      contract="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

[[ -n "${mode}" ]] || mode="--server-only"
[[ -f "${units_file}" ]] || { echo "missing targeted-unit list: ${units_file}" >&2; exit 1; }

write_contract() {
  mkdir -p "$(dirname "${contract}")"
  python3 - "${units_file}" "${contract}" "${repo_root}" <<'PY'
import json
import pathlib
import sys

units_path, contract_path, root = map(pathlib.Path, sys.argv[1:])
passes = [
    "cir-mlir-scf-prepare",
    "cir-to-mlir",
    "cir-to-cira",
    "cira-region-formation",
    "cira-outline-rv64-kernels",
    "convert-cira-to-llvm-hetero",
    "convert-scf-to-cf",
    "cira-prune-unreachable-blocks",
    "convert-cf-to-llvm",
    "convert-to-llvm",
    "cse",
    "convert-ub-to-llvm",
    "reconcile-unrealized-casts",
    "cira-finalize-llvm-materializations",
]
records = []
for line in units_path.read_text(encoding="utf-8").splitlines():
    if not line.strip() or line.lstrip().startswith("#"):
        continue
    source, queries, role = line.split()
    stem = pathlib.Path(source).with_suffix("")
    records.append({
        "source": source,
        "queries": queries.split(","),
        "role": role,
        "mode": "cira-hetero",
        "fallback_allowed": False,
        "compiler": str(root.parent / "clangir/build/bin/clang"),
        "passes": passes,
        "object": str(root / "build/monetdb_pipeline/obj" / stem) + ".o",
        "kernel_ids": [],
    })
contract_path.write_text(json.dumps({"schema_version": 1, "units": records}, indent=2) + "\n", encoding="utf-8")
PY
}

write_contract
if [[ "${mode}" == "--contract-only" ]]; then
  exit 0
fi

if [[ "${mode}" == "--check-units" ]]; then
  exec python3 "${repo_root}/scripts/monetdb_cira_compile.py" \
    --check-units --units "${units_file}" --contract "${contract}"
fi

exec python3 "${repo_root}/scripts/monetdb_cira_compile.py" \
  --build-server --units "${units_file}" --contract "${contract}"
