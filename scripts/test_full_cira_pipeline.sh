#!/usr/bin/env bash
# Verify the source-level CIRA path without accepting a direct/marker fallback.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLANGIR_ROOT="${CLANGIR_ROOT:-${REPO_ROOT}/../clangir}"
CLANGIR_BIN="${CLANGIR_BIN:-${CLANGIR_ROOT}/build/bin/clang}"
CIR_OPT_BIN="${CIR_OPT_BIN:-${CLANGIR_ROOT}/build/bin/cir-opt}"
CIRA_BIN="${CIRA_BIN:-${REPO_ROOT}/build/bin/cira}"
CIRA_REQUIRE_JIT="${CIRA_REQUIRE_JIT:-1}"

for tool in "${CLANGIR_BIN}" "${CIR_OPT_BIN}" "${CIRA_BIN}"; do
  [[ -x "${tool}" ]] || {
    echo "missing required tool: ${tool}" >&2
    exit 1
  }
done

work_dir="$(mktemp -d "${TMPDIR:-/tmp}/cira-full-pipeline.XXXXXX")"
trap 'rm -rf "${work_dir}"' EXIT

source_file="${REPO_ROOT}/test/cir/remote_gather.c"
cir_file="${work_dir}/remote_gather.cir"
mlir_file="${work_dir}/remote_gather.mlir"
regions_file="${work_dir}/remote_gather.regions.mlir"
host_file="${work_dir}/remote_gather.host.mlir"

"${CLANGIR_BIN}" -std=c11 -fclangir -emit-cir -S "${source_file}" -o "${cir_file}"
"${CIR_OPT_BIN}" -allow-unregistered-dialect -cir-mlir-scf-prepare -cir-to-mlir \
  "${cir_file}" -o "${mlir_file}"
"${CIRA_BIN}" -allow-unregistered-dialect "${mlir_file}" \
  --cira-region-formation="require-profitable=false" -o "${regions_file}"
rg -q 'cira\.offload' "${regions_file}" || {
  echo "no CIRA region was formed from the source input" >&2
  exit 1
}

"${CIRA_BIN}" -allow-unregistered-dialect "${regions_file}" \
  --convert-cira-to-llvm-hetero -o "${host_file}"
rg -q 'func\.func @__vortex_kernel_' "${host_file}" || {
  echo "heterogeneous lowering did not extract a device kernel" >&2
  exit 1
}
rg -q 'llvm\.call @cira_get_device_func_addr' "${host_file}" || {
  echo "host lowering did not emit an offload submission target lookup" >&2
  exit 1
}

if rg -q 'cira_jit_(dispatch|specialize|invoke)' "${host_file}"; then
  echo "full source -> CIRA -> hetero -> runtime-JIT path verified"
elif [[ "${CIRA_REQUIRE_JIT}" == "1" ]]; then
  echo "compiler formed and lowered a CIRA region, but emitted no runtime JIT dispatch" >&2
  exit 2
else
  echo "source -> CIRA -> hetero path verified; JIT dispatch is intentionally not required"
fi
