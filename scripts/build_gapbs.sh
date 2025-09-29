#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${REPO_ROOT}/bench/gapbs/src"
WORK_ROOT="${REPO_ROOT}/build/gapbs_pipeline"
BIN_ROOT="${REPO_ROOT}/bin/gapbs"

CLANGIR_BIN="${CLANGIR_BIN:-$(command -v clang++ || true)}"
CLANGIR_OPT_BIN="${CLANGIR_OPT_BIN:-$(command -v cir-opt || true)}"
CIRA_BIN="${CIRA_BIN:-${REPO_ROOT}/build/bin/cira}"
LLC_BIN="${LLC_BIN:-$(command -v llc || true)}"
LINKER_BIN="${LINKER_BIN:-$(command -v clang++ || true)}"
RUNTIME_LIB_DIR="${RUNTIME_LIB_DIR:-${REPO_ROOT}/build/lib}"

die() {
  echo "[gapbs] $*" >&2
  exit 1
}

info() {
  echo "[gapbs] $*"
}

[[ -d "${SRC_DIR}" ]] || die "Expected sources under ${SRC_DIR}; ensure the gapbs submodule is fetched"

mapfile -d '' SOURCES < <(find "${SRC_DIR}" -type f \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' \) -print0 | sort -z)
(( ${#SOURCES[@]} > 0 )) || die "No GAPBS source files found in ${SRC_DIR}"

[[ -n "${CLANGIR_BIN}" && -x "${CLANGIR_BIN}" ]] || die "clang++ with -fclangir support not found (set CLANGIR_BIN)"
[[ -n "${CLANGIR_OPT_BIN}" && -x "${CLANGIR_OPT_BIN}" ]] || die "cir-opt not found (set CLANGIR_OPT_BIN)"
[[ -x "${CIRA_BIN}" ]] || die "cira binary not found at ${CIRA_BIN}"
[[ -n "${LLC_BIN}" && -x "${LLC_BIN}" ]] || die "llc not found (set LLC_BIN)"
[[ -n "${LINKER_BIN}" && -x "${LINKER_BIN}" ]] || die "clang++ linker not found (set LINKER_BIN)"
[[ -d "${RUNTIME_LIB_DIR}" ]] || die "Cira runtime library directory missing (${RUNTIME_LIB_DIR})"

mkdir -p "${WORK_ROOT}" "${BIN_ROOT}"

CIR_DIR="${WORK_ROOT}/cir"
MLIR_DIR="${WORK_ROOT}/mlir"
CIRA_DIR="${WORK_ROOT}/cira"
LLVM_DIR="${WORK_ROOT}/llvm"
OBJ_DIR="${WORK_ROOT}/obj"

for dir in "${CIR_DIR}" "${MLIR_DIR}" "${CIRA_DIR}" "${LLVM_DIR}" "${OBJ_DIR}"; do
  mkdir -p "${dir}"
done

CXXFLAGS=(-O3 -std=c++17)
if [[ -n "${GAPBS_EXTRA_CXXFLAGS:-}" ]]; then
  read -r -a EXTRA <<< "${GAPBS_EXTRA_CXXFLAGS}"
  CXXFLAGS+=("${EXTRA[@]}")
fi

INCLUDE_FLAGS=("-I${SRC_DIR}" "-I${REPO_ROOT}/include" "-I${REPO_ROOT}/runtime/include")
if [[ -n "${GAPBS_EXTRA_INCLUDES:-}" ]]; then
  read -r -a EXTRA_INC <<< "${GAPBS_EXTRA_INCLUDES}"
  INCLUDE_FLAGS+=("${EXTRA_INC[@]}")
fi

LINK_FLAGS=("-L${RUNTIME_LIB_DIR}" "-Wl,-rpath,${RUNTIME_LIB_DIR}" -lcira_runtime)
if [[ -n "${GAPBS_EXTRA_LDFLAGS:-}" ]]; then
  read -r -a EXTRA_LD <<< "${GAPBS_EXTRA_LDFLAGS}"
  LINK_FLAGS+=("${EXTRA_LD[@]}")
fi

ARCHES=("x86_64-unknown-linux-gnu" "aarch64-unknown-linux-gnu")
LL_FILES=()

info "Translating GAPBS sources through ClangIR and Cira"
for src in "${SOURCES[@]}"; do
  rel="${src#${SRC_DIR}/}"
  stem="${rel%.*}"
  cir_path="${CIR_DIR}/${stem}.cir"
  mlir_path="${MLIR_DIR}/${stem}.mlir"
  cira_path="${CIRA_DIR}/${stem}.cira.mlir"
  llvm_path="${LLVM_DIR}/${stem}.ll"

  mkdir -p "$(dirname "${cir_path}")" "$(dirname "${mlir_path}")" "$(dirname "${cira_path}")" "$(dirname "${llvm_path}")"

  info "  > ${rel}"
  "${CLANGIR_BIN}" -fno-exceptions -fno-rtti -fclangir -emit-cir -S "${CXXFLAGS[@]}" "${INCLUDE_FLAGS[@]}" "${src}" -o "${cir_path}"
  "${CLANGIR_OPT_BIN}" -cir-mlir-scf-prepare -cir-to-mlir "${cir_path}" -o "${mlir_path}"
  "${CIRA_BIN}" "${mlir_path}" --cir-to-cira --rmem-search-remote -o "${cira_path}"
  "${CIRA_BIN}" "${cira_path}" --convert-cira-to-llvm-hetero --convert-func-to-llvm --reconcile-unrealized-casts -o "${llvm_path}"
  LL_FILES+=("${llvm_path}")
done

info "Lowering Cira output to target objects"
for arch in "${ARCHES[@]}"; do
  arch_dir="${OBJ_DIR}/${arch}"
  mkdir -p "${arch_dir}"
  for ll in "${LL_FILES[@]}"; do
    rel="${ll#${LLVM_DIR}/}"
    stem="${rel%.ll}"
    obj="${arch_dir}/${stem}.o"
    mkdir -p "$(dirname "${obj}")"
    "${LLC_BIN}" -filetype=obj -relocation-model=pic -mtriple="${arch}" "${ll}" -o "${obj}"
  done
done

info "Linking x86_64 binary"
X86_ARCH="x86_64-unknown-linux-gnu"
X86_OBJ_DIR="${OBJ_DIR}/${X86_ARCH}"
mapfile -t X86_OBJS < <(find "${X86_OBJ_DIR}" -type f -name '*.o' -print | sort)
(( ${#X86_OBJS[@]} > 0 )) || die "No x86_64 object files to link"

mkdir -p "${BIN_ROOT}/${X86_ARCH}"
"${LINKER_BIN}" -target "${X86_ARCH}" "${X86_OBJS[@]}" "${LINK_FLAGS[@]}" -o "${BIN_ROOT}/${X86_ARCH}/gapbs"

AARCH64_ARCH="aarch64-unknown-linux-gnu"
mkdir -p "${BIN_ROOT}/${AARCH64_ARCH}"
find "${OBJ_DIR}/${AARCH64_ARCH}" -type f -name '*.o' -exec cp {} "${BIN_ROOT}/${AARCH64_ARCH}" \;

info "Build completed"
info "  x86_64 binary: ${BIN_ROOT}/${X86_ARCH}/gapbs"
info "  aarch64 objects: ${BIN_ROOT}/${AARCH64_ARCH}"
info "Use clang++ --target=aarch64-unknown-linux-gnu to link the copied objects if a sysroot is available."
