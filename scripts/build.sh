#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CLANGIR_BIN="${CLANGIR_BIN:-$(command -v clang++ || true)}"
CLANGIR_OPT_BIN="${CLANGIR_OPT_BIN:-$(command -v cir-opt || true)}"
CIRA_BIN="${CIRA_BIN:-${REPO_ROOT}/build/bin/cira}"
MLIR_TRANSLATE_BIN="${MLIR_TRANSLATE_BIN:-$(command -v mlir-translate || true)}"
LLC_BIN="${LLC_BIN:-$(command -v llc || true)}"
LINKER_BIN="${LINKER_BIN:-$(command -v clang++ || true)}"
RUNTIME_LIB_DIR="${RUNTIME_LIB_DIR:-${REPO_ROOT}/build/lib}"

LOG_PREFIX="build"

die() {
  echo "[${LOG_PREFIX}] $*" >&2
  exit 1
}

info() {
  echo "[${LOG_PREFIX}] $*"
}

add_source() {
  local src="$1"
  [[ -n "${src}" ]] || return 0
  if [[ -n "${SOURCES_SEEN["${src}"]:-}" ]]; then
    return 0
  fi
  SOURCES_SEEN["${src}"]=1
  SOURCES+=("${src}")
}

should_exclude() {
  local path="$1"
  for pattern in "${EXCLUDE_PATTERNS[@]:-}"; do
    [[ -z "${pattern}" ]] && continue
    if [[ "${path}" == *"${pattern}"* ]]; then
      return 0
    fi
  done
  return 1
}

append_flags_from_env() {
  local env_name="$1"
  local -n target_ref="$2"
  local env_value="${!env_name:-}"
  [[ -z "${env_value}" ]] && return 0
  local -a parsed=()
  read -r -a parsed <<< "${env_value}"
  target_ref+=("${parsed[@]}")
}

add_include_flag() {
  local include_dir="$1"
  [[ -n "${include_dir}" ]] || return 0
  local flag="-I${include_dir}"
  if [[ -n "${INCLUDE_FLAG_SEEN["${flag}"]:-}" ]]; then
    return 0
  fi
  INCLUDE_FLAG_SEEN["${flag}"]=1
  INCLUDE_FLAGS+=("${flag}")
}

append_includes_from_env() {
  local env_name="$1"
  local env_value="${!env_name:-}"
  [[ -z "${env_value}" ]] && return 0
  local -a parsed=()
  read -r -a parsed <<< "${env_value}"
  for inc in "${parsed[@]}"; do
    add_include_flag "${inc}"
  done
}

if (( $# > 1 )); then
  die "Usage: $(basename "$0") [gapbs|mcf|llama.cpp|monetdb|dataframe]"
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<EOF
Usage: $(basename "$0") [gapbs|mcf|llama.cpp|monetdb|dataframe]

Builds the requested benchmark (default: gapbs) through the ClangIR -> Cira pipeline.
Aliases:
  llama.cpp -> llama, llama_cpp
  dataframe -> df
EOF
  exit 0
fi

BENCHMARK="${1:-gapbs}"
BENCHMARK_KEY="${BENCHMARK,,}"
LOG_PREFIX="${BENCHMARK}"

declare BENCHMARK_ID=""
declare BIN_NAME=""
declare WORK_ROOT=""
declare BIN_ROOT=""
declare -a SOURCE_DIRS=()
declare -a EXTRA_SOURCES=()
declare -a EXCLUDE_PATTERNS=()
declare -a BENCHMARK_EXTRA_CFLAGS=()
declare -a BENCHMARK_EXTRA_CXXFLAGS=()
declare -a BENCHMARK_EXTRA_INCLUDES=()
declare -a BENCHMARK_EXTRA_LDFLAGS=()
SKIP_LINK=false

case "${BENCHMARK_KEY}" in
  gapbs)
    BENCHMARK_ID="GAPBS"
    SOURCE_DIRS=("${REPO_ROOT}/bench/gapbs/src")
    WORK_ROOT="${REPO_ROOT}/build/gapbs_pipeline"
    BIN_ROOT="${REPO_ROOT}/bin/gapbs"
    BIN_NAME="gapbs"
    ;;
  mcf)
    BENCHMARK_ID="MCF"
    SOURCE_DIRS=("${REPO_ROOT}/bench/mcf")
    WORK_ROOT="${REPO_ROOT}/build/mcf_pipeline"
    BIN_ROOT="${REPO_ROOT}/bin/mcf"
    BIN_NAME="mcf"
    ;;
  llama.cpp|llama|llama_cpp)
    BENCHMARK_ID="LLAMACPP"
    SOURCE_DIRS=(
      "${REPO_ROOT}/bench/llama.cpp/src"
      "${REPO_ROOT}/bench/llama.cpp/common"
      "${REPO_ROOT}/bench/llama.cpp/ggml/src"
      "${REPO_ROOT}/bench/llama.cpp/ggml/src/ggml-cpu"
    )
    EXCLUDE_PATTERNS=(
      "/ggml/src/ggml-blas/"
      "/ggml/src/ggml-cann/"
      "/ggml/src/ggml-cuda/"
      "/ggml/src/ggml-hip/"
      "/ggml/src/ggml-metal/"
      "/ggml/src/ggml-musa/"
      "/ggml/src/ggml-opencl/"
      "/ggml/src/ggml-rpc/"
      "/ggml/src/ggml-sycl/"
      "/ggml/src/ggml-vulkan/"
      "/ggml/src/ggml-webgpu/"
      "/ggml/src/ggml-zdnn/"
      "/ggml/src/ggml-vulkan/"
    )
    EXTRA_SOURCES=("${REPO_ROOT}/bench/llama.cpp/examples/simple/simple.cpp")
    BENCHMARK_EXTRA_INCLUDES=(
      "${REPO_ROOT}/bench/llama.cpp"
      "${REPO_ROOT}/bench/llama.cpp/include"
      "${REPO_ROOT}/bench/llama.cpp/common"
      "${REPO_ROOT}/bench/llama.cpp/ggml/include"
      "${REPO_ROOT}/bench/llama.cpp/ggml/src"
    )
    BENCHMARK_EXTRA_CFLAGS+=(-D_GNU_SOURCE)
    BENCHMARK_EXTRA_CXXFLAGS+=(-fexceptions -frtti -D_GNU_SOURCE)
    BENCHMARK_EXTRA_LDFLAGS=(-lpthread -ldl -lm)
    WORK_ROOT="${REPO_ROOT}/build/llama_pipeline"
    BIN_ROOT="${REPO_ROOT}/bin/llama.cpp"
    BIN_NAME="llama-simple"
    ;;
  monetdb)
    BENCHMARK_ID="MONETDB"
    SOURCE_DIRS=(
      "${REPO_ROOT}/bench/MonetDB/common"
      "${REPO_ROOT}/bench/MonetDB/gdk"
      "${REPO_ROOT}/bench/MonetDB/monetdb5"
      "${REPO_ROOT}/bench/MonetDB/sql/backends"
      "${REPO_ROOT}/bench/MonetDB/sql/common"
      "${REPO_ROOT}/bench/MonetDB/sql/server"
      "${REPO_ROOT}/bench/MonetDB/sql/storage"
    )
    EXCLUDE_PATTERNS=(
      "/sql/benchmarks/"
      "/sql/test/"
      "/sql/scripts/"
      "/sql/jdbc/"
      "/sql/odbc/"
      "/sql/NT/"
      "/monetdb5/Tests/"
      "/common/Tests/"
      "/gdk/Tests/"
      "/testing/"
      "/documentation/"
      "/clients/"
      "/geom/"
      "/tools/merovingian/"
      "/tools/monetdbe/"
      "/ctest/"
    )
    EXTRA_SOURCES=("${REPO_ROOT}/bench/MonetDB/tools/mserver/mserver5.c")
    BENCHMARK_EXTRA_INCLUDES=(
      "${REPO_ROOT}/bench/MonetDB/common"
      "${REPO_ROOT}/bench/MonetDB/gdk"
      "${REPO_ROOT}/bench/MonetDB/monetdb5"
      "${REPO_ROOT}/bench/MonetDB/sql/include"
      "${REPO_ROOT}/bench/MonetDB/sql/common"
      "${REPO_ROOT}/bench/MonetDB/sql/server"
      "${REPO_ROOT}/bench/MonetDB/sql/storage"
      "${REPO_ROOT}/bench/MonetDB/sql/backends/monet5"
      "${REPO_ROOT}/bench/MonetDB/tools/mserver"
    )
    BENCHMARK_EXTRA_CFLAGS+=(-D_GNU_SOURCE)
    BENCHMARK_EXTRA_LDFLAGS=(-lpthread -ldl -lm)
    WORK_ROOT="${REPO_ROOT}/build/monetdb_pipeline"
    BIN_ROOT="${REPO_ROOT}/bin/monetdb"
    BIN_NAME="mserver5"
    ;;
  dataframe|df)
    BENCHMARK_ID="DATAFRAME"
    SOURCE_DIRS=("${REPO_ROOT}/bench/DataFrame/src")
    EXTRA_SOURCES=("${REPO_ROOT}/bench/DataFrame/benchmarks/dataframe_performance.cc")
    BENCHMARK_EXTRA_INCLUDES=("${REPO_ROOT}/bench/DataFrame/include")
    BENCHMARK_EXTRA_CXXFLAGS+=(-fexceptions -frtti)
    BENCHMARK_EXTRA_LDFLAGS=(-lpthread)
    if [[ "$(uname -s 2>/dev/null)" == "Linux" ]]; then
      BENCHMARK_EXTRA_LDFLAGS+=(-lrt)
    fi
    WORK_ROOT="${REPO_ROOT}/build/dataframe_pipeline"
    BIN_ROOT="${REPO_ROOT}/bin/dataframe"
    BIN_NAME="dataframe_benchmark"
    ;;
  *)
    die "Unsupported benchmark '${BENCHMARK}'. Expected one of gapbs, mcf, llama.cpp, monetdb, dataframe."
    ;;
esac

for dir in "${SOURCE_DIRS[@]}"; do
  [[ -d "${dir}" ]] || die "Expected sources under ${dir}; ensure the ${BENCHMARK} sources are available"
done

declare -a SOURCES=()
declare -A SOURCES_SEEN=()

collect_sources() {
  local search_dir="$1"
  while IFS= read -r -d '' candidate; do
    if should_exclude "${candidate}"; then
      continue
    fi
    add_source "${candidate}"
  done < <(find "${search_dir}" -type f \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' \) -print0 | sort -z)
}

for dir in "${SOURCE_DIRS[@]}"; do
  collect_sources "${dir}"
done

for extra in "${EXTRA_SOURCES[@]}"; do
  if [[ ! -f "${extra}" ]]; then
    die "Expected additional source '${extra}' for ${BENCHMARK} benchmark"
  fi
  add_source "${extra}"
done

(( ${#SOURCES[@]} > 0 )) || die "No ${BENCHMARK} source files discovered"

[[ -n "${CLANGIR_BIN}" && -x "${CLANGIR_BIN}" ]] || die "clang++ with -fclangir support not found (set CLANGIR_BIN)"
[[ -n "${CLANGIR_OPT_BIN}" && -x "${CLANGIR_OPT_BIN}" ]] || die "cir-opt not found (set CLANGIR_OPT_BIN)"
[[ -x "${CIRA_BIN}" ]] || die "cira binary not found at ${CIRA_BIN}"
[[ -n "${MLIR_TRANSLATE_BIN}" && -x "${MLIR_TRANSLATE_BIN}" ]] || die "mlir-translate not found (set MLIR_TRANSLATE_BIN)"
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

COMMON_FLAGS=(-O3)

CFLAGS=(-std=c11)
CFLAGS+=("${BENCHMARK_EXTRA_CFLAGS[@]}")
append_flags_from_env PIPELINE_EXTRA_CFLAGS CFLAGS
append_flags_from_env "${BENCHMARK_ID}_EXTRA_CFLAGS" CFLAGS

CXXFLAGS=(-std=c++17 -fno-exceptions -fno-rtti)
CXXFLAGS+=("${BENCHMARK_EXTRA_CXXFLAGS[@]}")
append_flags_from_env PIPELINE_EXTRA_CXXFLAGS CXXFLAGS
append_flags_from_env "${BENCHMARK_ID}_EXTRA_CXXFLAGS" CXXFLAGS

declare -a INCLUDE_FLAGS=()
declare -A INCLUDE_FLAG_SEEN=()
for dir in "${SOURCE_DIRS[@]}"; do
  add_include_flag "${dir}"
done
add_include_flag "${REPO_ROOT}/include"
add_include_flag "${REPO_ROOT}/runtime/include"
if [[ "${BENCHMARK_ID}" == "MONETDB" ]]; then
  MONETDB_CONFIG_HEADER="${MONETDB_CONFIG_HEADER:-${REPO_ROOT}/bench/MonetDB/monetdb_config.h}"
  if [[ ! -f "${MONETDB_CONFIG_HEADER}" ]]; then
    die "MonetDB benchmark requires a configured monetdb_config.h; set MONETDB_CONFIG_HEADER to its location."
  fi
  add_include_flag "$(dirname "${MONETDB_CONFIG_HEADER}")"
fi
for inc in "${BENCHMARK_EXTRA_INCLUDES[@]}"; do
  add_include_flag "${inc}"
done
append_includes_from_env PIPELINE_EXTRA_INCLUDES
append_includes_from_env "${BENCHMARK_ID}_EXTRA_INCLUDES"

LINK_FLAGS=("-L${RUNTIME_LIB_DIR}" "-Wl,-rpath,${RUNTIME_LIB_DIR}" -lcira_runtime)
LINK_FLAGS+=("${BENCHMARK_EXTRA_LDFLAGS[@]}")
append_flags_from_env PIPELINE_EXTRA_LDFLAGS LINK_FLAGS
append_flags_from_env "${BENCHMARK_ID}_EXTRA_LDFLAGS" LINK_FLAGS

ARCHES=("x86_64-unknown-linux-gnu" "aarch64-unknown-linux-gnu")
LL_FILES=()

info "Translating ${BENCHMARK} sources through ClangIR and Cira"
for src in "${SOURCES[@]}"; do
  if [[ "${src}" == "${REPO_ROOT}/"* ]]; then
    rel="${src#${REPO_ROOT}/}"
  else
    rel="$(basename "${src}")"
  fi
  stem="${rel%.*}"
  cir_path="${CIR_DIR}/${stem}.cir"
  mlir_path="${MLIR_DIR}/${stem}.mlir"
  cira_path="${CIRA_DIR}/${stem}.cira.mlir"
  llvm_mlir_path="${LLVM_DIR}/${stem}.llvm.mlir"
  llvm_path="${LLVM_DIR}/${stem}.ll"

  mkdir -p "$(dirname "${cir_path}")" "$(dirname "${mlir_path}")" "$(dirname "${cira_path}")" "$(dirname "${llvm_path}")"

  info "  > ${rel}"
  ext="${src##*.}"
  compile_flags=("${COMMON_FLAGS[@]}")
  case "${ext}" in
    c)
      compile_flags+=("${CFLAGS[@]}" -x c)
      ;;
    *)
      compile_flags+=("${CXXFLAGS[@]}" -x c++)
      ;;
  esac


  "${CLANGIR_BIN}" -fclangir -emit-cir -S -fno-strict-aliasing "${compile_flags[@]}" "${INCLUDE_FLAGS[@]}" "${src}" -o "${cir_path}"
  "${CLANGIR_OPT_BIN}" -allow-unregistered-dialect -cir-mlir-scf-prepare -cir-to-mlir "${cir_path}" -o "${mlir_path}"

  "${CIRA_BIN}" -allow-unregistered-dialect "${mlir_path}" \
    --cir-to-cira --rmem-search-remote -o "${cira_path}.unfixed"

  # Fix scf.while blocks by adding missing scf.yield terminators
  python3 "${REPO_ROOT}/scripts/fix_scf_while_yields.py" "${cira_path}.unfixed" "${cira_path}"
  rm -f "${cira_path}.unfixed"

  "${CIRA_BIN}" -allow-unregistered-dialect \
    --pass-pipeline='builtin.module(convert-cira-to-llvm-hetero,convert-scf-to-cf,convert-cf-to-llvm,convert-to-llvm,reconcile-unrealized-casts)' \
    "${cira_path}" -o "${llvm_mlir_path}.dirty"

  python3 "${REPO_ROOT}/scripts/cleanup_unrealized_casts.py" "${llvm_mlir_path}.dirty" > "${llvm_mlir_path}.tmp"
  python3 "${REPO_ROOT}/scripts/remove_leftover_scf.py" "${llvm_mlir_path}.tmp" "${llvm_mlir_path}"
  rm -f "${llvm_mlir_path}.dirty" "${llvm_mlir_path}.tmp"

  "${MLIR_TRANSLATE_BIN}" --allow-unregistered-dialect --mlir-to-llvmir "${llvm_mlir_path}" -o "${llvm_path}"


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

if [[ "${SKIP_LINK}" != "true" ]]; then
  info "Linking x86_64 binary"
  X86_ARCH="x86_64-unknown-linux-gnu"
  X86_OBJ_DIR="${OBJ_DIR}/${X86_ARCH}"
  mapfile -t X86_OBJS < <(find "${X86_OBJ_DIR}" -type f -name '*.o' -print | sort)
  (( ${#X86_OBJS[@]} > 0 )) || die "No x86_64 object files to link"

  mkdir -p "${BIN_ROOT}/${X86_ARCH}"
  "${LINKER_BIN}" -target "${X86_ARCH}" "${X86_OBJS[@]}" "${LINK_FLAGS[@]}" -o "${BIN_ROOT}/${X86_ARCH}/${BIN_NAME}"
else
  info "Skipping binary linkage for ${BENCHMARK} (configured to skip)"
fi

AARCH64_ARCH="aarch64-unknown-linux-gnu"
mkdir -p "${BIN_ROOT}/${AARCH64_ARCH}"

# Check if we should build heterogeneous aarch64 binary
if [[ "${BUILD_HETERO:-false}" == "true" ]] || [[ "${BENCHMARK}" == "mcf" ]]; then
  info "Linking aarch64 binary with x86 memory offloading"
  AARCH64_OBJ_DIR="${OBJ_DIR}/${AARCH64_ARCH}"
  mapfile -t AARCH64_OBJS < <(find "${AARCH64_OBJ_DIR}" -type f -name '*.o' -print | sort)

  if (( ${#AARCH64_OBJS[@]} > 0 )); then
    # Link aarch64 binary with cross-compilation support
    # Note: This requires aarch64 cross-compilation toolchain
    if command -v aarch64-linux-gnu-g++ &> /dev/null; then
      # Compile compiler runtime stubs if needed
      STUB_FILE="${REPO_ROOT}/runtime/compiler_rt_stubs_aarch64.o"
      if [[ ! -f "${STUB_FILE}" ]]; then
        if [[ -f "${REPO_ROOT}/runtime/compiler_rt_stubs.c" ]]; then
          aarch64-linux-gnu-gcc -c "${REPO_ROOT}/runtime/compiler_rt_stubs.c" -o "${STUB_FILE}" 2>/dev/null
        fi
      fi

      # Link with stubs if available
      LINK_OBJS=("${AARCH64_OBJS[@]}")
      [[ -f "${STUB_FILE}" ]] && LINK_OBJS+=("${STUB_FILE}")

      aarch64-linux-gnu-g++ "${LINK_OBJS[@]}" \
        -o "${BIN_ROOT}/${AARCH64_ARCH}/${BIN_NAME}" 2>/dev/null || {
          info "  Note: aarch64 linking failed, copying objects instead"
          find "${OBJ_DIR}/${AARCH64_ARCH}" -type f -name '*.o' -exec cp {} "${BIN_ROOT}/${AARCH64_ARCH}" \;
        }
    else
      info "  Note: aarch64-linux-gnu-g++ not found, copying objects for manual linking"
      find "${OBJ_DIR}/${AARCH64_ARCH}" -type f -name '*.o' -exec cp {} "${BIN_ROOT}/${AARCH64_ARCH}" \;
    fi
  fi
else
  find "${OBJ_DIR}/${AARCH64_ARCH}" -type f -name '*.o' -exec cp {} "${BIN_ROOT}/${AARCH64_ARCH}" \;
fi

info "Build completed"
if [[ "${SKIP_LINK}" != "true" ]]; then
  info "  x86_64 binary: ${BIN_ROOT}/${X86_ARCH}/${BIN_NAME}"
else
  info "  x86_64 objects available under ${OBJ_DIR}/${X86_ARCH} (link step skipped)"
fi

# Check if aarch64 binary was created
if [[ -f "${BIN_ROOT}/${AARCH64_ARCH}/${BIN_NAME}" ]]; then
  info "  aarch64 binary: ${BIN_ROOT}/${AARCH64_ARCH}/${BIN_NAME}"
else
  info "  aarch64 objects: ${BIN_ROOT}/${AARCH64_ARCH}"
  info "Use aarch64-linux-gnu-g++ to link the objects with -lcira_runtime"
fi
