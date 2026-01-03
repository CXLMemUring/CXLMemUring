#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"

LOCAL_CLANG_BIN="${REPO_ROOT}/../clangir/build/bin/clang"
LOCAL_CIR_OPT_BIN="${REPO_ROOT}/../clangir/build/bin/cir-opt"
LOCAL_MLIR_TRANSLATE_BIN="${REPO_ROOT}/../clangir/build/bin/mlir-translate"
LOCAL_LLC_BIN="${REPO_ROOT}/../clangir/build/bin/llc"

# Prefer repo-built toolchain if available; fallback to system.
if [[ -z "${CLANGIR_BIN:-}" ]]; then
  if [[ -x "${LOCAL_CLANG_BIN}" ]]; then
    CLANGIR_BIN="${LOCAL_CLANG_BIN}"
  else
    CLANGIR_BIN="$(command -v clang++ || true)"
  fi
fi
if [[ -z "${CLANGIR_OPT_BIN:-}" ]]; then
  if [[ -x "${LOCAL_CIR_OPT_BIN}" ]]; then
    CLANGIR_OPT_BIN="${LOCAL_CIR_OPT_BIN}"
  else
    CLANGIR_OPT_BIN="$(command -v cir-opt || true)"
  fi
fi
CIRA_BIN="${CIRA_BIN:-${REPO_ROOT}/build/bin/cira}"
if [[ -z "${MLIR_TRANSLATE_BIN:-}" ]]; then
  if [[ -x "${LOCAL_MLIR_TRANSLATE_BIN}" ]]; then
    MLIR_TRANSLATE_BIN="${LOCAL_MLIR_TRANSLATE_BIN}"
  else
    MLIR_TRANSLATE_BIN="$(command -v mlir-translate || true)"
  fi
fi
if [[ -z "${LLC_BIN:-}" ]]; then
  # Prefer system LLC over clangir LLC due to codegen bugs in clangir's version.
  # The LLVM IR attribute syntax is fixed by sed post-processing above.
  # Use explicit /usr/bin/llc since PATH may have clangir first.
  if [[ -x "/usr/bin/llc" ]]; then
    LLC_BIN="/usr/bin/llc"
  elif [[ -x "${LOCAL_LLC_BIN}" ]]; then
    LLC_BIN="${LOCAL_LLC_BIN}"
  else
    LLC_BIN="$(command -v llc || true)"
  fi
fi
# Use the same compiler for linking as for compiling by default.
LINKER_BIN="${LINKER_BIN:-${CLANGIR_BIN}}"
RUNTIME_LIB_DIR="${RUNTIME_LIB_DIR:-${REPO_ROOT}/build/lib}"
WORK_BASE_DIR="${PIPELINE_WORK_ROOT:-${REPO_ROOT}/build}"
BIN_BASE_DIR="${PIPELINE_BIN_ROOT:-${REPO_ROOT}/bin}"

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

ensure_dir_writable() {
  local dir="$1"
  mkdir -p "${dir}" 2>/dev/null || true
  if [[ ! -d "${dir}" ]]; then
    die "Failed to create build directory '${dir}'. Check permissions or set PIPELINE_WORK_ROOT/PIPELINE_BIN_ROOT."
  fi
  if [[ ! -w "${dir}" ]]; then
    die "Build directory '${dir}' is not writable. Adjust permissions or override PIPELINE_WORK_ROOT/PIPELINE_BIN_ROOT."
  fi
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

Environment overrides:
  PIPELINE_WORK_ROOT  Base directory for intermediate build artifacts (default: \
                     "${REPO_ROOT}/build")
  PIPELINE_BIN_ROOT   Base directory for linked binaries (default: "${REPO_ROOT}/bin")
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
    BENCHMARK_EXTRA_LDFLAGS=(-lstdc++ -lpthread -lm)
    WORK_ROOT="${WORK_BASE_DIR}/gapbs_pipeline"
    BIN_ROOT="${BIN_BASE_DIR}/gapbs"
    BIN_NAME="gapbs"
    ;;
  mcf)
    BENCHMARK_ID="MCF"
    SOURCE_DIRS=("${REPO_ROOT}/bench/mcf")
    WORK_ROOT="${WORK_BASE_DIR}/mcf_pipeline"
    BIN_ROOT="${BIN_BASE_DIR}/mcf"
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
      "/src/llama-adapter.cpp"
      "/src/llama-batch.cpp"    # System clang crash in ValueHandleBase::AddToUseList (X86 DAG)
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
    WORK_ROOT="${WORK_BASE_DIR}/llama_pipeline"
    BIN_ROOT="${BIN_BASE_DIR}/llama.cpp"
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
      "${REPO_ROOT}/bench/MonetDB/clients/mapilib"
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
      "/clients/odbc/"
      "/clients/mapiclient/"
      "/clients/examples/"
      "/clients/ruby/"
      "/clients/Tests/"
      "/geom/"
      "/tools/merovingian/"
      "/tools/monetdbe/"
      "/ctest/"
      "/rapi/"
      "/pyapi3/"
      "/batxml.c"
      "/xml.c"
      "/fits/"                  # Missing fitsio.h external library
      "/netcdf/"                # Missing netcdf.h external library
      "/shp/"                   # Missing gdal.h external library
      "/openssl_windows.c"      # Windows-specific
    )
    # Include generated parser file and monet_version.c
    EXTRA_SOURCES=(
      "${REPO_ROOT}/bench/MonetDB/tools/mserver/mserver5.c"
      "${REPO_ROOT}/bench/MonetDB/build/tools/mserver/monet_version.c"
      "${REPO_ROOT}/bench/MonetDB/build/sql/server/sql_parser.tab.c"
    )
    BENCHMARK_EXTRA_INCLUDES=(
      "${REPO_ROOT}/bench/MonetDB/build/common/utils/"
      "${REPO_ROOT}/bench/MonetDB/build"
      "${REPO_ROOT}/bench/MonetDB/build/sql/server"
      "${REPO_ROOT}/bench/MonetDB/common/stream/"
      "${REPO_ROOT}/bench/MonetDB/common/options/"
      "${REPO_ROOT}/bench/MonetDB/common/utils/"
      "${REPO_ROOT}/bench/MonetDB"
      "${REPO_ROOT}/bench/MonetDB/common"
      "${REPO_ROOT}/bench/MonetDB/gdk"
      "${REPO_ROOT}/bench/MonetDB/monetdb5"
      "${REPO_ROOT}/bench/MonetDB/monetdb5/mal"
      "${REPO_ROOT}/bench/MonetDB/monetdb5/optimizer"
      "${REPO_ROOT}/bench/MonetDB/monetdb5/scheduler"
      "${REPO_ROOT}/bench/MonetDB/monetdb5/modules/atoms"
      "${REPO_ROOT}/bench/MonetDB/monetdb5/modules/kernel"
      "${REPO_ROOT}/bench/MonetDB/monetdb5/modules/mal"
      "${REPO_ROOT}/bench/MonetDB/sql/include"
      "${REPO_ROOT}/bench/MonetDB/sql/common"
      "${REPO_ROOT}/bench/MonetDB/sql/server"
      "${REPO_ROOT}/bench/MonetDB/sql/storage"
      "${REPO_ROOT}/bench/MonetDB/sql/backends/monet5"
      "${REPO_ROOT}/bench/MonetDB/tools/mserver"
      "${REPO_ROOT}/bench/MonetDB/clients/mapilib"
      "/usr/include/libxml2"
    )
    BENCHMARK_EXTRA_CFLAGS+=(-D_GNU_SOURCE -DLIBMAL -DLIBMONETDB5 -DLIBOPTIMIZER -DLIBGDK -DLIBSTREAM -DLIBSQL -DLIBMAPI -DHAVE_MAPI)
    # Add external library dependencies: zlib, liblzma, OpenSSL
    BENCHMARK_EXTRA_LDFLAGS=(-lpthread -ldl -lm -lz -llzma -lssl -lcrypto)
    WORK_ROOT="${WORK_BASE_DIR}/monetdb_pipeline"
    BIN_ROOT="${BIN_BASE_DIR}/monetdb"
    BIN_NAME="mserver5"
    ;;
  dataframe|df)
    BENCHMARK_ID="DATAFRAME"
    SOURCE_DIRS=("${REPO_ROOT}/bench/DataFrame/src")
    EXTRA_SOURCES=("${REPO_ROOT}/bench/DataFrame/benchmarks/dataframe_performance.cc")
    BENCHMARK_EXTRA_INCLUDES=("${REPO_ROOT}/bench/DataFrame/include")
    # DataFrame requires C++20 (set via CXX_STD) and exceptions/RTTI
    BENCHMARK_EXTRA_CXXFLAGS+=(-fexceptions -frtti)
    BENCHMARK_EXTRA_LDFLAGS=(-lpthread)
    if [[ "$(uname -s 2>/dev/null)" == "Linux" ]]; then
      BENCHMARK_EXTRA_LDFLAGS+=(-lrt)
    fi
    WORK_ROOT="${WORK_BASE_DIR}/dataframe_pipeline"
    BIN_ROOT="${BIN_BASE_DIR}/dataframe"
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

ensure_dir_writable "${WORK_ROOT}"
ensure_dir_writable "${BIN_ROOT}"

CIR_DIR="${WORK_ROOT}/cir"
MLIR_DIR="${WORK_ROOT}/mlir"
CIRA_DIR="${WORK_ROOT}/cira"
LLVM_DIR="${WORK_ROOT}/llvm"
OBJ_DIR="${WORK_ROOT}/obj"

for dir in "${CIR_DIR}" "${MLIR_DIR}" "${CIRA_DIR}" "${LLVM_DIR}" "${OBJ_DIR}"; do
  ensure_dir_writable "${dir}"
done

COMMON_FLAGS=(-O3 -I/root/CXLMemUring/bench/MonetDB/common/utils/ -I/root/CXLMemUring/bench/MonetDB/build)

CFLAGS=(-std=c11)
CFLAGS+=("${BENCHMARK_EXTRA_CFLAGS[@]}")
append_flags_from_env PIPELINE_EXTRA_CFLAGS CFLAGS
append_flags_from_env "${BENCHMARK_ID}_EXTRA_CFLAGS" CFLAGS

# Default C++ standard is C++17, but some benchmarks need C++20
CXX_STD="${CXX_STD:-c++17}"
if [[ "${BENCHMARK_ID}" == "DATAFRAME" ]]; then
    CXX_STD="c++20"
fi
CXXFLAGS=(-std=${CXX_STD} -fno-exceptions -fno-rtti)
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

LINK_FLAGS=("-L${RUNTIME_LIB_DIR}" "-Wl,-rpath,${RUNTIME_LIB_DIR}" -L/usr/local/lib -Wl,--allow-multiple-definition -lstdc++ -L/home/victoryang00/CXLMemUring/build/runtime -Wl,-rpath,/home/victoryang00/CXLMemUring/build/runtime  -lcira_runtime)
LINK_FLAGS+=("${BENCHMARK_EXTRA_LDFLAGS[@]}")
append_flags_from_env PIPELINE_EXTRA_LDFLAGS LINK_FLAGS
append_flags_from_env "${BENCHMARK_ID}_EXTRA_LDFLAGS" LINK_FLAGS

ARCHES=("x86_64-unknown-linux-gnu" "riscv64-unknown-linux-gnu")
LL_FILES=()

info "Translating ${BENCHMARK} sources through ClangIR and Cira"
X86_ARCH="x86_64-unknown-linux-gnu"
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

  ensure_dir_writable "$(dirname "${cir_path}")"
  ensure_dir_writable "$(dirname "${mlir_path}")"
  ensure_dir_writable "$(dirname "${cira_path}")"
  ensure_dir_writable "$(dirname "${llvm_path}")"

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


  # Workaround: GAPBS cxx_runtime_support.cpp provides vtables and STL symbols
  # that the CIR pipeline doesn't emit correctly. Must be compiled natively.
  if [[ "${BENCHMARK_ID}" == "GAPBS" && "${src}" == */cxx_runtime_support.cpp ]]; then
    arch_dir="${OBJ_DIR}/${X86_ARCH}"
    ensure_dir_writable "${arch_dir}"
    obj="${arch_dir}/${stem}.o"
    ensure_dir_writable "$(dirname "${obj}")"
    info "    (native compilation - C++ runtime support for vtables/STL)"
    NATIVE_CXX="/usr/bin/clang++"
    [[ ! -x "${NATIVE_CXX}" ]] && NATIVE_CXX="$(command -v g++ || echo "${LINKER_BIN}")"
    # Use standard C++ flags with RTTI and exceptions enabled for proper vtable emission
    "${NATIVE_CXX}" -c -std=c++17 -fno-strict-aliasing \
      "${INCLUDE_FLAGS[@]}" "${src}" \
      -O3 -o "${obj}"
    continue
  fi

  # NOTE: GAPBS native bypass disabled - compiling through CIR/Cira pipeline
  # The original workaround was for C++ exception handling cf.br issues.
  # if [[ "${BENCHMARK_ID}" == "GAPBS" && "${src}" == *.cc ]]; then
  #   arch_dir="${OBJ_DIR}/${X86_ARCH}"
  #   ensure_dir_writable "${arch_dir}"
  #   obj="${arch_dir}/${stem}.o"
  #   ensure_dir_writable "$(dirname "${obj}")"
  #   info "    (native compilation - C++ exception handling bypass)"
  #   NATIVE_CXX="/usr/bin/clang++"
  #   [[ ! -x "${NATIVE_CXX}" ]] && NATIVE_CXX="$(command -v g++ || echo "${LINKER_BIN}")"
  #   "${NATIVE_CXX}" -c -std=c++17 -fno-strict-aliasing \
  #     "${INCLUDE_FLAGS[@]}" "${src}" \
  #     -O3 -o "${obj}"
  #   continue
  # fi

  # Workaround: Some llama.cpp files trigger CIR frontend hangs or crashes.
  # Compile them natively for x86_64 and bypass the CIR/MLIR pipeline.
  # - llama-arch.cpp: extremely large nested initializer lists
  # - llama-batch.cpp: complex template instantiations causing CIR hang
  if [[ "${BENCHMARK_ID}" == "LLAMACPP" && ( "${src}" == */llama-arch.cpp || "${src}" == */llama-batch.cpp ) ]]; then
    arch_dir="${OBJ_DIR}/${X86_ARCH}"
    ensure_dir_writable "${arch_dir}"
    obj="${arch_dir}/${stem}.o"
    ensure_dir_writable "$(dirname "${obj}")"
    # Compile with optimizations fully disabled to avoid optimizer (SROA)
    # crashes on the enormous global initializer in this TU. Ensure -O0 and
    # -Xclang -disable-llvm-passes are applied after any inherited flags.
    "${LINKER_BIN}" -target "${X86_ARCH}" -c -fno-strict-aliasing \
      "${compile_flags[@]}" "${INCLUDE_FLAGS[@]}" "${src}" \
      -O0 -fno-vectorize -fno-slp-vectorize -fno-inline \
      -Xclang -disable-llvm-passes \
      -o "${obj}"
    # Skip CIR/MLIR for this file.
    continue
  fi

  # Workaround: Some MonetDB files cause issues in the CIR/MLIR pipeline:
  # - gdk_aggr.c: stack overflow in MLIR printer (2000+ nested if-else chains)
  # - switch statement files: scf.yield legalization issues
  # Compile these natively using system clang (not clangir which has bugs).
  # Files that need native compilation - patterns support glob matching
  MONETDB_NATIVE_PATTERNS=(
    # MAPI client library files - OpenSSL and va_copy issues
    "connect_openssl.c"       # Requires OpenSSL headers for TLS support
    "connect.c"               # Uses va_copy which isn't lowered correctly by ClangIR
    "msettings.c"             # Uses va_copy which isn't lowered correctly by ClangIR
    "monet_version.c"         # Generated file - compile natively
    "mserver5.c"              # Main entry point - compile natively for compatibility
    # GDK files have many issues with switch statements, deep nesting, etc.
    "gdk_*.c"                 # Most gdk files have conversion issues
    # Stream files have switch statement issues
    "bs.c"                    # switch -> scf.yield legalization
    "gz_stream.c"             # switch -> scf.yield legalization
    "mapi_stream.c"           # switch -> scf.yield legalization
    "pump.c"                  # switch -> scf.yield legalization
    "rw.c"                    # switch -> scf.yield legalization
    "stream.c"                # switch -> scf.yield legalization
    "text_stream.c"           # switch -> scf.yield legalization
    "socket_stream.c"         # switch -> scf.yield legalization
    "lz4_stream.c"            # switch -> scf.yield legalization
    "xz_stream.c"             # switch -> scf.yield legalization
    "msabaoth.c"              # switch -> scf.yield legalization
    # MonetDB5 optimizer files - mlir-translate crashes on these
    "opt_*.c"                 # mlir-translate crash during LLVM IR generation
    "optimizer.c"             # CIR parsing error (type mismatch in const_array)
    # MAL (MonetDB Assembly Language) files have switch statements
    "mal_*.c"                 # switch -> scf.yield legalization
    "mal.c"                   # CIR parsing error (type mismatch)
    # Atoms modules have mel_atom struct init issues (type mismatch in const_array)
    "blob.c"
    "color.c"
    "identifier.c"
    "inet.c"
    "json.c"
    "mtime.c"
    "str.c"
    "streams.c"
    "strptime.c"
    "url.c"
    "uuid.c"
    # Kernel modules - MonetDB kernel files have mel_func struct init issues (type mismatch in const_array)
    # These files use compound literal initializers that cause clangir CIR parsing errors
    # Add specific patterns for files in monetdb5/modules/kernel/
    "aggr.c"                  # CIR parsing error (type mismatch in const_array)
    "alarm.c"                 # CIR parsing error (type mismatch in const_array)
    "algebra.c"               # CIR parsing error (type mismatch in const_array)
    "bat5.c"                  # CIR parsing error (type mismatch in const_array)
    "batcolor.c"              # CIR parsing error (type mismatch in const_array)
    "batmmath.c"              # CIR parsing error (type mismatch in const_array)
    "batstr.c"                # CIR parsing error (type mismatch in const_array)
    "group.c"                 # CIR parsing error (type mismatch in const_array)
    "microbenchmark.c"        # CIR parsing error (type mismatch in const_array)
    "mmath.c"                 # CIR parsing error (type mismatch in const_array)
    # MAL modules - monetdb5/modules/mal/ files with the same mel_func init issues
    "batcalc.c"               # CIR parsing error (type mismatch in const_array)
    "batExtensions.c"         # CIR parsing error (type mismatch in const_array)
    "batMask.c"               # CIR parsing error (type mismatch in const_array)
    "bbp.c"                   # CIR parsing error (type mismatch in const_array)
    "calc.c"                  # CIR parsing error (type mismatch in const_array)
    "clients.c"               # CIR parsing error (type mismatch in const_array)
    "groupby.c"               # CIR parsing error (type mismatch in const_array)
    "inspect.c"               # CIR parsing error (type mismatch in const_array)
    "iterator.c"              # CIR parsing error (type mismatch in const_array)
    "language.c"              # CIR parsing error (type mismatch in const_array)
    "mal_io.c"                # CIR parsing error (type mismatch in const_array)
    "mal_mapi.c"              # CIR parsing error (type mismatch in const_array)
    "manifold.c"              # CIR parsing error (type mismatch in const_array)
    "manual.c"                # CIR parsing error (type mismatch in const_array)
    "mat.c"                   # CIR parsing error (type mismatch in const_array)
    "mdb.c"                   # CIR parsing error (type mismatch in const_array)
    "mkey.c"                  # CIR parsing error (type mismatch in const_array)
    "orderidx.c"              # CIR parsing error (type mismatch in const_array)
    "pcre.c"                  # CIR parsing error (type mismatch in const_array)
    "profiler.c"              # CIR parsing error (type mismatch in const_array)
    "projectionpath.c"        # CIR parsing error (type mismatch in const_array)
    "querylog.c"              # CIR parsing error (type mismatch in const_array)
    "remote.c"                # CIR parsing error (type mismatch in const_array)
    "sample.c"                # CIR parsing error (type mismatch in const_array)
    "sysmon.c"                # CIR parsing error (type mismatch in const_array)
    "tablet.c"                # CIR parsing error (type mismatch in const_array)
    "tracer.c"                # CIR parsing error (type mismatch in const_array)
    "txtsim.c"                # CIR parsing error (type mismatch in const_array)
    # SQL backend files with switch statements or type mismatch
    "dict.c"                  # switch -> scf.yield legalization (sql/backends/monet5)
    "for.c"                   # switch -> scf.yield legalization (sql/backends/monet5)
    "generator.c"             # CIR parsing error (type mismatch in const_array)
    "rel_bin.c"               # scf.yield legalization (sql/backends/monet5)
    # SQL backend monet5 files with mlir-translate crashes
    "sql_*.c"                 # mlir-translate crash (INVALIDBLOCK references in various files)
    "sql.c"                   # CIR parsing error (type mismatch in const_record)
    "capi.c"                  # cir-opt crash (UnrealizedConversionCastOp print)
    "udf.c"                   # scf.yield legalization failure
    "csv.c"                   # cir.const legalization for 2D arrays
    "fits.c"                  # Missing fitsio.h library
  )
  needs_native=false
  if [[ "${BENCHMARK_ID}" == "MONETDB" ]]; then
    for pattern in "${MONETDB_NATIVE_PATTERNS[@]}"; do
      # Support glob patterns in the match
      basename_src="$(basename "${src}")"
      if [[ "${basename_src}" == ${pattern} ]]; then
        needs_native=true
        break
      fi
    done
  fi
  if [[ "${needs_native}" == "true" ]]; then
    arch_dir="${OBJ_DIR}/${X86_ARCH}"
    ensure_dir_writable "${arch_dir}"
    obj="${arch_dir}/${stem}.o"
    ensure_dir_writable "$(dirname "${obj}")"
    info "    (native compilation - known CIR/MLIR conversion issue)"
    # Use system clang for native fallback (clangir's clang has bugs)
    # Prefer /usr/bin/clang over PATH since PATH may have clangir first
    NATIVE_CC="/usr/bin/clang"
    [[ ! -x "${NATIVE_CC}" ]] && NATIVE_CC="$(command -v gcc || echo "${LINKER_BIN}")"
    "${NATIVE_CC}" -c -fno-strict-aliasing \
      "${compile_flags[@]}" "${INCLUDE_FLAGS[@]}" "${src}" \
      -O3 -o "${obj}"
    continue
  fi

  # Note: -emit-cir has a known DiagStorageAllocator crash during cleanup.
  # The CIR file is written before the crash, so we check for valid output.
  echo "${CLANGIR_BIN}" -fclangir -emit-cir -S -fno-strict-aliasing "${compile_flags[@]}" "${INCLUDE_FLAGS[@]}" "${src}"
  "${CLANGIR_BIN}" -fclangir -emit-cir -S -fno-strict-aliasing "${compile_flags[@]}" "${INCLUDE_FLAGS[@]}" "${src}" -o "${cir_path}" 2>/dev/null || true
  if [[ ! -s "${cir_path}" ]]; then
    die "Failed to generate CIR for ${src}"
  fi
  
  # Check if we should use the direct CIR -> LLVM path (bypasses the crashing cir-to-mlir pass)
  USE_DIRECT_PATH=false
  if [[ "${PIPELINE_LLVM_LOWERING:-}" == "direct" || "${BENCHMARK_ID}" == "MCF" || "${BENCHMARK_ID}" == "LLAMACPP" || "${BENCHMARK_ID}" == "DATAFRAME" || "${BENCHMARK_ID}" == "MONETDB" || "${BENCHMARK_ID}" == "GAPBS" ]]; then
    USE_DIRECT_PATH=true
  fi

  if [[ "${USE_DIRECT_PATH}" == "false" ]]; then
    # Only run cir-to-mlir for heterogeneous path (non-direct benchmarks)
    echo "${CLANGIR_OPT_BIN}" -allow-unregistered-dialect -cir-mlir-scf-prepare -cir-to-mlir "${cir_path}" -o "${mlir_path}"
    "${CLANGIR_OPT_BIN}" -allow-unregistered-dialect -cir-mlir-scf-prepare -cir-to-mlir "${cir_path}" -o "${mlir_path}"
  fi

  if [[ "${USE_DIRECT_PATH}" == "true" ]]; then
    # Direct CIR -> MLIR -> LLVM dialect lowering and export
    direct_clean_path="${LLVM_DIR}/${stem}.clean.llvm.mlir"
    ensure_dir_writable "$(dirname "${direct_clean_path}")"

    # Build pass list for direct CIR -> MLIR -> LLVM lowering. FlattenCFG is
    # needed to convert unstructured control flow (early returns in loops) to
    # structured control flow that SCF lowering can handle.
    # - MCF: aggressive CFG simplification for legalizers
    # - DATAFRAME: has early returns inside loops that SCF can't handle directly
    PASS_FLAGS=(--allow-unregistered-dialect)
    if [[ "${BENCHMARK_ID}" == "MCF" || "${BENCHMARK_ID}" == "DATAFRAME" || "${BENCHMARK_ID}" == "GAPBS" || "${BENCHMARK_ID}" == "LLAMACPP" ]]; then
      PASS_FLAGS+=(--cir-flatten-cfg)
    fi
    # Use direct CIR to LLVM path for proper vtable and constructor support
    PASS_FLAGS+=(--cir-goto-solver --cir-to-llvm)

    echo "${CLANGIR_OPT_BIN}" "${PASS_FLAGS[@]}" "${cir_path}" -o "${direct_clean_path}"
    "${CLANGIR_OPT_BIN}" "${PASS_FLAGS[@]}" "${cir_path}" -o "${direct_clean_path}"

    # Scrub unrealized casts and any leftover scf/cf branches before export
    python3 "${REPO_ROOT}/scripts/cleanup_unrealized_casts.py" "${direct_clean_path}" > "${llvm_mlir_path}.tmp"
    python3 "${REPO_ROOT}/scripts/remove_leftover_scf.py" "${llvm_mlir_path}.tmp" "${llvm_mlir_path}"
    rm -f "${llvm_mlir_path}.tmp"

    "${MLIR_TRANSLATE_BIN}" --allow-unregistered-dialect --mlir-to-llvmir \
      "${llvm_mlir_path}" -o "${llvm_path}.raw"
    # Fix LLVM 22 syntax for compatibility with older LLC versions (v18)
    # - captures(none) -> nocapture
    # - getelementptr inbounds nuw -> getelementptr inbounds (remove nuw)
    # - errnomem: none -> remove (not supported in older LLVM)
    sed -e 's/captures(none)/nocapture/g' \
        -e 's/writeonly captures(none)/writeonly nocapture/g' \
        -e 's/readonly captures(none)/readonly nocapture/g' \
        -e 's/getelementptr inbounds nuw/getelementptr inbounds/g' \
        -e 's/, errnomem: none//g' \
        -e 's/errnomem: none, //g' \
        -e 's/errnomem: none//g' \
        "${llvm_path}.raw" > "${llvm_path}.tmp1"
    rm -f "${llvm_path}.raw"

    # Generate empty function bodies for trivial C++ constructors that are
    # declared but not defined (common issue with STL tag types, empty structs)
    python3 "${REPO_ROOT}/scripts/fix_trivial_constructors.py" "${llvm_path}.tmp1" > "${llvm_path}"
    rm -f "${llvm_path}.tmp1"
  else
    # Heterogeneous path via CIRA as before
    "${CIRA_BIN}" -allow-unregistered-dialect "${mlir_path}" \
      --cir-to-cira --rmem-search-remote -o "${cira_path}.unfixed"

    # Fix scf.while blocks by adding missing scf.yield terminators
    python3 "${REPO_ROOT}/scripts/fix_scf_while_yields.py" "${cira_path}.unfixed" "${cira_path}"
    rm -f "${cira_path}.unfixed"

    # Apply profile-guided offload pass if profile is available
    if [[ -n "${DISAGG_PROFILE_PATH:-}" && -f "${DISAGG_PROFILE_PATH}" ]]; then
      info "Applying profile-guided offload decisions from ${DISAGG_PROFILE_PATH}"

      # Determine offload target (default to vortex)
      OFFLOAD_TARGET_FLAG="${OFFLOAD_TARGET:-vortex}"
      MIN_OFFLOAD_ELEMENTS="${MIN_OFFLOAD_ELEMENTS:-1000}"
      SPEEDUP_THRESHOLD="${SPEEDUP_THRESHOLD:-1.5}"

      # Apply profile-guided pass
      "${CIRA_BIN}" -allow-unregistered-dialect \
        --profile-guided-offload \
        --offload-profile="${DISAGG_PROFILE_PATH}" \
        --offload-target="${OFFLOAD_TARGET_FLAG}" \
        --min-offload-elements="${MIN_OFFLOAD_ELEMENTS}" \
        --speedup-threshold="${SPEEDUP_THRESHOLD}" \
        "${cira_path}" -o "${cira_path}.pgo"

      mv "${cira_path}.pgo" "${cira_path}"
    fi

    # Select conversion pass based on offload target
    if [[ "${OFFLOAD_TARGET:-}" == "vortex" ]]; then
      # Use Vortex-specific lowering for GPU offload
      "${CIRA_BIN}" -allow-unregistered-dialect \
        --pass-pipeline='builtin.module(convert-cira-to-llvm-vortex,convert-scf-to-cf,convert-cf-to-llvm,convert-to-llvm,reconcile-unrealized-casts)' \
        "${cira_path}" -o "${llvm_mlir_path}.dirty"
    else
      # Default heterogeneous lowering
      "${CIRA_BIN}" -allow-unregistered-dialect \
        --pass-pipeline='builtin.module(convert-cira-to-llvm-hetero,convert-scf-to-cf,convert-cf-to-llvm,convert-to-llvm,reconcile-unrealized-casts)' \
        "${cira_path}" -o "${llvm_mlir_path}.dirty"
    fi

    python3 "${REPO_ROOT}/scripts/cleanup_unrealized_casts.py" "${llvm_mlir_path}.dirty" > "${llvm_mlir_path}.tmp"
    python3 "${REPO_ROOT}/scripts/remove_leftover_scf.py" "${llvm_mlir_path}.tmp" "${llvm_mlir_path}"
    rm -f "${llvm_mlir_path}.dirty" "${llvm_mlir_path}.tmp"

    "${MLIR_TRANSLATE_BIN}" --allow-unregistered-dialect --mlir-to-llvmir "${llvm_mlir_path}" -o "${llvm_path}.raw"
    # Fix LLVM 22 syntax for compatibility with older LLC versions (v18)
    # - captures(none) -> nocapture
    # - getelementptr inbounds nuw -> getelementptr inbounds (remove nuw)
    sed -e 's/captures(none)/nocapture/g' \
        -e 's/writeonly captures(none)/writeonly nocapture/g' \
        -e 's/readonly captures(none)/readonly nocapture/g' \
        -e 's/getelementptr inbounds nuw/getelementptr inbounds/g' \
        "${llvm_path}.raw" > "${llvm_path}"
    rm -f "${llvm_path}.raw"
  fi


  LL_FILES+=("${llvm_path}")
done

info "Lowering Cira output to target objects"
OPT_BIN="${REPO_ROOT}/../clangir/build/bin/opt"
for arch in "${ARCHES[@]}"; do
  arch_dir="${OBJ_DIR}/${arch}"
  ensure_dir_writable "${arch_dir}"
  for ll in "${LL_FILES[@]}"; do
    rel="${ll#${LLVM_DIR}/}"
    stem="${rel%.ll}"
    obj="${arch_dir}/${stem}.o"
    ll_opt="${ll%.ll}.opt.ll"
    ensure_dir_writable "$(dirname "${obj}")"
    # Run opt to apply LLVM optimizations including NRVO before LLC
    "${OPT_BIN}" -O2 -S "${ll}" -o "${ll_opt}" 2>/dev/null || cp "${ll}" "${ll_opt}"
    # Work around an LLVM CodeGenPrepare crash on some degenerate CFGs by
    # disabling CGP entirely for llc on these IR files.
    "${LLC_BIN}" -O2 -filetype=obj -relocation-model=pic -disable-cgp -mtriple="${arch}" "${ll_opt}" -o "${obj}"
  done
done

if [[ "${SKIP_LINK}" != "true" ]]; then
  info "Linking x86_64 binary"
  X86_ARCH="x86_64-unknown-linux-gnu"
  X86_OBJ_DIR="${OBJ_DIR}/${X86_ARCH}"
  mapfile -t X86_OBJS < <(find "${X86_OBJ_DIR}" -type f -name '*.o' -print | sort)
  (( ${#X86_OBJS[@]} > 0 )) || die "No x86_64 object files to link"

  ensure_dir_writable "${BIN_ROOT}/${X86_ARCH}"

  # For GAPBS, each source file is a standalone benchmark - link separately
  if [[ "${BENCHMARK_KEY}" == "gapbs" ]]; then
    info "  Building separate GAPBS binaries..."
    for obj in "${X86_OBJS[@]}"; do
      bench_name=$(basename "${obj}" .o)
      info "    Linking ${bench_name}..."
      "${LINKER_BIN}" -target "${X86_ARCH}" "${obj}" "${LINK_FLAGS[@]}" -o "${BIN_ROOT}/${X86_ARCH}/${bench_name}"
    done
  else
    "${LINKER_BIN}" -target "${X86_ARCH}" "${X86_OBJS[@]}" "${LINK_FLAGS[@]}" -o "${BIN_ROOT}/${X86_ARCH}/${BIN_NAME}"
  fi
else
  info "Skipping binary linkage for ${BENCHMARK} (configured to skip)"
fi

RISCV64_ARCH="riscv64-unknown-linux-gnu"
ensure_dir_writable "${BIN_ROOT}/${RISCV64_ARCH}"

# Check if we should build heterogeneous riscv64 binary
if [[ "${BUILD_HETERO:-false}" == "true" ]] || [[ "${BENCHMARK}" == "mcf" ]]; then
  info "Linking riscv64 binary with x86 memory offloading"
  RISCV64_OBJ_DIR="${OBJ_DIR}/${RISCV64_ARCH}"
  mapfile -t RISCV64_OBJS < <(find "${RISCV64_OBJ_DIR}" -type f -name '*.o' -print | sort)

  if (( ${#RISCV64_OBJS[@]} > 0 )); then
    # Link riscv64 binary with cross-compilation support
    # Note: This requires riscv64 cross-compilation toolchain
    if command -v riscv64-linux-gnu-g++ &> /dev/null; then
      # Compile compiler runtime stubs if needed
      STUB_FILE="${REPO_ROOT}/runtime/compiler_rt_stubs_riscv64.o"
      if [[ ! -f "${STUB_FILE}" ]]; then
        if [[ -f "${REPO_ROOT}/runtime/compiler_rt_stubs.c" ]]; then
          riscv64-linux-gnu-gcc -c "${REPO_ROOT}/runtime/compiler_rt_stubs.c" -o "${STUB_FILE}" 2>/dev/null
        fi
      fi

      # Link with stubs if available
      LINK_OBJS=("${RISCV64_OBJS[@]}")
      [[ -f "${STUB_FILE}" ]] && LINK_OBJS+=("${STUB_FILE}")

      # For GAPBS, link each benchmark separately
      if [[ "${BENCHMARK_KEY}" == "gapbs" ]]; then
        for obj in "${RISCV64_OBJS[@]}"; do
          bench_name=$(basename "${obj}" .o)
          riscv64-linux-gnu-g++ "${obj}" ${STUB_FILE:+"${STUB_FILE}"} \
            -o "${BIN_ROOT}/${RISCV64_ARCH}/${bench_name}" 2>/dev/null || true
        done
      else
        riscv64-linux-gnu-g++ "${LINK_OBJS[@]}" \
          -o "${BIN_ROOT}/${RISCV64_ARCH}/${BIN_NAME}" 2>/dev/null || {
            info "  Note: riscv64 linking failed, copying objects instead"
            find "${OBJ_DIR}/${RISCV64_ARCH}" -type f -name '*.o' -exec cp {} "${BIN_ROOT}/${RISCV64_ARCH}" \;
          }
      fi
    else
      info "  Note: riscv64-linux-gnu-g++ not found, copying objects for manual linking"
      find "${OBJ_DIR}/${RISCV64_ARCH}" -type f -name '*.o' -exec cp {} "${BIN_ROOT}/${RISCV64_ARCH}" \;
    fi
  fi
else
  find "${OBJ_DIR}/${RISCV64_ARCH}" -type f -name '*.o' -exec cp {} "${BIN_ROOT}/${RISCV64_ARCH}" \;
fi

info "Build completed"
if [[ "${SKIP_LINK}" != "true" ]]; then
  if [[ "${BENCHMARK_KEY}" == "gapbs" ]]; then
    info "  x86_64 binaries in: ${BIN_ROOT}/${X86_ARCH}/"
    for bin in "${BIN_ROOT}/${X86_ARCH}"/*; do
      [[ -x "$bin" ]] && info "    - $(basename "$bin")"
    done
  else
    info "  x86_64 binary: ${BIN_ROOT}/${X86_ARCH}/${BIN_NAME}"
  fi
else
  info "  x86_64 objects available under ${OBJ_DIR}/${X86_ARCH} (link step skipped)"
fi

# Check if riscv64 binary was created
if [[ "${BENCHMARK_KEY}" == "gapbs" ]]; then
  info "  riscv64 binaries/objects: ${BIN_ROOT}/${RISCV64_ARCH}/"
elif [[ -f "${BIN_ROOT}/${RISCV64_ARCH}/${BIN_NAME}" ]]; then
  info "  riscv64 binary: ${BIN_ROOT}/${RISCV64_ARCH}/${BIN_NAME}"
else
  info "  riscv64 objects: ${BIN_ROOT}/${RISCV64_ARCH}"
  info "Use riscv64-linux-gnu-g++ to link the objects with -lcira_runtime"
fi
