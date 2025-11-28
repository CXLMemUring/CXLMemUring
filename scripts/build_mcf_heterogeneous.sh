#!/usr/bin/env bash
# Build MCF with profile-guided heterogeneous offloading

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."

# Directories
MCF_SRC="${REPO_ROOT}/bench/mcf"
RUNTIME_DIR="${REPO_ROOT}/runtime"
HETERO_DIR="${REPO_ROOT}/bin/mcf_heterogeneous"
OUTPUT_DIR="${REPO_ROOT}/bin/mcf_heterogeneous/build"

# Compiler settings - use system gcc to avoid clangir bugs
CC=gcc
CFLAGS="-O2 -Wall"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }

mkdir -p "${OUTPUT_DIR}"

#==============================================================================
# Build Options
#==============================================================================

BUILD_MODE="${1:-cpu}"  # cpu, vortex-sim, or vortex-hw

case "${BUILD_MODE}" in
    cpu)
        log_info "Building MCF with CPU-only offload simulation"
        EXTRA_CFLAGS="-DMCF_VORTEX_OFFLOAD"
        EXTRA_LDFLAGS=""
        ;;
    vortex-sim)
        log_info "Building MCF with Vortex simulator support"
        EXTRA_CFLAGS="-DMCF_VORTEX_OFFLOAD -DUSE_VORTEX_SIM"
        EXTRA_CFLAGS+=" -I${REPO_ROOT}/vortex/runtime/include"
        EXTRA_LDFLAGS="-L${REPO_ROOT}/vortex/runtime -lvortex-driver"
        ;;
    vortex-hw)
        log_info "Building MCF with Vortex hardware support"
        EXTRA_CFLAGS="-DMCF_VORTEX_OFFLOAD -DUSE_VORTEX_HW"
        EXTRA_CFLAGS+=" -I${REPO_ROOT}/vortex/runtime/include"
        EXTRA_LDFLAGS="-L${REPO_ROOT}/vortex/runtime -lvortex-driver"
        ;;
    *)
        echo "Usage: $0 [cpu|vortex-sim|vortex-hw]"
        exit 1
        ;;
esac

#==============================================================================
# Source Files
#==============================================================================

MCF_SOURCES=(
    "${MCF_SRC}/mcf.c"
    "${MCF_SRC}/pbeampp.c"
    "${MCF_SRC}/pbla.c"
    "${MCF_SRC}/pflowup.c"
    "${MCF_SRC}/psimplex.c"
    "${MCF_SRC}/pstart.c"
    "${MCF_SRC}/mcfutil.c"
    "${MCF_SRC}/readmin.c"
    "${MCF_SRC}/implicit.c"
    "${MCF_SRC}/output.c"
    "${MCF_SRC}/treeup.c"
    "${MCF_SRC}/main_wrapper.c"
)

RUNTIME_SOURCES=(
    "${HETERO_DIR}/mcf_vortex_runtime.c"
)

#==============================================================================
# Build
#==============================================================================

log_info "Compiling MCF sources..."

OBJECTS=()

for src in "${MCF_SOURCES[@]}"; do
    obj="${OUTPUT_DIR}/$(basename "${src}" .c).o"
    log_info "  Compiling $(basename "${src}")..."
    ${CC} ${CFLAGS} ${EXTRA_CFLAGS} \
        -I"${MCF_SRC}" \
        -I"${RUNTIME_DIR}" \
        -I"${HETERO_DIR}" \
        -c "${src}" -o "${obj}"
    OBJECTS+=("${obj}")
done

log_info "Compiling runtime..."
for src in "${RUNTIME_SOURCES[@]}"; do
    if [ -f "${src}" ]; then
        obj="${OUTPUT_DIR}/$(basename "${src}" .c).o"
        log_info "  Compiling $(basename "${src}")..."
        ${CC} ${CFLAGS} ${EXTRA_CFLAGS} \
            -I"${RUNTIME_DIR}" \
            -I"${HETERO_DIR}" \
            -c "${src}" -o "${obj}"
        OBJECTS+=("${obj}")
    fi
done

log_info "Linking mcf_heterogeneous..."
${CC} ${CFLAGS} "${OBJECTS[@]}" ${EXTRA_LDFLAGS} -lm -o "${OUTPUT_DIR}/mcf_heterogeneous"

log_info ""
log_info "Build complete!"
log_info "Binary: ${OUTPUT_DIR}/mcf_heterogeneous"
log_info ""
log_info "To run:"
log_info "  ${OUTPUT_DIR}/mcf_heterogeneous /path/to/inp.in"
