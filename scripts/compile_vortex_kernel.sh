#!/usr/bin/env bash
# Compile CIRA IR to Vortex RISC-V kernel binary (.vxbin)
#
# Usage: compile_vortex_kernel.sh <input.cir|input.c> <output.vxbin> [options]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
VORTEX_HOME="${REPO_ROOT}/vortex"

# Vortex configuration
XLEN=32
TOOLDIR="/root/tools"
LLVM_VORTEX="${TOOLDIR}/llvm-vortex"
LIBC_VORTEX="${TOOLDIR}/libc${XLEN}"
LIBCRT_VORTEX="${TOOLDIR}/libcrt${XLEN}"
RISCV_TOOLCHAIN_PATH="${TOOLDIR}/riscv${XLEN}-gnu-toolchain"
RISCV_PREFIX="riscv${XLEN}-unknown-elf"
RISCV_SYSROOT="${RISCV_TOOLCHAIN_PATH}/${RISCV_PREFIX}"

# Build tools
CIRA_BIN="${REPO_ROOT}/build/bin/cira"
VX_CC="${LLVM_VORTEX}/bin/clang"
VX_CXX="${LLVM_VORTEX}/bin/clang++"
VX_DP="${LLVM_VORTEX}/bin/llvm-objdump"
VX_CP="${LLVM_VORTEX}/bin/llvm-objcopy"
VXBIN_SCRIPT="${VORTEX_HOME}/kernel/scripts/vxbin.py"
LINK_SCRIPT="${VORTEX_HOME}/kernel/scripts/link${XLEN}.ld"

# Compiler flags for Vortex RV32IMF
LLVM_CFLAGS=(
    "--sysroot=${RISCV_SYSROOT}"
    "--gcc-toolchain=${RISCV_TOOLCHAIN_PATH}"
    "-Xclang" "-target-feature" "-Xclang" "+vortex"
    "-Xclang" "-target-feature" "-Xclang" "+zicond"
    "-mllvm" "-disable-loop-idiom-all"
)

VX_CFLAGS=(
    "-march=rv32imaf"
    "-mabi=ilp32f"
    "-O3"
    "-mcmodel=medany"
    "-fno-rtti"
    "-fno-exceptions"
    "-nostartfiles"
    "-nostdlib"
    "-fdata-sections"
    "-ffunction-sections"
    "-I${VORTEX_HOME}/kernel/include"
    "-I${VORTEX_HOME}/hw"
    "-I${REPO_ROOT}/test/kernels"
    "-DXLEN_32"
    "-DNDEBUG"
)

VX_LIBS=(
    "-L${LIBC_VORTEX}/lib"
    "-lm"
    "-lc"
    "${LIBCRT_VORTEX}/lib/baremetal/libclang_rt.builtins-riscv${XLEN}.a"
)

STARTUP_ADDR="0x80000000"
VX_LDFLAGS=(
    "-Wl,-Bstatic,--gc-sections,-T,${LINK_SCRIPT},--defsym=STARTUP_ADDR=${STARTUP_ADDR}"
    "${VORTEX_HOME}/kernel/libvortex.a"
    "${VX_LIBS[@]}"
)

# Parse arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <input.cir|input.c> <output.vxbin> [cflags...]"
    echo ""
    echo "Compiles CIRA IR or C/C++ source to Vortex RISC-V kernel binary"
    echo ""
    echo "Examples:"
    echo "  $0 kernel.cir kernel.vxbin"
    echo "  $0 kernel.c kernel.vxbin -DDEBUG"
    exit 1
fi

INPUT="$1"
OUTPUT="$2"
shift 2

EXTRA_CFLAGS=("$@")

if [ ! -f "${INPUT}" ]; then
    echo "Error: Input file '${INPUT}' not found"
    exit 1
fi

# Get file extension
EXT="${INPUT##*.}"
BASENAME="$(basename "${INPUT}" .${EXT})"
WORKDIR="$(dirname "${OUTPUT}")"
WORKDIR="${WORKDIR:-.}"

echo "[Vortex Kernel Compiler]"
echo "  Input:  ${INPUT}"
echo "  Output: ${OUTPUT}"
echo "  Target: RV32IMF (Vortex GPGPU)"
echo ""

# Add input file directory to include path
INPUT_DIR="$(cd "$(dirname "${INPUT}")" && pwd)"
VX_CFLAGS+=("-I${INPUT_DIR}")

# Temporary files
TEMP_C="${WORKDIR}/${BASENAME}.temp.c"
TEMP_LL="${WORKDIR}/${BASENAME}.temp.ll"
TEMP_S="${WORKDIR}/${BASENAME}.temp.s"
TEMP_O="${WORKDIR}/${BASENAME}.temp.o"
TEMP_ELF="${WORKDIR}/${BASENAME}.temp.elf"

cleanup() {
    rm -f "${TEMP_C}" "${TEMP_LL}" "${TEMP_S}" "${TEMP_O}" "${TEMP_ELF}"
}
trap cleanup EXIT

# Step 1: Convert input to C/C++ if needed
if [ "${EXT}" = "cir" ] || [ "${EXT}" = "mlir" ]; then
    echo "[1/5] Converting CIRA IR to LLVM IR..."
    if [ ! -x "${CIRA_BIN}" ]; then
        echo "Error: CIRA compiler not found at ${CIRA_BIN}"
        echo "Run 'cd ${REPO_ROOT} && ninja -C build' to build it"
        exit 1
    fi

    # Convert CIRA -> LLVM IR with Vortex target
    "${CIRA_BIN}" \
        -target-arch=vortex \
        -emit-llvm \
        "${INPUT}" \
        -o "${TEMP_LL}"

    INPUT_FOR_COMPILE="${TEMP_LL}"
elif [ "${EXT}" = "ll" ]; then
    echo "[1/5] Using LLVM IR input..."
    INPUT_FOR_COMPILE="${INPUT}"
elif [ "${EXT}" = "c" ] || [ "${EXT}" = "cpp" ] || [ "${EXT}" = "cc" ]; then
    echo "[1/5] Using C/C++ source input..."
    INPUT_FOR_COMPILE="${INPUT}"
else
    echo "Error: Unsupported input file extension: ${EXT}"
    echo "Supported: .cir, .mlir, .ll, .c, .cpp, .cc"
    exit 1
fi

# Step 2: Compile to RISC-V object code
echo "[2/5] Compiling to RISC-V assembly..."
if [ "${INPUT_FOR_COMPILE}" = "${INPUT}" ] && [ "${EXT}" = "ll" ]; then
    # LLVM IR -> assembly
    "${LLVM_VORTEX}/bin/llc" \
        -march=riscv32 \
        -mattr=+m,+a,+f,+vortex,+zicond \
        -O3 \
        "${INPUT_FOR_COMPILE}" \
        -o "${TEMP_S}"
else
    # C/C++ -> assembly
    "${VX_CXX}" \
        "${LLVM_CFLAGS[@]}" \
        "${VX_CFLAGS[@]}" \
        "${EXTRA_CFLAGS[@]}" \
        -S \
        "${INPUT_FOR_COMPILE}" \
        -o "${TEMP_S}"
fi

echo "[3/5] Assembling to object file..."
"${VX_CXX}" \
    "${LLVM_CFLAGS[@]}" \
    "${VX_CFLAGS[@]}" \
    -c \
    "${TEMP_S}" \
    -o "${TEMP_O}"

# Step 3: Link to ELF executable
echo "[4/5] Linking Vortex kernel ELF..."
"${VX_CXX}" \
    "${LLVM_CFLAGS[@]}" \
    "${VX_CFLAGS[@]}" \
    "${TEMP_O}" \
    "${VX_LDFLAGS[@]}" \
    -o "${TEMP_ELF}"

# Step 4: Convert ELF to .vxbin format
echo "[5/5] Generating Vortex binary (.vxbin)..."
OBJCOPY="${VX_CP}" python3 "${VXBIN_SCRIPT}" "${TEMP_ELF}" "${OUTPUT}"

# Also generate raw .bin for RTL simulator
BIN_FILE="${OUTPUT%.vxbin}.bin"
"${VX_CP}" -O binary "${TEMP_ELF}" "${BIN_FILE}"

# Generate disassembly for debugging
DUMP_FILE="${OUTPUT%.vxbin}.dump"
"${VX_DP}" -D "${TEMP_ELF}" > "${DUMP_FILE}"

echo ""
echo "✅ Compilation successful!"
echo "   Kernel binary: ${OUTPUT}"
echo "   RTL binary:    ${BIN_FILE}"
echo "   Disassembly:   ${DUMP_FILE}"
echo ""
echo "   Size: $(stat -c%s "${OUTPUT}") bytes"
