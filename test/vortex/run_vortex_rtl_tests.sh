#!/bin/bash
# Run Vortex RTL Simulation Tests
# Executes kernels on cycle-accurate Verilator simulation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VORTEX_ROOT="${VORTEX_ROOT:-$SCRIPT_DIR/../../vortex}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Test results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Check if Vortex is set up
check_vortex_setup() {
    log_info "Checking Vortex RTL simulation environment..."

    if [ ! -d "$VORTEX_ROOT" ]; then
        log_error "Vortex not found at $VORTEX_ROOT"
        log_info "Run setup first: ./setup_vortex_simulation.sh"
        exit 1
    fi

    if ! command -v verilator &> /dev/null; then
        log_error "Verilator not installed"
        log_info "Run setup first: ./setup_vortex_simulation.sh"
        exit 1
    fi

    # Check if RTL simulator is built
    if [ ! -f "$VORTEX_ROOT/hw/rtl/obj_dir/Vvortex" ]; then
        log_warn "Vortex RTL simulator not built"
        log_info "Building now..."
        cd "$VORTEX_ROOT"
        make -C hw/rtl || {
            log_error "Failed to build RTL simulator"
            exit 1
        }
    fi

    log_success "Vortex environment ready"
}

# Compile OpenCL kernel to Vortex ISA
compile_kernel() {
    local kernel_file=$1
    local output_name=$2

    log_info "Compiling kernel: $kernel_file"

    cd "$VORTEX_ROOT"

    # Use Vortex's OpenCL compiler
    ./ci/toolchain_install.sh || true

    # Compile kernel
    ./runtime/vxcc -O3 "$kernel_file" -o "$output_name" || {
        log_error "Kernel compilation failed"
        return 1
    }

    log_success "Kernel compiled: $output_name"
    return 0
}

# Run kernel on RTL simulator
run_rtl_simulation() {
    local kernel_binary=$1
    local test_name=$2

    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    log_info "Running RTL simulation: $test_name"

    cd "$VORTEX_ROOT"

    # Set simulation parameters
    export VX_NUM_CORES=1
    export VX_NUM_WARPS=4
    export VX_NUM_THREADS=32

    # Run simulation
    timeout 300 ./hw/rtl/obj_dir/Vvortex "$kernel_binary" > "$SCRIPT_DIR/rtl_output_$test_name.log" 2>&1

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        log_success "$test_name PASSED"
        PASSED_TESTS=$((PASSED_TESTS + 1))

        # Extract performance metrics
        if grep -q "cycles" "$SCRIPT_DIR/rtl_output_$test_name.log"; then
            local cycles=$(grep "cycles" "$SCRIPT_DIR/rtl_output_$test_name.log" | awk '{print $NF}')
            log_info "Performance: $cycles cycles"
        fi

        return 0
    elif [ $exit_code -eq 124 ]; then
        log_error "$test_name TIMEOUT"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    else
        log_error "$test_name FAILED (exit code: $exit_code)"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

# Test 1: Vector Addition
test_vector_add() {
    log_info "=== Test 1: Vector Addition RTL Simulation ==="

    local kernel="$SCRIPT_DIR/kernels/vector_add.cl"
    local binary="$SCRIPT_DIR/build/vector_add.vxbin"

    if [ ! -f "$kernel" ]; then
        log_error "Kernel not found: $kernel"
        return 1
    fi

    mkdir -p "$SCRIPT_DIR/build"

    # Compile kernel
    if ! compile_kernel "$kernel" "$binary"; then
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi

    # Run simulation
    run_rtl_simulation "$binary" "vector_add"
}

# Test 2: Parallel Reduction
test_reduction() {
    log_info "=== Test 2: Parallel Reduction RTL Simulation ==="

    local kernel="$SCRIPT_DIR/kernels/reduction.cl"
    local binary="$SCRIPT_DIR/build/reduction.vxbin"

    if [ ! -f "$kernel" ]; then
        log_error "Kernel not found: $kernel"
        return 1
    fi

    mkdir -p "$SCRIPT_DIR/build"

    # Compile kernel
    if ! compile_kernel "$kernel" "$binary"; then
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi

    # Run simulation
    run_rtl_simulation "$binary" "reduction"
}

# Test 3: Matrix Multiplication
test_matmul() {
    log_info "=== Test 3: Matrix Multiplication RTL Simulation ==="

    local kernel="$SCRIPT_DIR/kernels/matmul.cl"
    local binary="$SCRIPT_DIR/build/matmul.vxbin"

    if [ ! -f "$kernel" ]; then
        log_error "Kernel not found: $kernel"
        return 1
    fi

    mkdir -p "$SCRIPT_DIR/build"

    # Compile kernel
    if ! compile_kernel "$kernel" "$binary"; then
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi

    # Run simulation
    run_rtl_simulation "$binary" "matmul"
}

# Run sample Vortex tests
run_vortex_samples() {
    log_info "=== Running Vortex Sample Tests ==="

    cd "$VORTEX_ROOT"

    # Run basic vectoradd test
    log_info "Running vecadd sample..."
    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    if ./ci/blackbox.sh --driver=rtlsim --app=vecadd --cores=1 --warps=4 --threads=32; then
        log_success "vecadd sample PASSED"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        log_error "vecadd sample FAILED"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
}

# Generate performance report
generate_report() {
    log_info "Generating performance report..."

    local report_file="$SCRIPT_DIR/vortex_rtl_performance_report.txt"

    cat > "$report_file" <<EOF
Vortex RTL Simulation Performance Report
Generated: $(date)

Configuration:
  Cores: $VX_NUM_CORES
  Warps per Core: $VX_NUM_WARPS
  Threads per Warp: $VX_NUM_THREADS
  Total Threads: $((VX_NUM_CORES * VX_NUM_WARPS * VX_NUM_THREADS))

Test Results:
  Total Tests: $TOTAL_TESTS
  Passed: $PASSED_TESTS
  Failed: $FAILED_TESTS
  Pass Rate: $(( TOTAL_TESTS > 0 ? (100 * PASSED_TESTS / TOTAL_TESTS) : 0 ))%

Performance Metrics:
EOF

    # Extract cycle counts from logs
    for log_file in "$SCRIPT_DIR"/rtl_output_*.log; do
        if [ -f "$log_file" ]; then
            local test_name=$(basename "$log_file" .log | sed 's/rtl_output_//')
            echo "  $test_name:" >> "$report_file"

            if grep -q "cycles" "$log_file"; then
                grep "cycles\|instructions\|IPC" "$log_file" | sed 's/^/    /' >> "$report_file"
            else
                echo "    No performance data available" >> "$report_file"
            fi
        fi
    done

    cat "$report_file"
    log_success "Report saved: $report_file"
}

# Main execution
main() {
    echo ""
    log_info "╔═══════════════════════════════════════════════════════╗"
    log_info "║   Vortex RTL Simulation Tests                        ║"
    log_info "║   Cycle-Accurate Verilator Simulation                ║"
    log_info "╚═══════════════════════════════════════════════════════╝"
    echo ""

    check_vortex_setup
    echo ""

    # Run tests
    test_vector_add || true
    echo ""

    test_reduction || true
    echo ""

    test_matmul || true
    echo ""

    run_vortex_samples || true
    echo ""

    # Summary
    log_info "═══════════════════════════════════════════════════════"
    log_info "Test Summary"
    log_info "═══════════════════════════════════════════════════════"
    log_info "Total Tests: $TOTAL_TESTS"
    log_success "Passed: $PASSED_TESTS"

    if [ $FAILED_TESTS -gt 0 ]; then
        log_error "Failed: $FAILED_TESTS"
    else
        log_info "Failed: $FAILED_TESTS"
    fi

    echo ""
    generate_report

    if [ $FAILED_TESTS -eq 0 ] && [ $TOTAL_TESTS -gt 0 ]; then
        echo ""
        log_success "🎉 All RTL simulation tests passed!"
        exit 0
    else
        echo ""
        log_warn "⚠️  Some tests failed or no tests ran"
        log_info "Check logs in: $SCRIPT_DIR/rtl_output_*.log"
        exit 1
    fi
}

# Run tests
main
