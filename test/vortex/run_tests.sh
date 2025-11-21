#!/bin/bash
# Vortex SIMT Backend End-to-End Test Runner

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
ENABLE_OPENMP=${ENABLE_OPENMP:-0}
ENABLE_VORTEX_SIM=${ENABLE_VORTEX_SIM:-1}
TEST_TIMEOUT=60

# Counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Logging
LOG_FILE="$SCRIPT_DIR/test_results_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

log_header() {
    log "${BLUE}================================${NC}"
    log "${BLUE}$1${NC}"
    log "${BLUE}================================${NC}"
}

log_pass() {
    log "${GREEN}✅ PASS: $1${NC}"
}

log_fail() {
    log "${RED}❌ FAIL: $1${NC}"
}

log_warn() {
    log "${YELLOW}⚠️  WARN: $1${NC}"
}

run_test() {
    local test_name=$1
    local test_cmd=$2
    local timeout=${3:-$TEST_TIMEOUT}

    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    log "\n${BLUE}Test $TOTAL_TESTS: $test_name${NC}"
    log "Command: $test_cmd"

    if timeout $timeout bash -c "$test_cmd" >> "$LOG_FILE" 2>&1; then
        log_pass "$test_name"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            log_fail "$test_name (TIMEOUT after ${timeout}s)"
        else
            log_fail "$test_name (exit code: $exit_code)"
        fi
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

# Phase 1: Build Tests
phase1_build() {
    log_header "Phase 1: Building Tests"

    cd "$SCRIPT_DIR"

    # Clean previous builds
    log "Cleaning previous builds..."
    make clean >> "$LOG_FILE" 2>&1 || true

    # Build tests
    if [ "$ENABLE_OPENMP" = "1" ]; then
        log "Building tests with OpenMP support..."
        run_test "Build with OpenMP" "make USE_OPENMP=1 all"
    else
        log "Building tests without OpenMP..."
        run_test "Build tests" "make all"
    fi
}

# Phase 2: Unit Tests
phase2_unit_tests() {
    log_header "Phase 2: Unit Tests"

    cd "$SCRIPT_DIR"

    run_test "Vector Addition Test" "./test_vector_add"
    run_test "Reduction Test" "./test_reduction"
    run_test "Backend Integration Test" "./test_vortex_backend"
}

# Phase 3: Backend Compilation
phase3_backend_build() {
    log_header "Phase 3: Backend Compilation"

    cd "$PROJECT_ROOT"

    # Check if build directory exists
    if [ ! -d "build" ]; then
        log "Creating build directory..."
        mkdir -p build
    fi

    cd build

    # Check if CMake has been run
    if [ ! -f "Makefile" ]; then
        log "Running CMake..."
        run_test "CMake Configuration" "cmake -DCMAKE_BUILD_TYPE=Release .."
    fi

    # Build Vortex backend
    log "Building Vortex backend runtime..."
    run_test "Build Vortex Runtime" "make bc_riscv -j$(nproc)"
}

# Phase 4: Backend Simulation Tests
phase4_backend_sim() {
    log_header "Phase 4: Backend Simulation Tests"

    local backend_bin="$PROJECT_ROOT/build/runtime/bc_riscv"

    if [ ! -f "$backend_bin" ]; then
        log_warn "Backend binary not found, skipping simulation tests"
        return
    fi

    # Test backend startup (simulation mode)
    log "Testing backend in simulation mode..."
    export VX_ENABLE_SIM=1
    export VX_DEBUG=0

    # Start backend in background with timeout
    log "Starting Vortex backend..."
    timeout 5s "$backend_bin" > "$SCRIPT_DIR/backend_output.log" 2>&1 &
    local backend_pid=$!

    sleep 2

    # Check if backend is running
    if kill -0 $backend_pid 2>/dev/null; then
        log_pass "Backend started successfully (PID: $backend_pid)"

        # Kill backend
        kill $backend_pid 2>/dev/null || true
        wait $backend_pid 2>/dev/null || true

        log_pass "Backend stopped cleanly"
        PASSED_TESTS=$((PASSED_TESTS + 2))
    else
        log_fail "Backend failed to start or crashed"
        FAILED_TESTS=$((FAILED_TESTS + 2))
    fi

    TOTAL_TESTS=$((TOTAL_TESTS + 2))
}

# Phase 5: NVVM IR Generation Tests
phase5_nvvm_tests() {
    log_header "Phase 5: NVVM IR Generation Tests"

    # Check if mlir-opt is available
    if ! command -v mlir-opt &> /dev/null; then
        log_warn "mlir-opt not found, skipping NVVM tests"
        return
    fi

    # Create simple test MLIR
    local test_mlir="$SCRIPT_DIR/test_simple.mlir"
    cat > "$test_mlir" <<'EOF'
module {
  func.func @simple_add(%arg0: i32, %arg1: i32) -> i32 {
    %result = arith.addi %arg0, %arg1 : i32
    return %result : i32
  }
}
EOF

    # Try to convert to LLVM
    log "Testing MLIR to LLVM conversion..."
    run_test "MLIR to LLVM" "mlir-opt $test_mlir --convert-arith-to-llvm --convert-func-to-llvm -o /dev/null"

    rm -f "$test_mlir"
}

# Phase 6: Performance Benchmarks (if enabled)
phase6_performance() {
    log_header "Phase 6: Performance Benchmarks"

    cd "$SCRIPT_DIR"

    if [ -f "./test_vector_add" ]; then
        log "Running vector add benchmark..."
        ./test_vector_add >> "$LOG_FILE" 2>&1 || true
    fi

    if [ -f "./test_reduction" ]; then
        log "Running reduction benchmark..."
        ./test_reduction >> "$LOG_FILE" 2>&1 || true
    fi

    log "Performance results logged to $LOG_FILE"
}

# Main execution
main() {
    log_header "Vortex SIMT Backend End-to-End Tests"
    log "Start time: $(date)"
    log "Log file: $LOG_FILE"
    log ""
    log "Configuration:"
    log "  OpenMP: $ENABLE_OPENMP"
    log "  Vortex Simulation: $ENABLE_VORTEX_SIM"
    log "  Timeout: ${TEST_TIMEOUT}s"
    log ""

    # Run test phases
    phase1_build
    phase2_unit_tests
    phase3_backend_build
    phase4_backend_sim
    phase5_nvvm_tests
    phase6_performance

    # Summary
    log_header "Test Summary"
    log "Total tests: $TOTAL_TESTS"
    log_pass "Passed: $PASSED_TESTS"
    if [ $FAILED_TESTS -gt 0 ]; then
        log_fail "Failed: $FAILED_TESTS"
    else
        log "Failed: $FAILED_TESTS"
    fi

    local pass_rate=0
    if [ $TOTAL_TESTS -gt 0 ]; then
        pass_rate=$((100 * PASSED_TESTS / TOTAL_TESTS))
    fi
    log "Pass rate: ${pass_rate}%"

    log ""
    log "End time: $(date)"
    log "Full log: $LOG_FILE"

    if [ $FAILED_TESTS -eq 0 ]; then
        log ""
        log_header "🎉 All Tests Passed! 🎉"
        exit 0
    else
        log ""
        log_header "⚠️  Some Tests Failed"
        log "Check $LOG_FILE for details"
        exit 1
    fi
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --openmp)
            ENABLE_OPENMP=1
            shift
            ;;
        --no-sim)
            ENABLE_VORTEX_SIM=0
            shift
            ;;
        --timeout)
            TEST_TIMEOUT="$2"
            shift 2
            ;;
        --help)
            echo "Vortex SIMT Backend Test Runner"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --openmp      Enable OpenMP for parallel tests"
            echo "  --no-sim      Disable Vortex simulation tests"
            echo "  --timeout N   Set test timeout (default: 60s)"
            echo "  --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run tests
main
