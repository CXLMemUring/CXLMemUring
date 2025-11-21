#!/bin/bash
# Vortex GPGPU RTL Simulation Setup Script
# Sets up Verilator-based cycle-accurate simulation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VORTEX_ROOT="${VORTEX_ROOT:-$PROJECT_ROOT/vortex}"
VERILATOR_VERSION="5.006"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."

    # Check for required tools
    local missing_tools=()

    if ! command -v git &> /dev/null; then
        missing_tools+=("git")
    fi

    if ! command -v make &> /dev/null; then
        missing_tools+=("make")
    fi

    if ! command -v gcc &> /dev/null; then
        missing_tools+=("gcc")
    fi

    if ! command -v g++ &> /dev/null; then
        missing_tools+=("g++")
    fi

    if [ ${#missing_tools[@]} -gt 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Install with: sudo apt-get install ${missing_tools[*]}"
        return 1
    fi

    log_success "All basic requirements met"
}

# Install Verilator
install_verilator() {
    log_info "Installing Verilator ${VERILATOR_VERSION}..."

    if command -v verilator &> /dev/null; then
        local installed_version=$(verilator --version | head -1 | awk '{print $2}')
        log_info "Verilator already installed: version $installed_version"
        return 0
    fi

    log_info "Installing dependencies..."
    sudo apt-get update
    sudo apt-get install -y \
        git perl python3 make autoconf g++ flex bison ccache \
        libgoogle-perftools-dev numactl perl-doc \
        libfl2 libfl-dev zlibc zlib1g zlib1g-dev

    log_info "Cloning Verilator..."
    cd /tmp
    git clone https://github.com/verilator/verilator
    cd verilator
    git checkout v${VERILATOR_VERSION}

    log_info "Building Verilator..."
    autoconf
    ./configure
    make -j$(nproc)
    sudo make install

    log_success "Verilator installed successfully"
    verilator --version
}

# Clone Vortex repository
clone_vortex() {
    log_info "Cloning Vortex GPGPU repository..."

    if [ -d "$VORTEX_ROOT" ]; then
        log_info "Vortex already exists at $VORTEX_ROOT"
        cd "$VORTEX_ROOT"
        git pull origin master || true
    else
        log_info "Cloning to $VORTEX_ROOT..."
        cd "$PROJECT_ROOT"
        git clone https://github.com/vortexgpgpu/vortex.git
        cd vortex
    fi

    log_success "Vortex repository ready"
}

# Install Vortex dependencies
install_vortex_deps() {
    log_info "Installing Vortex dependencies..."

    cd "$VORTEX_ROOT"

    # Install RISC-V toolchain
    if ! command -v riscv64-unknown-elf-gcc &> /dev/null; then
        log_info "Installing RISC-V toolchain..."
        sudo apt-get install -y gcc-riscv64-unknown-elf || {
            log_warn "Package not available, need to build from source"
            log_info "See: https://github.com/riscv/riscv-gnu-toolchain"
        }
    fi

    # Install POCL (Portable OpenCL)
    if ! command -v poclcc &> /dev/null; then
        log_info "Installing POCL..."
        sudo apt-get install -y \
            pocl-opencl-icd libpocl2 libpocl-dev \
            opencl-headers ocl-icd-opencl-dev
    fi

    log_success "Vortex dependencies installed"
}

# Build Vortex RTL simulator
build_vortex_rtl() {
    log_info "Building Vortex RTL simulator with Verilator..."

    cd "$VORTEX_ROOT"

    # Configure build
    log_info "Configuring Vortex..."
    export VERILATOR_ROOT=/usr/local/share/verilator
    export PATH=/usr/local/bin:$PATH
    ./configure
    # Build simulator
    log_info "Building RTL simulation..."
    make install
    log_success "Vortex RTL simulator built successfully"
}

# Verify installation
verify_installation() {
    log_info "Verifying Vortex installation..."

    local errors=0

    # Check Verilator
    if ! command -v verilator &> /dev/null; then
        log_error "Verilator not found in PATH"
        errors=$((errors + 1))
    else
        log_success "Verilator: $(verilator --version | head -1)"
    fi

    # # Check Vortex simulator
    # if [ -f "$VORTEX_ROOT/hw/rtl/obj_dir/Vvortex" ]; then
    #     log_success "Vortex RTL simulator: $VORTEX_ROOT/hw/rtl/obj_dir/Vvortex"
    # else
    #     log_error "Vortex RTL simulator not found"
    #     errors=$((errors + 1))
    # fi

    # Check runtime
    if [ -f "$VORTEX_ROOT/runtime/libvortex.a" ]; then
        log_success "Vortex runtime: $VORTEX_ROOT/runtime/libvortex.a"
    else
        log_warn "Vortex runtime not found (optional)"
    fi

    if [ $errors -gt 0 ]; then
        log_error "Verification failed with $errors error(s)"
        return 1
    fi

    log_success "Installation verified successfully!"
}

# Create test environment
setup_test_env() {
    log_info "Setting up test environment..."

    # Create environment file
    cat > "$SCRIPT_DIR/vortex_env.sh" <<EOF
#!/bin/bash
# Vortex RTL Simulation Environment

export VORTEX_ROOT="$VORTEX_ROOT"
export VERILATOR_ROOT="/usr/local/share/verilator"
export PATH="\$VORTEX_ROOT/bin:\$VERILATOR_ROOT/bin:\$PATH"
export LD_LIBRARY_PATH="\$VORTEX_ROOT/runtime:\$LD_LIBRARY_PATH"

# Vortex configuration
export VX_ENABLE_RTL_SIM=1
export VX_NUM_CORES=1
export VX_NUM_WARPS=4
export VX_NUM_THREADS=32

echo "Vortex RTL Simulation Environment"
echo "  VORTEX_ROOT: \$VORTEX_ROOT"
echo "  Cores: \$VX_NUM_CORES"
echo "  Warps: \$VX_NUM_WARPS"
echo "  Threads/Warp: \$VX_NUM_THREADS"
EOF

    chmod +x "$SCRIPT_DIR/vortex_env.sh"

    log_success "Test environment created: $SCRIPT_DIR/vortex_env.sh"
    log_info "Source with: source $SCRIPT_DIR/vortex_env.sh"
}

# Print usage instructions
print_usage() {
    cat <<EOF

${GREEN}╔════════════════════════════════════════════════════════════╗
║   Vortex RTL Simulation Setup Complete!                   ║
╚════════════════════════════════════════════════════════════╝${NC}

${BLUE}Next Steps:${NC}

1. ${YELLOW}Activate Vortex environment:${NC}
   source $SCRIPT_DIR/vortex_env.sh

2. ${YELLOW}Run a simple test:${NC}
   cd $VORTEX_ROOT/tests/opencl/vecadd
   make run

3. ${YELLOW}Run RTL simulation:${NC}
   cd $VORTEX_ROOT
   ./ci/blackbox.sh --driver=rtlsim --app=vecadd

4. ${YELLOW}Integrate with your tests:${NC}
   cd $SCRIPT_DIR
   ./run_vortex_rtl_tests.sh

${BLUE}Simulation Modes:${NC}
  - ${GREEN}RTL Simulation${NC}: Cycle-accurate, slow but accurate
  - ${GREEN}ISA Simulation${NC}: Functional, fast prototyping
  - ${GREEN}FPGA${NC}: Real hardware deployment

${BLUE}Documentation:${NC}
  Vortex: $VORTEX_ROOT/README.md
  Tests: $SCRIPT_DIR/README_TESTING.md

${BLUE}Troubleshooting:${NC}
  If issues occur, check:
  - Verilator version: verilator --version
  - Vortex build logs: $VORTEX_ROOT/hw/rtl/build.log
  - Environment: echo \$VORTEX_ROOT

EOF
}

# Main installation flow
main() {
    log_info "Starting Vortex RTL Simulation Setup"
    echo ""

    check_requirements || exit 1
    echo ""

    install_verilator
    echo ""

    clone_vortex
    echo ""

    install_vortex_deps
    echo ""

    build_vortex_rtl
    echo ""

    verify_installation || exit 1
    echo ""

    setup_test_env
    echo ""

    print_usage
}

# Parse command-line arguments
SKIP_BUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --vortex-root)
            VORTEX_ROOT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --skip-build         Skip Vortex build (if already built)"
            echo "  --vortex-root PATH   Specify Vortex installation directory"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main installation
main
