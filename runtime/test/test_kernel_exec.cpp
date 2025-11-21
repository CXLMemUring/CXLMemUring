// End-to-End Test: Compile and Execute Kernel on Vortex RTL Simulator
// Tests the complete pipeline: kernel compilation -> loading -> execution -> verification

#include "../vortex_device.h"
#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <vector>

// Colors for output
#define COLOR_RESET   "\033[0m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_RED     "\033[31m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_CYAN    "\033[36m"

void print_header(const char* test_name) {
    std::cout << "\n" << COLOR_CYAN << "═══════════════════════════════════════════════════════"
              << COLOR_RESET << std::endl;
    std::cout << COLOR_CYAN << "  " << test_name << COLOR_RESET << std::endl;
    std::cout << COLOR_CYAN << "═══════════════════════════════════════════════════════"
              << COLOR_RESET << std::endl;
}

bool verify_float_array(const float* result, const float* expected, size_t count,
                        float epsilon = 0.0001f) {
    for (size_t i = 0; i < count; i++) {
        if (std::fabs(result[i] - expected[i]) > epsilon) {
            std::cerr << COLOR_RED << "❌ Mismatch at index " << i << ": "
                      << "got " << result[i] << ", expected " << expected[i]
                      << COLOR_RESET << std::endl;
            return false;
        }
    }
    return true;
}

int test_vector_addition() {
    print_header("Test: Vector Addition Kernel Execution");

    const char* rtlsim_path = "/root/CXLMemUring/vortex/sim/rtlsim/rtlsim";
    const char* kernel_path = "/root/CXLMemUring/test/kernels/simple_add.vxbin";

    // Test configuration
    constexpr size_t NUM_ELEMENTS = 256;
    constexpr size_t ARRAY_SIZE = NUM_ELEMENTS * sizeof(float);

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Elements:     " << NUM_ELEMENTS << std::endl;
    std::cout << "  Array size:   " << ARRAY_SIZE << " bytes" << std::endl;
    std::cout << "  RTL sim:      " << rtlsim_path << std::endl;
    std::cout << "  Kernel:       " << kernel_path << std::endl;
    std::cout << std::endl;

    // Initialize device
    std::cout << "[1/8] Initializing Vortex device..." << std::endl;
    vortex_device_h device;
    int ret = vortex_device_init(&device, rtlsim_path);
    if (ret != VORTEX_SUCCESS) {
        std::cerr << COLOR_RED << "❌ Device initialization failed" << COLOR_RESET << std::endl;
        return 1;
    }

    vortex_device_caps_t caps;
    vortex_device_get_caps(device, &caps);

    std::cout << "  Cores:        " << caps.num_cores << std::endl;
    std::cout << "  Warps:        " << caps.num_warps_per_core << std::endl;
    std::cout << "  Threads:      " << caps.num_threads_per_warp << std::endl;
    std::cout << "  Total:        " << (caps.num_cores * caps.num_warps_per_core * caps.num_threads_per_warp) << " hardware threads" << std::endl;
    std::cout << std::endl;

    // Prepare host data
    std::cout << "[2/8] Preparing host data..." << std::endl;
    std::vector<float> h_a(NUM_ELEMENTS);
    std::vector<float> h_b(NUM_ELEMENTS);
    std::vector<float> h_c(NUM_ELEMENTS);
    std::vector<float> h_expected(NUM_ELEMENTS);

    for (size_t i = 0; i < NUM_ELEMENTS; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
        h_expected[i] = h_a[i] + h_b[i];
    }
    std::cout << "  Initialized " << NUM_ELEMENTS << " elements" << std::endl;
    std::cout << "  Example: " << h_a[0] << " + " << h_b[0] << " = " << h_expected[0] << std::endl;
    std::cout << std::endl;

    // Allocate device buffers
    std::cout << "[3/8] Allocating device memory..." << std::endl;
    vortex_buffer_h d_a, d_b, d_c;

    ret = vortex_malloc(device, &d_a, ARRAY_SIZE);
    if (ret != VORTEX_SUCCESS) {
        std::cerr << COLOR_RED << "❌ Failed to allocate buffer A" << COLOR_RESET << std::endl;
        goto cleanup_device;
    }

    ret = vortex_malloc(device, &d_b, ARRAY_SIZE);
    if (ret != VORTEX_SUCCESS) {
        std::cerr << COLOR_RED << "❌ Failed to allocate buffer B" << COLOR_RESET << std::endl;
        goto cleanup_a;
    }

    ret = vortex_malloc(device, &d_c, ARRAY_SIZE);
    if (ret != VORTEX_SUCCESS) {
        std::cerr << COLOR_RED << "❌ Failed to allocate buffer C" << COLOR_RESET << std::endl;
        goto cleanup_b;
    }

    std::cout << "  Allocated 3 buffers × " << ARRAY_SIZE << " bytes" << std::endl;
    std::cout << std::endl;

    // Transfer data to device
    std::cout << "[4/8] Transferring data to device..." << std::endl;
    ret = vortex_memcpy(d_a, h_a.data(), ARRAY_SIZE, VORTEX_MEMCPY_HOST_TO_DEVICE);
    if (ret != VORTEX_SUCCESS) {
        std::cerr << COLOR_RED << "❌ Failed to copy buffer A" << COLOR_RESET << std::endl;
        goto cleanup_c;
    }

    ret = vortex_memcpy(d_b, h_b.data(), ARRAY_SIZE, VORTEX_MEMCPY_HOST_TO_DEVICE);
    if (ret != VORTEX_SUCCESS) {
        std::cerr << COLOR_RED << "❌ Failed to copy buffer B" << COLOR_RESET << std::endl;
        goto cleanup_c;
    }

    std::cout << "  Transferred " << (2 * ARRAY_SIZE) << " bytes to device" << std::endl;
    std::cout << std::endl;

    // Load kernel
    std::cout << "[5/8] Loading kernel from file..." << std::endl;
    vortex_kernel_h kernel;
    ret = vortex_kernel_load_file(device, &kernel, kernel_path);
    if (ret != VORTEX_SUCCESS) {
        std::cerr << COLOR_RED << "❌ Failed to load kernel" << COLOR_RESET << std::endl;
        goto cleanup_c;
    }
    std::cout << "  Kernel loaded successfully" << std::endl;
    std::cout << std::endl;

    // Set kernel arguments
    std::cout << "[6/8] Setting kernel arguments..." << std::endl;
    {
        // Kernel argument structure (must match simple_add.c kernel_arg_t)
        struct {
            uint64_t a_addr;
            uint64_t b_addr;
            uint64_t c_addr;
            uint32_t num_elements;
        } kernel_args;

        vortex_buffer_get_address(d_a, &kernel_args.a_addr);
        vortex_buffer_get_address(d_b, &kernel_args.b_addr);
        vortex_buffer_get_address(d_c, &kernel_args.c_addr);
        kernel_args.num_elements = NUM_ELEMENTS;

        vortex_kernel_arg_t args[] = {
            { &kernel_args, sizeof(kernel_args), 1, VORTEX_ARG_VALUE }
        };

        ret = vortex_kernel_set_args(kernel, args, 1);
        if (ret != VORTEX_SUCCESS) {
            std::cerr << COLOR_RED << "❌ Failed to set kernel arguments" << COLOR_RESET << std::endl;
            vortex_kernel_unload(kernel);
            goto cleanup_c;
        }

        std::cout << "  Arguments:" << std::endl;
        std::cout << "    A address:    0x" << std::hex << kernel_args.a_addr << std::dec << std::endl;
        std::cout << "    B address:    0x" << std::hex << kernel_args.b_addr << std::dec << std::endl;
        std::cout << "    C address:    0x" << std::hex << kernel_args.c_addr << std::dec << std::endl;
        std::cout << "    Num elements: " << kernel_args.num_elements << std::endl;
    }
    std::cout << std::endl;

    // Launch kernel
    std::cout << "[7/8] Launching kernel on Vortex RTL simulator..." << std::endl;
    vortex_launch_params_t launch_params;
    launch_params.grid_dim_x = 1;
    launch_params.grid_dim_y = 1;
    launch_params.grid_dim_z = 1;
    launch_params.block_dim_x = NUM_ELEMENTS;  // Launch enough threads
    launch_params.block_dim_y = 1;
    launch_params.block_dim_z = 1;
    launch_params.shared_mem_bytes = 0;

    std::cout << "  Grid:  " << launch_params.grid_dim_x << " × "
              << launch_params.grid_dim_y << " × " << launch_params.grid_dim_z << std::endl;
    std::cout << "  Block: " << launch_params.block_dim_x << " × "
              << launch_params.block_dim_y << " × " << launch_params.block_dim_z << std::endl;

    ret = vortex_kernel_launch(device, kernel, &launch_params);
    if (ret != VORTEX_SUCCESS) {
        std::cerr << COLOR_RED << "❌ Kernel launch failed" << COLOR_RESET << std::endl;
        vortex_kernel_unload(kernel);
        goto cleanup_c;
    }
    std::cout << COLOR_GREEN << "  ✓ Kernel executed successfully" << COLOR_RESET << std::endl;
    std::cout << std::endl;

    // Read back results
    std::cout << "[8/8] Reading results from device..." << std::endl;
    ret = vortex_memcpy(h_c.data(), d_c, ARRAY_SIZE, VORTEX_MEMCPY_DEVICE_TO_HOST);
    if (ret != VORTEX_SUCCESS) {
        std::cerr << COLOR_RED << "❌ Failed to read results" << COLOR_RESET << std::endl;
        vortex_kernel_unload(kernel);
        goto cleanup_c;
    }
    std::cout << "  Retrieved " << ARRAY_SIZE << " bytes from device" << std::endl;
    std::cout << std::endl;

    // Verify results
    std::cout << "Verifying results..." << std::endl;
    if (verify_float_array(h_c.data(), h_expected.data(), NUM_ELEMENTS)) {
        std::cout << COLOR_GREEN << "✅ PASS: All " << NUM_ELEMENTS
                  << " elements verified correctly!" << COLOR_RESET << std::endl;

        // Show sample results
        std::cout << "\nSample results:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), NUM_ELEMENTS); i++) {
            std::cout << "  [" << i << "] " << h_a[i] << " + " << h_b[i]
                      << " = " << h_c[i] << " ✓" << std::endl;
        }
        vortex_kernel_unload(kernel);
        vortex_free(d_c);
        vortex_free(d_b);
        vortex_free(d_a);
        vortex_device_destroy(device);
        return 0;
    } else {
        std::cout << COLOR_RED << "❌ FAIL: Result verification failed" << COLOR_RESET << std::endl;
        vortex_kernel_unload(kernel);
        goto cleanup_c;
    }

cleanup_c:
    vortex_free(d_c);
cleanup_b:
    vortex_free(d_b);
cleanup_a:
    vortex_free(d_a);
cleanup_device:
    vortex_device_destroy(device);
    return 1;
}

int main() {
    std::cout << COLOR_CYAN << R"(
╔═══════════════════════════════════════════════════════════════╗
║     Vortex RISC-V SIMT Kernel Execution Test Suite          ║
║     End-to-End: Compile → Load → Execute → Verify           ║
╚═══════════════════════════════════════════════════════════════╝
)" << COLOR_RESET << std::endl;

    int failures = 0;

    failures += test_vector_addition();

    std::cout << "\n" << COLOR_CYAN << "═══════════════════════════════════════════════════════"
              << COLOR_RESET << std::endl;
    if (failures == 0) {
        std::cout << COLOR_GREEN << "  ALL TESTS PASSED ✅" << COLOR_RESET << std::endl;
    } else {
        std::cout << COLOR_RED << "  " << failures << " TEST(S) FAILED ❌" << COLOR_RESET << std::endl;
    }
    std::cout << COLOR_CYAN << "═══════════════════════════════════════════════════════"
              << COLOR_RESET << std::endl;

    return failures;
}
