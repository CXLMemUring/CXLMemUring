// End-to-End Profiled Example
// Runs kernel with CPU-side timing for prefetcher distance optimization

#include "../vortex_device.h"
#include "../offload_profiler.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>

#define CHECK(call) do { \
    int ret = (call); \
    if (ret != VORTEX_SUCCESS) { \
        std::cerr << "Error: " #call " failed with " << ret << std::endl; \
        return 1; \
    } \
} while(0)

int main(int argc, char** argv) {
    // Configuration
    size_t num_elements = 1024;
    const char* kernel_path = "/root/CXLMemUring/test/kernels/simple_add.vxbin";
    const char* profile_output = "profile_output.json";

    if (argc > 1) num_elements = atoi(argv[1]);
    if (argc > 2) kernel_path = argv[2];
    if (argc > 3) profile_output = argv[3];

    size_t array_size = num_elements * sizeof(float);

    std::cout << "=== Vortex E2E Profiled Execution ===\n";
    std::cout << "Elements: " << num_elements << "\n";
    std::cout << "Array size: " << array_size << " bytes\n";
    std::cout << "Kernel: " << kernel_path << "\n\n";

    // Initialize profiling context
    offload_profile_ctx_t profile_ctx;
    offload_timing_t timing;
    offload_profile_start(&profile_ctx);

    // Initialize device
    vortex_device_h device;
    CHECK(vortex_device_init(&device, "/root/CXLMemUring/vortex/sim/rtlsim/rtlsim"));

    // Prepare host data
    std::vector<float> h_a(num_elements);
    std::vector<float> h_b(num_elements);
    std::vector<float> h_c(num_elements);

    for (size_t i = 0; i < num_elements; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    // Allocate device memory
    vortex_buffer_h d_a, d_b, d_c;
    CHECK(vortex_malloc(device, &d_a, array_size));
    CHECK(vortex_malloc(device, &d_b, array_size));
    CHECK(vortex_malloc(device, &d_c, array_size));

    // === H2D Transfer (Profiled) ===
    offload_profile_h2d_start(&profile_ctx, array_size * 2);
    CHECK(vortex_memcpy(d_a, h_a.data(), array_size, VORTEX_MEMCPY_HOST_TO_DEVICE));
    CHECK(vortex_memcpy(d_b, h_b.data(), array_size, VORTEX_MEMCPY_HOST_TO_DEVICE));
    offload_profile_h2d_end(&profile_ctx);

    // Load kernel
    vortex_kernel_h kernel;
    CHECK(vortex_kernel_load_file(device, &kernel, kernel_path));

    // Set kernel arguments
    struct {
        uint64_t a_addr;
        uint64_t b_addr;
        uint64_t c_addr;
        uint32_t num_elements;
    } kernel_args;

    vortex_buffer_get_address(d_a, &kernel_args.a_addr);
    vortex_buffer_get_address(d_b, &kernel_args.b_addr);
    vortex_buffer_get_address(d_c, &kernel_args.c_addr);
    kernel_args.num_elements = num_elements;

    vortex_kernel_arg_t args[] = {
        { &kernel_args, sizeof(kernel_args), 1, VORTEX_ARG_VALUE }
    };
    CHECK(vortex_kernel_set_args(kernel, args, 1));

    // === Kernel Execution (Profiled) ===
    vortex_launch_params_t launch_params = {1, 1, 1, (uint32_t)num_elements, 1, 1, 0};

    offload_profile_kernel_start(&profile_ctx, num_elements);
    CHECK(vortex_kernel_launch(device, kernel, &launch_params));
    offload_profile_kernel_end(&profile_ctx);

    // === D2H Transfer (Profiled) ===
    offload_profile_d2h_start(&profile_ctx, array_size);
    CHECK(vortex_memcpy(h_c.data(), d_c, array_size, VORTEX_MEMCPY_DEVICE_TO_HOST));
    offload_profile_d2h_end(&profile_ctx);

    // Calculate timing results
    offload_profile_finish(&profile_ctx, &timing);

    // Verify results
    bool pass = true;
    for (size_t i = 0; i < num_elements; i++) {
        float expected = h_a[i] + h_b[i];
        if (std::fabs(h_c[i] - expected) > 0.001f) {
            std::cerr << "Mismatch at " << i << ": " << h_c[i] << " != " << expected << "\n";
            pass = false;
            break;
        }
    }

    if (pass) {
        std::cout << "Result verification: PASSED\n";
    } else {
        std::cout << "Result verification: FAILED\n";
    }

    // Print and save profiling results
    offload_profile_print(&timing, "vector_add");
    offload_profile_to_json(&timing, "vector_add", profile_output);
    std::cout << "Profile saved to: " << profile_output << "\n";

    // Cleanup
    vortex_kernel_unload(kernel);
    vortex_free(d_c);
    vortex_free(d_b);
    vortex_free(d_a);
    vortex_device_destroy(device);

    return pass ? 0 : 1;
}
