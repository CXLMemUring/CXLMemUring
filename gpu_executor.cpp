/**
 * GPU Executor for CXLMemUring
 *
 * Host-side harness that executes offloaded operations on Vortex GPU.
 * Integrates with Vortex runtime and CXL coherent memory.
 *
 * Compilation:
 *   g++ -o gpu_executor gpu_executor.cpp -I/home/victoryang00/vortex/runtime/include \
 *       -L/home/victoryang00/vortex/build/sim/rtlsim -lvortex_simx
 *
 * Execution:
 *   ./gpu_executor kernel.vxbin data.bin
 */

#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <chrono>

// Vortex runtime API
extern "C" {
    typedef void* vx_device_h;
    typedef void* vx_buffer_h;

    // Device management
    int vx_dev_open(vx_device_h* device);
    int vx_dev_close(vx_device_h device);

    // Memory management
    int vx_mem_alloc(vx_device_h device, uint64_t size, int flags, vx_buffer_h* buf);
    int vx_mem_free(vx_buffer_h buf);
    int vx_mem_address(vx_buffer_h buf, uint64_t* addr);

    // Data transfer
    int vx_copy_to_dev(vx_buffer_h buf, const void* host_data, uint64_t offset, uint64_t size);
    int vx_copy_from_dev(void* host_data, vx_buffer_h buf, uint64_t offset, uint64_t size);

    // Kernel management
    int vx_upload_kernel_file(vx_device_h device, const char* filename, vx_buffer_h* buf);
    int vx_upload_bytes(vx_device_h device, const void* data, uint64_t size, vx_buffer_h* buf);

    // Execution
    int vx_start(vx_device_h device, vx_buffer_h kernel, vx_buffer_h args);
    int vx_ready_wait(vx_device_h device, uint64_t timeout);
}

// Memory flags
#define VX_MEM_READ  0
#define VX_MEM_WRITE 1
#define VX_MAX_TIMEOUT 0xFFFFFFFF

// Kernel argument structure (must match GPU side)
typedef struct {
    uint64_t src0_addr;
    uint64_t src1_addr;
    uint64_t dst_addr;
    uint32_t M, K, N;
    uint32_t num_points;
} kernel_arg_t;

// Error checking macro
#define GPU_CHECK(expr) do { \
    int ret = (expr); \
    if (ret != 0) { \
        std::cerr << "GPU Error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::cerr << "  Expression: " << #expr << std::endl; \
        std::cerr << "  Error code: " << ret << std::endl; \
        cleanup(); \
        exit(-1); \
    } \
} while(false)

// Global state
vx_device_h g_device = nullptr;
vx_buffer_h g_src0_buf = nullptr;
vx_buffer_h g_src1_buf = nullptr;
vx_buffer_h g_dst_buf = nullptr;
vx_buffer_h g_kernel_buf = nullptr;
vx_buffer_h g_args_buf = nullptr;

void cleanup() {
    if (g_device) {
        if (g_src0_buf) vx_mem_free(g_src0_buf);
        if (g_src1_buf) vx_mem_free(g_src1_buf);
        if (g_dst_buf) vx_mem_free(g_dst_buf);
        if (g_kernel_buf) vx_mem_free(g_kernel_buf);
        if (g_args_buf) vx_mem_free(g_args_buf);
        vx_dev_close(g_device);
    }
}

/**
 * Execute MatMul on GPU: C = A * B
 *
 * @param A: Input matrix A (M x K)
 * @param B: Input matrix B (K x N)
 * @param C: Output matrix C (M x N)
 * @param M: Rows of A
 * @param K: Columns of A / Rows of B
 * @param N: Columns of B
 * @param kernel_file: Path to compiled Vortex kernel
 */
void execute_matmul_gpu(const std::vector<float>& A,
                        const std::vector<float>& B,
                        std::vector<float>& C,
                        uint32_t M, uint32_t K, uint32_t N,
                        const char* kernel_file) {
    std::cout << "\n=== GPU MatMul Execution ===" << std::endl;
    std::cout << "Dimensions: " << M << "x" << K << " * " << K << "x" << N << std::endl;

    // Validate input
    if (A.size() != M * K || B.size() != K * N || C.size() != M * N) {
        std::cerr << "ERROR: Invalid matrix dimensions" << std::endl;
        return;
    }

    // Open device
    std::cout << "Opening GPU device..." << std::endl;
    GPU_CHECK(vx_dev_open(&g_device));

    uint64_t buf_size_A = M * K * sizeof(float);
    uint64_t buf_size_B = K * N * sizeof(float);
    uint64_t buf_size_C = M * N * sizeof(float);

    std::cout << "Allocating GPU memory..." << std::endl;
    std::cout << "  Buffer A: " << buf_size_A << " bytes" << std::endl;
    std::cout << "  Buffer B: " << buf_size_B << " bytes" << std::endl;
    std::cout << "  Buffer C: " << buf_size_C << " bytes" << std::endl;

    // Allocate GPU memory
    GPU_CHECK(vx_mem_alloc(g_device, buf_size_A, VX_MEM_READ, &g_src0_buf));
    GPU_CHECK(vx_mem_alloc(g_device, buf_size_B, VX_MEM_READ, &g_src1_buf));
    GPU_CHECK(vx_mem_alloc(g_device, buf_size_C, VX_MEM_WRITE, &g_dst_buf));

    // Get device addresses
    uint64_t dev_addr_A = 0, dev_addr_B = 0, dev_addr_C = 0;
    GPU_CHECK(vx_mem_address(g_src0_buf, &dev_addr_A));
    GPU_CHECK(vx_mem_address(g_src1_buf, &dev_addr_B));
    GPU_CHECK(vx_mem_address(g_dst_buf, &dev_addr_C));

    std::cout << "GPU memory addresses:" << std::endl;
    std::cout << "  Buffer A: 0x" << std::hex << dev_addr_A << std::endl;
    std::cout << "  Buffer B: 0x" << std::hex << dev_addr_B << std::endl;
    std::cout << "  Buffer C: 0x" << std::hex << dev_addr_C << std::endl;
    std::cout << std::dec;

    // Transfer data to GPU
    std::cout << "Uploading input data to GPU..." << std::endl;
    GPU_CHECK(vx_copy_to_dev(g_src0_buf, A.data(), 0, buf_size_A));
    GPU_CHECK(vx_copy_to_dev(g_src1_buf, B.data(), 0, buf_size_B));

    // Upload kernel
    std::cout << "Uploading kernel: " << kernel_file << std::endl;
    GPU_CHECK(vx_upload_kernel_file(g_device, kernel_file, &g_kernel_buf));

    // Setup kernel arguments
    kernel_arg_t kernel_args = {
        .src0_addr = dev_addr_A,
        .src1_addr = dev_addr_B,
        .dst_addr = dev_addr_C,
        .M = M,
        .K = K,
        .N = N,
        .num_points = M * N
    };

    std::cout << "Uploading kernel arguments..." << std::endl;
    GPU_CHECK(vx_upload_bytes(g_device, &kernel_args, sizeof(kernel_arg_t), &g_args_buf));

    // Execute kernel
    std::cout << "Starting GPU kernel execution..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    GPU_CHECK(vx_start(g_device, g_kernel_buf, g_args_buf));

    // Wait for completion
    std::cout << "Waiting for GPU completion..." << std::endl;
    GPU_CHECK(vx_ready_wait(g_device, VX_MAX_TIMEOUT));

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "GPU execution completed in " << duration.count() << " ms" << std::endl;

    // Download results
    std::cout << "Downloading results from GPU..." << std::endl;
    GPU_CHECK(vx_copy_from_dev(C.data(), g_dst_buf, 0, buf_size_C));

    std::cout << "GPU MatMul execution completed successfully" << std::endl;
}

/**
 * Verify GPU results against CPU computation
 */
void verify_matmul(const std::vector<float>& A,
                   const std::vector<float>& B,
                   const std::vector<float>& C_gpu,
                   uint32_t M, uint32_t K, uint32_t N,
                   float tolerance = 1e-5) {
    std::cout << "\n=== Verifying Results ===" << std::endl;

    // CPU computation
    std::vector<float> C_cpu(M * N, 0.0f);
    for (uint32_t i = 0; i < M; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C_cpu[i * N + j] = sum;
        }
    }

    // Compare
    int errors = 0;
    for (uint32_t i = 0; i < M * N; ++i) {
        float diff = std::fabs(C_gpu[i] - C_cpu[i]);
        if (diff > tolerance) {
            if (errors < 10) {
                std::cout << "ERROR at [" << i << "]: GPU=" << C_gpu[i]
                         << ", CPU=" << C_cpu[i] << ", diff=" << diff << std::endl;
            }
            errors++;
        }
    }

    if (errors == 0) {
        std::cout << "✓ All results verified correctly!" << std::endl;
    } else {
        std::cout << "✗ Found " << errors << " errors!" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "=== CXLMemUring GPU Executor ===" << std::endl;
    std::cout << "Target: Vortex GPU" << std::endl;

    // Parse arguments
    const char* kernel_file = "kernel.vxbin";
    uint32_t M = 128, K = 128, N = 128;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-k" && i + 1 < argc) {
            kernel_file = argv[++i];
        } else if (arg == "-M" && i + 1 < argc) {
            M = std::atoi(argv[++i]);
        } else if (arg == "-K" && i + 1 < argc) {
            K = std::atoi(argv[++i]);
        } else if (arg == "-N" && i + 1 < argc) {
            N = std::atoi(argv[++i]);
        } else if (arg == "-h") {
            std::cout << "Usage: gpu_executor [-k kernel.vxbin] [-M rows] [-K inner] [-N cols]" << std::endl;
            return 0;
        }
    }

    // Generate test data
    std::cout << "\nGenerating test data..." << std::endl;
    std::vector<float> A(M * K), B(K * N), C(M * N);

    // Initialize with random values
    for (auto& val : A) val = static_cast<float>(rand()) / RAND_MAX;
    for (auto& val : B) val = static_cast<float>(rand()) / RAND_MAX;

    std::cout << "Data sizes: A=" << A.size() << ", B=" << B.size() << ", C=" << C.size() << std::endl;

    // Execute on GPU
    try {
        execute_matmul_gpu(A, B, C, M, K, N, kernel_file);
        verify_matmul(A, B, C, M, K, N);
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        cleanup();
        return -1;
    }

    cleanup();
    std::cout << "\n✓ GPU Execution Completed Successfully" << std::endl;
    return 0;
}
