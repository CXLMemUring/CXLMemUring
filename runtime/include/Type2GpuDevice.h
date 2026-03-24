/**
 * Type2GpuDevice.h
 *
 * CXL Type 2 GPU device interface for llama.cpp integration.
 * Provides matrix multiplication offloading to Intel IA-780i Vortex GPU
 * via coherent CXL.mem and CXL.cache protocols.
 */

#ifndef CXL_TYPE2_GPU_DEVICE_H
#define CXL_TYPE2_GPU_DEVICE_H

#include "CiraRuntime.h"
#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace cira {
namespace runtime {

/**
 * Type 2 GPU kernel execution request
 */
struct Type2KernelRequest {
    // Kernel entry point address (device memory)
    uint64_t kernel_addr;

    // Kernel arguments address (shared memory)
    uint64_t args_addr;

    // Grid dimensions
    uint32_t grid_x, grid_y, grid_z;

    // Block/thread block dimensions
    uint32_t block_x, block_y, block_z;

    // Completion callback address (for DCOH signaling)
    uint64_t completion_addr;

    // Enable cache-coherent completion
    bool dcoh_enabled;

    // Timeout in milliseconds (0 = no timeout)
    uint32_t timeout_ms;
};

/**
 * Type 2 GPU kernel completion status
 */
struct Type2KernelCompletion {
    static constexpr uint32_t MAGIC = 0xDEADBEEF;

    uint32_t magic;         // Sanity check (0xDEADBEEF)
    uint32_t status;        // 0 = success, >0 = error code
    uint64_t result;        // Kernel-specific result value
    uint64_t cycles;        // Execution cycle counter
    uint64_t timestamp;     // Timestamp at completion
    uint8_t  reserved[32];  // Pad to 64 bytes (cache line)
} __attribute__((aligned(64)));

/**
 * GEMM kernel arguments (matches Vortex kernel layout)
 */
struct Type2GemmArgs {
    static constexpr size_t CACHE_LINE_ALIGN = 64;

    uint64_t A_addr;        // Matrix A address
    uint64_t B_addr;        // Matrix B address
    uint64_t C_addr;        // Matrix C address (in/out)
    uint32_t M, N, K;       // Dimensions
    uint32_t lda, ldb, ldc; // Leading dimensions
    float    alpha, beta;   // Scalars
    uint64_t completion_addr; // DCOH completion address
    uint8_t  pad[4];        // Pad to 72 bytes
} __attribute__((aligned(CACHE_LINE_ALIGN)));

/**
 * Type 2 GPU device controller
 */
class Type2GpuDevice {
public:
    virtual ~Type2GpuDevice() = default;

    // Device lifecycle
    virtual bool initialize() = 0;
    virtual void shutdown() = 0;
    virtual bool is_available() const = 0;

    // Device information
    virtual const char* device_name() const = 0;
    virtual uint32_t device_id() const = 0;
    virtual size_t shared_memory_size() const = 0;
    virtual size_t device_memory_size() const = 0;

    // Memory operations
    virtual void* allocate_shared(size_t size) = 0;
    virtual void* allocate_device(size_t size) = 0;
    virtual void free_shared(void* ptr) = 0;
    virtual void free_device(void* ptr) = 0;

    // Copy operations
    virtual bool copy_host_to_device(void* dev_ptr, const void* host_ptr, size_t size) = 0;
    virtual bool copy_device_to_host(void* host_ptr, const void* dev_ptr, size_t size) = 0;

    // Kernel execution
    virtual bool launch_kernel(const Type2KernelRequest& request) = 0;
    virtual bool wait_kernel_completion(uint32_t timeout_ms = 0) = 0;

    // GEMM acceleration
    virtual bool gemm_f32(
        float* C,
        const float* A,
        const float* B,
        uint32_t M, uint32_t N, uint32_t K,
        float alpha = 1.0f,
        float beta = 0.0f,
        uint32_t timeout_ms = 0
    ) = 0;

    // Statistics
    virtual uint64_t get_kernel_cycles() const = 0;
    virtual uint64_t get_kernel_instructions() const = 0;
};

/**
 * Factory function to create Type 2 GPU device instance
 * Supports both real hardware and simulation fallback.
 *
 * @param pci_device PCIe device path (e.g., "0000:ad:00.0")
 * @param dax_device DAX device path (e.g., "/dev/dax12.0")
 * @param use_simulation Force simulation mode if true
 * @return Unique pointer to Type2GpuDevice, nullptr if unavailable
 */
std::unique_ptr<Type2GpuDevice> create_type2_gpu_device(
    const std::string& pci_device = "0000:ad:00.0",
    const std::string& dax_device = "/dev/dax12.0",
    bool use_simulation = false
);

} // namespace runtime
} // namespace cira

#endif // CXL_TYPE2_GPU_DEVICE_H
