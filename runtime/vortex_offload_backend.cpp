// Vortex Offload Backend for CIRA Runtime
// Integrates Vortex device with CIRA's offload engine

#include "vortex_device.h"
#include "shared_protocol.h"
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>
#include <cstring>

// Backend state management
class VortexOffloadBackend {
private:
    vortex_device_h device_;
    std::unordered_map<uint64_t, vortex_buffer_h> buffer_map_;
    std::unordered_map<uint64_t, vortex_kernel_h> kernel_map_;
    uint64_t next_buffer_id_ = 1;
    uint64_t next_kernel_id_ = 1;
    bool initialized_ = false;

public:
    VortexOffloadBackend() = default;
    ~VortexOffloadBackend();

    // Initialize backend with simulator path
    int initialize(const char* sim_path = nullptr);

    // Allocate device memory and return handle
    uint64_t allocate_device_memory(size_t size);

    // Free device memory
    int free_device_memory(uint64_t buffer_id);

    // Transfer data from host to device
    int host_to_device(uint64_t buffer_id, const void* host_data,
                       size_t offset, size_t size);

    // Transfer data from device to host
    int device_to_host(void* host_data, uint64_t buffer_id,
                       size_t offset, size_t size);

    // Load compiled kernel
    uint64_t load_kernel(const void* binary_data, size_t binary_size);

    // Load kernel from file
    uint64_t load_kernel_from_file(const char* kernel_path);

    // Set kernel arguments
    int set_kernel_arguments(uint64_t kernel_id, void** arg_ptrs,
                            size_t* arg_sizes, uint32_t num_args);

    // Launch kernel with grid/block configuration
    int launch_kernel(uint64_t kernel_id, uint32_t grid_x, uint32_t grid_y,
                     uint32_t grid_z, uint32_t block_x, uint32_t block_y,
                     uint32_t block_z);

    // Synchronize device
    int synchronize();

    // Get device information
    vortex_device_caps_t get_device_caps();

    // Enable debug output
    void set_debug(bool enable);
};

VortexOffloadBackend::~VortexOffloadBackend() {
    // Clean up all kernels
    for (auto& kv : kernel_map_) {
        vortex_kernel_unload(kv.second);
    }
    kernel_map_.clear();

    // Clean up all buffers
    for (auto& kv : buffer_map_) {
        vortex_free(kv.second);
    }
    buffer_map_.clear();

    // Close device
    if (initialized_) {
        vortex_device_destroy(device_);
    }
}

int VortexOffloadBackend::initialize(const char* sim_path) {
    if (initialized_) {
        std::cerr << "Backend already initialized" << std::endl;
        return -1;
    }

    int ret = vortex_device_init(&device_, sim_path);
    if (ret != VORTEX_SUCCESS) {
        std::cerr << "Failed to initialize Vortex device: " << ret << std::endl;
        return ret;
    }

    initialized_ = true;
    std::cout << "Vortex offload backend initialized" << std::endl;
    return 0;
}

uint64_t VortexOffloadBackend::allocate_device_memory(size_t size) {
    if (!initialized_) {
        std::cerr << "Backend not initialized" << std::endl;
        return 0;
    }

    vortex_buffer_h buffer;
    int ret = vortex_malloc(device_, &buffer, size);
    if (ret != VORTEX_SUCCESS) {
        std::cerr << "Failed to allocate device memory: " << ret << std::endl;
        return 0;
    }

    uint64_t buffer_id = next_buffer_id_++;
    buffer_map_[buffer_id] = buffer;

    return buffer_id;
}

int VortexOffloadBackend::free_device_memory(uint64_t buffer_id) {
    auto it = buffer_map_.find(buffer_id);
    if (it == buffer_map_.end()) {
        return -1;
    }

    vortex_free(it->second);
    buffer_map_.erase(it);
    return 0;
}

int VortexOffloadBackend::host_to_device(uint64_t buffer_id,
                                         const void* host_data,
                                         size_t offset, size_t size) {
    auto it = buffer_map_.find(buffer_id);
    if (it == buffer_map_.end()) {
        std::cerr << "Invalid buffer ID: " << buffer_id << std::endl;
        return -1;
    }

    int ret = vortex_memcpy_offset(it->second, offset, host_data, size,
                                   VORTEX_MEMCPY_HOST_TO_DEVICE);
    if (ret != VORTEX_SUCCESS) {
        std::cerr << "Failed to copy data to device: " << ret << std::endl;
        return ret;
    }

    return 0;
}

int VortexOffloadBackend::device_to_host(void* host_data, uint64_t buffer_id,
                                         size_t offset, size_t size) {
    auto it = buffer_map_.find(buffer_id);
    if (it == buffer_map_.end()) {
        std::cerr << "Invalid buffer ID: " << buffer_id << std::endl;
        return -1;
    }

    int ret = vortex_memcpy_offset(it->second, offset, host_data, size,
                                   VORTEX_MEMCPY_DEVICE_TO_HOST);
    if (ret != VORTEX_SUCCESS) {
        std::cerr << "Failed to copy data from device: " << ret << std::endl;
        return ret;
    }

    return 0;
}

uint64_t VortexOffloadBackend::load_kernel(const void* binary_data,
                                           size_t binary_size) {
    if (!initialized_) {
        std::cerr << "Backend not initialized" << std::endl;
        return 0;
    }

    vortex_kernel_h kernel;
    int ret = vortex_kernel_load(device_, &kernel, binary_data, binary_size);
    if (ret != VORTEX_SUCCESS) {
        std::cerr << "Failed to load kernel: " << ret << std::endl;
        return 0;
    }

    uint64_t kernel_id = next_kernel_id_++;
    kernel_map_[kernel_id] = kernel;

    return kernel_id;
}

uint64_t VortexOffloadBackend::load_kernel_from_file(const char* kernel_path) {
    if (!initialized_) {
        std::cerr << "Backend not initialized" << std::endl;
        return 0;
    }

    vortex_kernel_h kernel;
    int ret = vortex_kernel_load_file(device_, &kernel, kernel_path);
    if (ret != VORTEX_SUCCESS) {
        std::cerr << "Failed to load kernel from file: " << ret << std::endl;
        return 0;
    }

    uint64_t kernel_id = next_kernel_id_++;
    kernel_map_[kernel_id] = kernel;

    return kernel_id;
}

int VortexOffloadBackend::set_kernel_arguments(uint64_t kernel_id,
                                               void** arg_ptrs,
                                               size_t* arg_sizes,
                                               uint32_t num_args) {
    auto it = kernel_map_.find(kernel_id);
    if (it == kernel_map_.end()) {
        std::cerr << "Invalid kernel ID: " << kernel_id << std::endl;
        return -1;
    }

    // Convert to vortex_kernel_arg_t array
    std::vector<vortex_kernel_arg_t> args(num_args);
    for (uint32_t i = 0; i < num_args; i++) {
        args[i].host_ptr = arg_ptrs[i];
        args[i].size = arg_sizes[i];
        args[i].alignment = 8;  // Default 8-byte alignment

        // Determine if this is a buffer or value
        // Assume pointer-sized args are buffer references
        if (arg_sizes[i] == sizeof(void*)) {
            args[i].type = VORTEX_ARG_BUFFER;
        } else {
            args[i].type = VORTEX_ARG_VALUE;
        }
    }

    int ret = vortex_kernel_set_args(it->second, args.data(), num_args);
    if (ret != VORTEX_SUCCESS) {
        std::cerr << "Failed to set kernel arguments: " << ret << std::endl;
        return ret;
    }

    return 0;
}

int VortexOffloadBackend::launch_kernel(uint64_t kernel_id,
                                        uint32_t grid_x, uint32_t grid_y,
                                        uint32_t grid_z, uint32_t block_x,
                                        uint32_t block_y, uint32_t block_z) {
    auto it = kernel_map_.find(kernel_id);
    if (it == kernel_map_.end()) {
        std::cerr << "Invalid kernel ID: " << kernel_id << std::endl;
        return -1;
    }

    vortex_launch_params_t params = {};
    params.grid_dim_x = grid_x;
    params.grid_dim_y = grid_y;
    params.grid_dim_z = grid_z;
    params.block_dim_x = block_x;
    params.block_dim_y = block_y;
    params.block_dim_z = block_z;
    params.shared_mem_bytes = 0;

    int ret = vortex_kernel_launch(device_, it->second, &params);
    if (ret != VORTEX_SUCCESS) {
        std::cerr << "Failed to launch kernel: " << ret << std::endl;
        return ret;
    }

    return 0;
}

int VortexOffloadBackend::synchronize() {
    if (!initialized_) {
        return -1;
    }

    int ret = vortex_device_synchronize(device_);
    if (ret != VORTEX_SUCCESS) {
        std::cerr << "Device synchronization failed: " << ret << std::endl;
        return ret;
    }

    return 0;
}

vortex_device_caps_t VortexOffloadBackend::get_device_caps() {
    vortex_device_caps_t caps = {};
    if (initialized_) {
        vortex_device_get_caps(device_, &caps);
    }
    return caps;
}

void VortexOffloadBackend::set_debug(bool enable) {
    if (initialized_) {
        vortex_device_set_debug(device_, enable ? 1 : 0);
    }
}

// ============================================================================
// C API for integration with existing CIRA runtime
// ============================================================================

static VortexOffloadBackend* g_backend = nullptr;

extern "C" {

// Initialize the Vortex backend
int vortex_backend_init(const char* sim_path) {
    if (g_backend != nullptr) {
        std::cerr << "Backend already initialized" << std::endl;
        return -1;
    }

    g_backend = new VortexOffloadBackend();
    int ret = g_backend->initialize(sim_path);
    if (ret != 0) {
        delete g_backend;
        g_backend = nullptr;
        return ret;
    }

    return 0;
}

// Cleanup backend
void vortex_backend_cleanup() {
    if (g_backend) {
        delete g_backend;
        g_backend = nullptr;
    }
}

// Allocate device memory
uint64_t vortex_backend_malloc(size_t size) {
    if (!g_backend) return 0;
    return g_backend->allocate_device_memory(size);
}

// Free device memory
int vortex_backend_free(uint64_t buffer_id) {
    if (!g_backend) return -1;
    return g_backend->free_device_memory(buffer_id);
}

// Host to device transfer
int vortex_backend_h2d(uint64_t buffer_id, const void* host_data,
                       size_t offset, size_t size) {
    if (!g_backend) return -1;
    return g_backend->host_to_device(buffer_id, host_data, offset, size);
}

// Device to host transfer
int vortex_backend_d2h(void* host_data, uint64_t buffer_id,
                       size_t offset, size_t size) {
    if (!g_backend) return -1;
    return g_backend->device_to_host(host_data, buffer_id, offset, size);
}

// Load kernel
uint64_t vortex_backend_load_kernel(const void* binary, size_t size) {
    if (!g_backend) return 0;
    return g_backend->load_kernel(binary, size);
}

// Load kernel from file
uint64_t vortex_backend_load_kernel_file(const char* path) {
    if (!g_backend) return 0;
    return g_backend->load_kernel_from_file(path);
}

// Set kernel arguments
int vortex_backend_set_args(uint64_t kernel_id, void** args,
                            size_t* arg_sizes, uint32_t num_args) {
    if (!g_backend) return -1;
    return g_backend->set_kernel_arguments(kernel_id, args, arg_sizes, num_args);
}

// Launch kernel
int vortex_backend_launch(uint64_t kernel_id, uint32_t grid_x, uint32_t grid_y,
                          uint32_t grid_z, uint32_t block_x, uint32_t block_y,
                          uint32_t block_z) {
    if (!g_backend) return -1;
    return g_backend->launch_kernel(kernel_id, grid_x, grid_y, grid_z,
                                   block_x, block_y, block_z);
}

// Synchronize device
int vortex_backend_sync() {
    if (!g_backend) return -1;
    return g_backend->synchronize();
}

// Enable debug
void vortex_backend_set_debug(int enable) {
    if (g_backend) {
        g_backend->set_debug(enable != 0);
    }
}

} // extern "C"
