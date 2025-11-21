// Vortex Device Interface Implementation
// Bridges CIRA runtime (x86_64 host) with Vortex RISC-V SIMT device

#include "vortex_device.h"
#include <vortex.h>  // Vortex runtime header
#include <iostream>
#include <vector>
#include <map>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

// Internal device structure
struct vortex_device {
    vx_device_h vx_dev;
    vortex_device_caps_t caps;
    bool initialized;
    bool debug_enabled;
    std::map<std::string, uint64_t> perf_counters;
};

// Internal buffer structure
struct vortex_buffer {
    vx_buffer_h vx_buf;
    vortex_device_h device;
    uint64_t dev_addr;
    size_t size;
};

// Internal kernel structure
struct vortex_kernel {
    vortex_device_h device;
    vx_buffer_h kernel_buffer;
    vx_buffer_h args_buffer;
    uint64_t kernel_addr;
    uint64_t args_addr;
    std::vector<vortex_kernel_arg_t> args;
};

// ============================================================================
// Device Management
// ============================================================================

int vortex_device_init(vortex_device_h* device, const char* sim_path) {
    if (!device) return VORTEX_ERROR_INVALID;

    // Allocate device structure
    *device = new vortex_device();
    if (!*device) return VORTEX_ERROR_MEMORY;

    // Set environment variable for simulator path if provided
    if (sim_path) {
        setenv("VORTEX_SIM_PATH", sim_path, 1);
    }

    // Open Vortex device
    int ret = vx_dev_open(&((*device)->vx_dev));
    if (ret != 0) {
        delete *device;
        *device = nullptr;
        std::cerr << "Failed to open Vortex device: " << ret << std::endl;
        return VORTEX_ERROR_INIT;
    }

    // Query device capabilities
    uint64_t value;

    vx_dev_caps((*device)->vx_dev, VX_CAPS_NUM_CORES, &value);
    (*device)->caps.num_cores = (uint32_t)value;

    vx_dev_caps((*device)->vx_dev, VX_CAPS_NUM_WARPS, &value);
    (*device)->caps.num_warps_per_core = (uint32_t)value;

    vx_dev_caps((*device)->vx_dev, VX_CAPS_NUM_THREADS, &value);
    (*device)->caps.num_threads_per_warp = (uint32_t)value;

    vx_dev_caps((*device)->vx_dev, VX_CAPS_GLOBAL_MEM_SIZE, &value);
    (*device)->caps.global_mem_size = value;

    vx_dev_caps((*device)->vx_dev, VX_CAPS_LOCAL_MEM_SIZE, &value);
    (*device)->caps.local_mem_size = value;

    vx_dev_caps((*device)->vx_dev, VX_CAPS_CACHE_LINE_SIZE, &value);
    (*device)->caps.cache_line_size = (uint32_t)value;

    vx_dev_caps((*device)->vx_dev, VX_CAPS_ISA_FLAGS, &value);
    (*device)->caps.isa_flags = (uint32_t)value;

    (*device)->initialized = true;
    (*device)->debug_enabled = false;

    std::cout << "Vortex Device Initialized:" << std::endl;
    std::cout << "  Cores: " << (*device)->caps.num_cores << std::endl;
    std::cout << "  Warps per core: " << (*device)->caps.num_warps_per_core << std::endl;
    std::cout << "  Threads per warp: " << (*device)->caps.num_threads_per_warp << std::endl;
    std::cout << "  Global memory: " << (*device)->caps.global_mem_size << " bytes" << std::endl;

    return VORTEX_SUCCESS;
}

int vortex_device_get_caps(vortex_device_h device, vortex_device_caps_t* caps) {
    if (!device || !caps) return VORTEX_ERROR_INVALID;
    if (!device->initialized) return VORTEX_ERROR_INIT;

    *caps = device->caps;
    return VORTEX_SUCCESS;
}

int vortex_device_destroy(vortex_device_h device) {
    if (!device) return VORTEX_ERROR_INVALID;

    if (device->initialized) {
        vx_dev_close(device->vx_dev);
    }

    delete device;
    return VORTEX_SUCCESS;
}

// ============================================================================
// Memory Management
// ============================================================================

int vortex_malloc(vortex_device_h device, vortex_buffer_h* buffer, size_t size) {
    if (!device || !buffer || size == 0) return VORTEX_ERROR_INVALID;
    if (!device->initialized) return VORTEX_ERROR_INIT;

    // Allocate buffer structure
    *buffer = new vortex_buffer();
    if (!*buffer) return VORTEX_ERROR_MEMORY;

    (*buffer)->device = device;
    (*buffer)->size = size;

    // Allocate device memory
    int ret = vx_mem_alloc(device->vx_dev, size, VX_MEM_READ_WRITE,
                           &((*buffer)->vx_buf));
    if (ret != 0) {
        delete *buffer;
        *buffer = nullptr;
        return VORTEX_ERROR_MEMORY;
    }

    // Get device address
    ret = vx_mem_address((*buffer)->vx_buf, &((*buffer)->dev_addr));
    if (ret != 0) {
        vx_mem_free((*buffer)->vx_buf);
        delete *buffer;
        *buffer = nullptr;
        return VORTEX_ERROR_MEMORY;
    }

    if (device->debug_enabled) {
        std::cout << "Allocated " << size << " bytes at device address 0x"
                  << std::hex << (*buffer)->dev_addr << std::dec << std::endl;
    }

    return VORTEX_SUCCESS;
}

int vortex_free(vortex_buffer_h buffer) {
    if (!buffer) return VORTEX_ERROR_INVALID;

    vx_mem_free(buffer->vx_buf);
    delete buffer;
    return VORTEX_SUCCESS;
}

int vortex_buffer_get_address(vortex_buffer_h buffer, uint64_t* dev_addr) {
    if (!buffer || !dev_addr) return VORTEX_ERROR_INVALID;

    *dev_addr = buffer->dev_addr;
    return VORTEX_SUCCESS;
}

int vortex_memcpy(void* dst, const void* src, size_t size,
                  vortex_memcpy_kind_t kind) {
    if (!dst || !src || size == 0) return VORTEX_ERROR_INVALID;

    // This is a simplified version - assumes dst/src are buffer handles for device transfers
    switch (kind) {
        case VORTEX_MEMCPY_HOST_TO_DEVICE: {
            vortex_buffer_h dev_buf = static_cast<vortex_buffer_h>(dst);
            int ret = vx_copy_to_dev(dev_buf->vx_buf, src, 0, size);
            return (ret == 0) ? VORTEX_SUCCESS : VORTEX_ERROR_MEMORY;
        }
        case VORTEX_MEMCPY_DEVICE_TO_HOST: {
            vortex_buffer_h dev_buf = static_cast<vortex_buffer_h>(const_cast<void*>(src));
            int ret = vx_copy_from_dev(dst, dev_buf->vx_buf, 0, size);
            return (ret == 0) ? VORTEX_SUCCESS : VORTEX_ERROR_MEMORY;
        }
        case VORTEX_MEMCPY_DEVICE_TO_DEVICE:
            // Not directly supported - would need staging buffer
            return VORTEX_ERROR_INVALID;
    }
    return VORTEX_ERROR_INVALID;
}

int vortex_memcpy_offset(vortex_buffer_h buffer, uint64_t offset,
                         const void* src, size_t size,
                         vortex_memcpy_kind_t kind) {
    if (!buffer || !src || size == 0) return VORTEX_ERROR_INVALID;
    if (offset + size > buffer->size) return VORTEX_ERROR_INVALID;

    switch (kind) {
        case VORTEX_MEMCPY_HOST_TO_DEVICE: {
            int ret = vx_copy_to_dev(buffer->vx_buf, src, offset, size);
            return (ret == 0) ? VORTEX_SUCCESS : VORTEX_ERROR_MEMORY;
        }
        case VORTEX_MEMCPY_DEVICE_TO_HOST: {
            int ret = vx_copy_from_dev(const_cast<void*>(src), buffer->vx_buf, offset, size);
            return (ret == 0) ? VORTEX_SUCCESS : VORTEX_ERROR_MEMORY;
        }
        default:
            return VORTEX_ERROR_INVALID;
    }
}

// ============================================================================
// Kernel Management
// ============================================================================

int vortex_kernel_load(vortex_device_h device, vortex_kernel_h* kernel,
                       const void* binary_data, size_t binary_size) {
    if (!device || !kernel || !binary_data || binary_size == 0)
        return VORTEX_ERROR_INVALID;
    if (!device->initialized) return VORTEX_ERROR_INIT;

    // Allocate kernel structure
    *kernel = new vortex_kernel();
    if (!*kernel) return VORTEX_ERROR_MEMORY;

    (*kernel)->device = device;

    // Upload kernel binary to device
    int ret = vx_upload_kernel_bytes(device->vx_dev, binary_data, binary_size,
                                     &((*kernel)->kernel_buffer));
    if (ret != 0) {
        delete *kernel;
        *kernel = nullptr;
        return VORTEX_ERROR_KERNEL;
    }

    // Get kernel address
    ret = vx_mem_address((*kernel)->kernel_buffer, &((*kernel)->kernel_addr));
    if (ret != 0) {
        vx_mem_free((*kernel)->kernel_buffer);
        delete *kernel;
        *kernel = nullptr;
        return VORTEX_ERROR_KERNEL;
    }

    (*kernel)->args_buffer = nullptr;
    (*kernel)->args_addr = 0;

    if (device->debug_enabled) {
        std::cout << "Loaded kernel binary (" << binary_size << " bytes) at 0x"
                  << std::hex << (*kernel)->kernel_addr << std::dec << std::endl;
    }

    return VORTEX_SUCCESS;
}

int vortex_kernel_load_file(vortex_device_h device, vortex_kernel_h* kernel,
                            const char* binary_path) {
    if (!device || !kernel || !binary_path) return VORTEX_ERROR_INVALID;

    // Read binary file
    int fd = open(binary_path, O_RDONLY);
    if (fd < 0) {
        std::cerr << "Failed to open kernel file: " << binary_path << std::endl;
        return VORTEX_ERROR_KERNEL;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return VORTEX_ERROR_KERNEL;
    }

    size_t file_size = st.st_size;
    std::vector<uint8_t> binary_data(file_size);

    ssize_t bytes_read = read(fd, binary_data.data(), file_size);
    close(fd);

    if (bytes_read != (ssize_t)file_size) {
        return VORTEX_ERROR_KERNEL;
    }

    return vortex_kernel_load(device, kernel, binary_data.data(), file_size);
}

int vortex_kernel_set_args(vortex_kernel_h kernel,
                           const vortex_kernel_arg_t* args,
                           uint32_t num_args) {
    if (!kernel || !args || num_args == 0) return VORTEX_ERROR_INVALID;

    // Store arguments
    kernel->args.clear();
    kernel->args.reserve(num_args);

    // Calculate total argument buffer size
    size_t total_size = 0;
    for (uint32_t i = 0; i < num_args; i++) {
        // Align each argument
        size_t aligned_size = (args[i].size + args[i].alignment - 1) &
                              ~(args[i].alignment - 1);
        total_size += aligned_size;
        kernel->args.push_back(args[i]);
    }

    // Allocate argument buffer on device
    if (kernel->args_buffer != nullptr) {
        vx_mem_free(kernel->args_buffer);
    }

    int ret = vx_mem_alloc(kernel->device->vx_dev, total_size, VX_MEM_READ,
                           &(kernel->args_buffer));
    if (ret != 0) {
        return VORTEX_ERROR_MEMORY;
    }

    // Get argument buffer address
    ret = vx_mem_address(kernel->args_buffer, &(kernel->args_addr));
    if (ret != 0) {
        vx_mem_free(kernel->args_buffer);
        kernel->args_buffer = nullptr;
        return VORTEX_ERROR_MEMORY;
    }

    // Pack arguments into buffer
    std::vector<uint8_t> arg_data(total_size);
    size_t offset = 0;

    for (uint32_t i = 0; i < num_args; i++) {
        // Align offset
        offset = (offset + args[i].alignment - 1) & ~(args[i].alignment - 1);

        if (args[i].type == VORTEX_ARG_VALUE) {
            // Copy value
            memcpy(arg_data.data() + offset, args[i].host_ptr, args[i].size);
        } else if (args[i].type == VORTEX_ARG_BUFFER) {
            // Copy buffer address
            vortex_buffer_h buf = static_cast<vortex_buffer_h>(args[i].host_ptr);
            uint64_t addr = buf->dev_addr;
            memcpy(arg_data.data() + offset, &addr, sizeof(uint64_t));
        }

        offset += args[i].size;
    }

    // Upload arguments to device
    ret = vx_copy_to_dev(kernel->args_buffer, arg_data.data(), 0, total_size);
    if (ret != 0) {
        return VORTEX_ERROR_MEMORY;
    }

    if (kernel->device->debug_enabled) {
        std::cout << "Set " << num_args << " kernel arguments ("
                  << total_size << " bytes total)" << std::endl;
    }

    return VORTEX_SUCCESS;
}

int vortex_kernel_unload(vortex_kernel_h kernel) {
    if (!kernel) return VORTEX_ERROR_INVALID;

    if (kernel->kernel_buffer) {
        vx_mem_free(kernel->kernel_buffer);
    }
    if (kernel->args_buffer) {
        vx_mem_free(kernel->args_buffer);
    }

    delete kernel;
    return VORTEX_SUCCESS;
}

// ============================================================================
// Kernel Execution
// ============================================================================

int vortex_kernel_launch(vortex_device_h device, vortex_kernel_h kernel,
                         const vortex_launch_params_t* params) {
    if (!device || !kernel || !params) return VORTEX_ERROR_INVALID;
    if (!device->initialized) return VORTEX_ERROR_INIT;

    // Note: Vortex handles threading internally via vx_spawn_threads() in the kernel
    // The grid/block parameters are passed as part of the kernel arguments
    // We don't need to write DCR registers - vx_start() handles everything

    if (device->debug_enabled) {
        std::cout << "Launching kernel:" << std::endl;
        std::cout << "  Grid: (" << params->grid_dim_x << ", "
                  << params->grid_dim_y << ", " << params->grid_dim_z << ")" << std::endl;
        std::cout << "  Block: (" << params->block_dim_x << ", "
                  << params->block_dim_y << ", " << params->block_dim_z << ")" << std::endl;
    }

    // Start execution
    int ret = vx_start(device->vx_dev, kernel->kernel_buffer, kernel->args_buffer);
    if (ret != 0) {
        std::cerr << "Failed to start kernel execution: " << ret << std::endl;
        return VORTEX_ERROR_LAUNCH;
    }

    // Wait for completion
    ret = vx_ready_wait(device->vx_dev, VX_MAX_TIMEOUT);
    if (ret != 0) {
        std::cerr << "Kernel execution timed out or failed" << std::endl;
        return VORTEX_ERROR_TIMEOUT;
    }

    return VORTEX_SUCCESS;
}

int vortex_device_wait(vortex_device_h device, uint64_t timeout_ms) {
    if (!device) return VORTEX_ERROR_INVALID;
    if (!device->initialized) return VORTEX_ERROR_INIT;

    int ret = vx_ready_wait(device->vx_dev, timeout_ms);
    if (ret != 0) {
        return VORTEX_ERROR_TIMEOUT;
    }

    return VORTEX_SUCCESS;
}

int vortex_device_synchronize(vortex_device_h device) {
    // Wait with maximum timeout
    return vortex_device_wait(device, VX_MAX_TIMEOUT);
}

// ============================================================================
// Performance and Debugging
// ============================================================================

int vortex_device_get_perf_counter(vortex_device_h device,
                                   const char* counter_name,
                                   uint64_t* value) {
    if (!device || !counter_name || !value) return VORTEX_ERROR_INVALID;

    // Check if we have this counter cached
    auto it = device->perf_counters.find(counter_name);
    if (it != device->perf_counters.end()) {
        *value = it->second;
        return VORTEX_SUCCESS;
    }

    return VORTEX_ERROR_INVALID;
}

int vortex_device_dump_perf(vortex_device_h device, const char* output_path) {
    if (!device) return VORTEX_ERROR_INVALID;

    FILE* f = stdout;
    if (output_path) {
        f = fopen(output_path, "w");
        if (!f) return VORTEX_ERROR_INVALID;
    }

    vx_dump_perf(device->vx_dev, f);

    if (output_path) {
        fclose(f);
    }

    return VORTEX_SUCCESS;
}

int vortex_device_set_debug(vortex_device_h device, int enable) {
    if (!device) return VORTEX_ERROR_INVALID;

    device->debug_enabled = (enable != 0);
    return VORTEX_SUCCESS;
}
