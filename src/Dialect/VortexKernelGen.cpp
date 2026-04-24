//===- VortexKernelGen.cpp - Vortex GPU Kernel Generation ====//
//
// Generates Vortex GPU kernels from MLIR operations.
// Creates .cpp kernel source code that can be compiled to .vxbin.
//
//===------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "Dialect/CiraOps.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <sstream>

#define DEBUG_TYPE "vortex-kernel-gen"

using namespace mlir;

namespace mlir {

//===-------------------------------------------------------===//
// Vortex Kernel Generator Pass
//===-------------------------------------------------------===//

class VortexKernelGenPass
    : public PassWrapper<VortexKernelGenPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VortexKernelGenPass)

    VortexKernelGenPass() = default;

    StringRef getArgument() const final { return "vortex-kernel-gen"; }
    StringRef getDescription() const final {
        return "Generate Vortex GPU kernel code from offloaded operations";
    }

    void runOnOperation() override {
        auto module = getOperation();

        llvm::outs() << "\n=== Vortex Kernel Generation ===\n";
        llvm::outs() << "Generating GPU kernel source code...\n\n";

        int kernel_count = 0;
        int prefetch_kernel_count = 0;
        std::stringstream kernel_code;

        // Generate kernel header (includes prefetch/cache management support)
        kernel_code << generateKernelHeader();

        // Walk operations and generate kernels for GPU-offloaded linalg ops
        module.walk([&](Operation *op) {
            if (auto offloadAttr = op->getAttr("gpu.offload")) {
                if (auto boolAttr = dyn_cast<BoolAttr>(offloadAttr)) {
                    if (boolAttr.getValue()) {
                        std::string kernel_name = "unknown";
                        if (auto kernelAttr = op->getAttr("gpu.kernel_name")) {
                            if (auto strAttr = dyn_cast<StringAttr>(kernelAttr)) {
                                kernel_name = strAttr.getValue().str();
                            }
                        }

                        llvm::outs() << "  Generating: " << kernel_name << "\n";

                        if (isa<linalg::MatmulOp>(op)) {
                            kernel_code << generateMatmulKernel(
                                dyn_cast<linalg::MatmulOp>(op), kernel_name);
                            kernel_count++;
                        } else if (isa<linalg::GenericOp>(op)) {
                            kernel_code << generateGenericKernel(
                                dyn_cast<linalg::GenericOp>(op), kernel_name);
                            kernel_count++;
                        }
                    }
                }
            }
        });

        // Walk CIRA operations and generate prefetch/cache management kernels
        module.walk([&](cira::OffloadRegionOp offloadOp) {
            std::string kernel_name = "prefetch_offload_" +
                                      std::to_string(prefetch_kernel_count);

            // Analyze the offload body for prefetch/cache ops
            bool hasPrefetchChain = false;
            bool hasInstallCacheline = false;
            bool hasPrefetchStream = false;
            bool hasPrefetchIndirect = false;
            int64_t chainDepth = 16;
            int64_t ptrOffset = 8;

            offloadOp.getBody().walk([&](Operation *bodyOp) {
                if (auto chainOp = dyn_cast<cira::PrefetchChainOp>(bodyOp)) {
                    hasPrefetchChain = true;
                    chainDepth = chainOp.getDepthAttr().getInt();
                } else if (isa<cira::InstallCachelineOp>(bodyOp)) {
                    hasInstallCacheline = true;
                } else if (isa<cira::PrefetchStreamOp>(bodyOp)) {
                    hasPrefetchStream = true;
                } else if (auto indirectOp = dyn_cast<cira::PrefetchIndirectOp>(bodyOp)) {
                    hasPrefetchIndirect = true;
                    ptrOffset = indirectOp.getNextPtrOffsetAttr().getInt();
                    chainDepth = indirectOp.getDepthAttr().getInt();
                }
            });

            if (hasPrefetchChain || hasPrefetchIndirect) {
                llvm::outs() << "  Generating prefetch chain kernel: "
                            << kernel_name << " (depth=" << chainDepth << ")\n";
                kernel_code << generatePrefetchChainKernel(
                    kernel_name, chainDepth, ptrOffset, hasInstallCacheline);
                prefetch_kernel_count++;
            } else if (hasPrefetchStream) {
                llvm::outs() << "  Generating stream prefetch kernel: "
                            << kernel_name << "\n";
                kernel_code << generateStreamPrefetchKernel(kernel_name);
                prefetch_kernel_count++;
            } else if (hasInstallCacheline) {
                llvm::outs() << "  Generating cacheline install kernel: "
                            << kernel_name << "\n";
                kernel_code << generateCachelineInstallKernel(kernel_name);
                prefetch_kernel_count++;
            }
        });

        // Also generate kernels from OffloadStartOp regions
        module.walk([&](cira::OffloadStartOp offloadStart) {
            std::string kernel_name = "offload_start_" +
                                      std::to_string(prefetch_kernel_count);
            bool hasPrefetchChain = false;
            int64_t chainDepth = 16;

            offloadStart.getBody().walk([&](Operation *bodyOp) {
                if (auto chainOp = dyn_cast<cira::PrefetchChainOp>(bodyOp)) {
                    hasPrefetchChain = true;
                    chainDepth = chainOp.getDepthAttr().getInt();
                }
            });

            if (hasPrefetchChain) {
                llvm::outs() << "  Generating offload_start kernel: "
                            << kernel_name << " (depth=" << chainDepth << ")\n";
                kernel_code << generatePrefetchChainKernel(
                    kernel_name, chainDepth, 8, true);
                prefetch_kernel_count++;
            }
        });

        // Generate kernel main (dispatch table)
        kernel_code << generateKernelMain();

        // Write kernel to file
        std::ofstream outfile("/home/victoryang00/CXLMemUring/kernel_gpu.cpp");
        outfile << kernel_code.str();
        outfile.close();

        llvm::outs() << "\nGenerated kernel code: kernel_gpu.cpp\n";
        llvm::outs() << "  GEMM kernels: " << kernel_count << "\n";
        llvm::outs() << "  Prefetch/cache kernels: " << prefetch_kernel_count << "\n";
        llvm::outs() << "  Total kernels: "
                    << (kernel_count + prefetch_kernel_count) << "\n";
        llvm::outs() << "\nNext: Compile with Vortex toolchain\n";
        llvm::outs() << "  cd /home/victoryang00/vortex\n";
        llvm::outs() << "  make -C tests/regression/matmul\n\n";
    }

private:
    std::string generateKernelHeader() {
        return R"(
#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <stdint.h>

// =============================================================================
// Kernel argument structures
// =============================================================================

// GEMM kernel arguments
typedef struct {
    uint64_t src0_addr;
    uint64_t src1_addr;
    uint64_t dst_addr;
    uint32_t M, K, N;
    uint32_t num_points;
} kernel_arg_t;

// Prefetch chain kernel arguments (pointer chasing)
typedef struct {
    uint64_t start_node_addr;    // Starting address of linked structure
    uint64_t buf_addr;           // Host-visible LLC tile buffer
    uint64_t completion_addr;    // CompletionData address (DCOH-coherent)
    uint32_t chain_depth;        // Number of nodes to chase ahead
    uint32_t next_ptr_offset;    // Byte offset of 'next' pointer in node struct
    uint32_t data_offset;        // Byte offset of data payload in node
    uint32_t data_size;          // Size of data payload to copy back
} prefetch_chain_arg_t;

// Stream prefetch kernel arguments (sequential/strided)
typedef struct {
    uint64_t base_addr;          // Base address of stream
    uint64_t buf_addr;           // Host-visible LLC tile buffer
    uint64_t completion_addr;    // CompletionData address (DCOH-coherent)
    uint32_t count;              // Number of elements to prefetch
    uint32_t stride;             // Stride between elements (bytes)
    uint32_t elem_size;          // Element size (bytes)
    uint32_t pad;
} stream_prefetch_arg_t;

// Cacheline install kernel arguments
typedef struct {
    uint64_t src_addr;           // Source address in CXL memory
    uint64_t dst_addr;           // Destination in host-visible memory (LLC tile)
    uint64_t completion_addr;    // CompletionData address (DCOH-coherent)
    uint32_t num_cachelines;     // Number of 64B cache lines to install
    uint32_t cache_level;        // Target cache level (1=L1, 2=L2, 3=LLC)
} cacheline_install_arg_t;

// CompletionData structure (must match host-side Type2KernelCompletion)
typedef struct __attribute__((aligned(64))) {
    uint32_t magic;              // 0xDEADBEEF when done
    uint32_t status;             // 0 = success
    uint64_t result;             // Kernel-specific result
    uint64_t cycles;             // Execution cycles
    uint64_t timestamp;          // Completion timestamp
    uint8_t  reserved[32];       // Pad to 64 bytes (one cache line)
} completion_data_t;

#define COMPLETION_MAGIC 0xDEADBEEF
#define CACHELINE_SIZE   64

)";
    }

    std::string generateMatmulKernel(linalg::MatmulOp op,
                                     const std::string &kernel_name) {
        if (op->getNumOperands() < 2) return "";

        Type lhsType = op->getOperand(0).getType();
        Type rhsType = op->getOperand(1).getType();

        auto lhsMemref = dyn_cast<MemRefType>(lhsType);
        auto rhsMemref = dyn_cast<MemRefType>(rhsType);

        if (!lhsMemref || !rhsMemref) return "";

        auto lhsShape = lhsMemref.getShape();
        auto rhsShape = rhsMemref.getShape();

        if (lhsShape.size() < 2 || rhsShape.size() < 2) return "";

        int64_t M = lhsShape[0];
        int64_t K = lhsShape[1];
        int64_t N = rhsShape[1];

        std::stringstream ss;
        ss << "// MatMul Kernel: " << M << "x" << K << "x" << N << "\n";
        ss << "void " << kernel_name << "_kernel(kernel_arg_t* __UNIFORM__ arg) {\n";
        ss << "    auto src0_ptr = reinterpret_cast<float*>(arg->src0_addr);\n";
        ss << "    auto src1_ptr = reinterpret_cast<float*>(arg->src1_addr);\n";
        ss << "    auto dst_ptr  = reinterpret_cast<float*>(arg->dst_addr);\n";
        ss << "\n";
        ss << "    uint32_t row = blockIdx.x;\n";
        ss << "    uint32_t col = threadIdx.x;\n";
        ss << "\n";
        ss << "    if (row < " << M << " && col < " << N << ") {\n";
        ss << "        float sum = 0.0f;\n";
        ss << "        for (uint32_t k = 0; k < " << K << "; ++k) {\n";
        ss << "            float a = src0_ptr[row * " << K << " + k];\n";
        ss << "            float b = src1_ptr[k * " << N << " + col];\n";
        ss << "            sum += a * b;\n";
        ss << "        }\n";
        ss << "        dst_ptr[row * " << N << " + col] = sum;\n";
        ss << "    }\n";
        ss << "}\n\n";

        return ss.str();
    }

    std::string generateGenericKernel(linalg::GenericOp op,
                                      const std::string &kernel_name) {
        std::stringstream ss;
        ss << "// Generic Kernel\n";
        ss << "void " << kernel_name << "_kernel(kernel_arg_t* __UNIFORM__ arg) {\n";
        ss << "    // Generic operation - implement operation-specific logic\n";
        ss << "    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
        ss << "    if (idx < arg->num_points) {\n";
        ss << "        // TODO: Insert operation-specific code here\n";
        ss << "    }\n";
        ss << "}\n\n";

        return ss.str();
    }

    // =========================================================================
    // Prefetch chain kernel: chases pointer chain and installs into host LLC
    // =========================================================================
    std::string generatePrefetchChainKernel(const std::string &kernel_name,
                                             int64_t depth, int64_t ptrOffset,
                                             bool installCacheline) {
        std::stringstream ss;
        ss << "// Prefetch Chain Kernel: chase " << depth << " nodes ahead\n";
        ss << "// Pointer offset: " << ptrOffset << " bytes\n";
        ss << "void " << kernel_name << "_kernel(prefetch_chain_arg_t* __UNIFORM__ arg) {\n";
        ss << "    uint32_t tid = vx_thread_id();\n";
        ss << "    uint32_t wid = vx_warp_id();\n";
        ss << "\n";
        ss << "    // Only thread 0 of warp 0 chases the chain\n";
        ss << "    // (pointer chasing is inherently serial)\n";
        ss << "    if (tid == 0 && wid == 0) {\n";
        ss << "        volatile uint8_t* node = (volatile uint8_t*)arg->start_node_addr;\n";
        ss << "        uint8_t* buf = (uint8_t*)arg->buf_addr;\n";
        ss << "        uint32_t depth = arg->chain_depth;\n";
        ss << "        uint32_t next_off = arg->next_ptr_offset;\n";
        ss << "        uint32_t data_off = arg->data_offset;\n";
        ss << "        uint32_t data_sz  = arg->data_size;\n";
        ss << "\n";
        ss << "        for (uint32_t i = 0; i < depth && node != 0; ++i) {\n";
        if (installCacheline) {
        ss << "            // Read node data (this pulls the cache line into device cache)\n";
        ss << "            // Then write to host-visible buffer (DCOH pushes to host LLC)\n";
        ss << "            for (uint32_t b = 0; b < data_sz; b += 4) {\n";
        ss << "                uint32_t val = *(volatile uint32_t*)(node + data_off + b);\n";
        ss << "                *(volatile uint32_t*)(buf + i * data_sz + b) = val;\n";
        ss << "            }\n";
        } else {
        ss << "            // Touch the cache line to bring it into device cache\n";
        ss << "            volatile uint32_t dummy = *(volatile uint32_t*)node;\n";
        ss << "            (void)dummy;\n";
        }
        ss << "\n";
        ss << "            // Chase the 'next' pointer\n";
        ss << "            node = *(volatile uint8_t**)(node + next_off);\n";
        ss << "        }\n";
        ss << "\n";
        ss << "        // Signal completion via DCOH writeback\n";
        ss << "        volatile completion_data_t* comp =\n";
        ss << "            (volatile completion_data_t*)arg->completion_addr;\n";
        ss << "        comp->cycles = vx_num_cycles();\n";
        ss << "        comp->status = 0;\n";
        ss << "        comp->magic = COMPLETION_MAGIC;  // DCOH delivers to host LLC\n";
        ss << "    }\n";
        ss << "}\n\n";
        return ss.str();
    }

    // =========================================================================
    // Stream prefetch kernel: prefetch sequential/strided data in parallel
    // =========================================================================
    std::string generateStreamPrefetchKernel(const std::string &kernel_name) {
        std::stringstream ss;
        ss << "// Stream Prefetch Kernel: parallel sequential/strided prefetch\n";
        ss << "void " << kernel_name << "_kernel(stream_prefetch_arg_t* __UNIFORM__ arg) {\n";
        ss << "    uint32_t tid = vx_thread_id();\n";
        ss << "    uint32_t num_threads = vx_num_threads();\n";
        ss << "\n";
        ss << "    uint8_t* base = (uint8_t*)arg->base_addr;\n";
        ss << "    uint8_t* buf  = (uint8_t*)arg->buf_addr;\n";
        ss << "    uint32_t count  = arg->count;\n";
        ss << "    uint32_t stride = arg->stride;\n";
        ss << "    uint32_t elem_sz = arg->elem_size;\n";
        ss << "\n";
        ss << "    // Each thread handles a subset of elements (SIMT parallelism)\n";
        ss << "    for (uint32_t i = tid; i < count; i += num_threads) {\n";
        ss << "        uint8_t* src = base + i * stride;\n";
        ss << "        uint8_t* dst = buf + i * elem_sz;\n";
        ss << "\n";
        ss << "        // Copy element to host-visible buffer (DCOH pushes to host LLC)\n";
        ss << "        for (uint32_t b = 0; b < elem_sz; b += 4) {\n";
        ss << "            *(volatile uint32_t*)(dst + b) = *(volatile uint32_t*)(src + b);\n";
        ss << "        }\n";
        ss << "    }\n";
        ss << "\n";
        ss << "    // Thread 0 signals completion\n";
        ss << "    vx_barrier(0, num_threads);\n";
        ss << "    if (tid == 0) {\n";
        ss << "        volatile completion_data_t* comp =\n";
        ss << "            (volatile completion_data_t*)arg->completion_addr;\n";
        ss << "        comp->cycles = vx_num_cycles();\n";
        ss << "        comp->status = 0;\n";
        ss << "        comp->magic = COMPLETION_MAGIC;\n";
        ss << "    }\n";
        ss << "}\n\n";
        return ss.str();
    }

    // =========================================================================
    // Cacheline install kernel: bulk install cache lines from CXL to host LLC
    // =========================================================================
    std::string generateCachelineInstallKernel(const std::string &kernel_name) {
        std::stringstream ss;
        ss << "// Cacheline Install Kernel: bulk install from CXL to host LLC\n";
        ss << "void " << kernel_name << "_kernel(cacheline_install_arg_t* __UNIFORM__ arg) {\n";
        ss << "    uint32_t tid = vx_thread_id();\n";
        ss << "    uint32_t num_threads = vx_num_threads();\n";
        ss << "\n";
        ss << "    uint8_t* src = (uint8_t*)arg->src_addr;\n";
        ss << "    uint8_t* dst = (uint8_t*)arg->dst_addr;\n";
        ss << "    uint32_t num_cls = arg->num_cachelines;\n";
        ss << "\n";
        ss << "    // Each thread installs a subset of cache lines (SIMT parallel)\n";
        ss << "    for (uint32_t i = tid; i < num_cls; i += num_threads) {\n";
        ss << "        // Read from CXL memory (device-side)\n";
        ss << "        // Write to host-visible buffer (DCOH auto-installs in host LLC)\n";
        ss << "        uint8_t* cl_src = src + i * CACHELINE_SIZE;\n";
        ss << "        uint8_t* cl_dst = dst + i * CACHELINE_SIZE;\n";
        ss << "\n";
        ss << "        // Copy one cache line (64 bytes = 16 x uint32_t)\n";
        ss << "        for (uint32_t w = 0; w < CACHELINE_SIZE; w += 4) {\n";
        ss << "            *(volatile uint32_t*)(cl_dst + w) = *(volatile uint32_t*)(cl_src + w);\n";
        ss << "        }\n";
        ss << "    }\n";
        ss << "\n";
        ss << "    // Thread 0 signals completion after all threads done\n";
        ss << "    vx_barrier(0, num_threads);\n";
        ss << "    if (tid == 0) {\n";
        ss << "        volatile completion_data_t* comp =\n";
        ss << "            (volatile completion_data_t*)arg->completion_addr;\n";
        ss << "        comp->result = (uint64_t)num_cls;  // report lines installed\n";
        ss << "        comp->cycles = vx_num_cycles();\n";
        ss << "        comp->status = 0;\n";
        ss << "        comp->magic = COMPLETION_MAGIC;\n";
        ss << "    }\n";
        ss << "}\n\n";
        return ss.str();
    }

    std::string generateKernelMain() {
        return R"(
// =============================================================================
// Kernel type constants (set by host via CSR before launch)
// =============================================================================
#define KERNEL_TYPE_GEMM              1
#define KERNEL_TYPE_GENERIC           2
#define KERNEL_TYPE_PREFETCH_CHAIN    3
#define KERNEL_TYPE_STREAM_PREFETCH   4
#define KERNEL_TYPE_CACHELINE_INSTALL 5

// Kernel entry point — dispatches based on kernel type CSR
int main() {
    uint32_t kernel_type = csr_read(VX_CSR_KERNEL_TYPE);

    switch (kernel_type) {
    case KERNEL_TYPE_GEMM: {
        kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
        uint32_t grid[1] = {arg->N};
        return vx_spawn_threads(arg->M, grid, nullptr,
                               (vx_kernel_func_cb)gemm_kernel_128x128x128_kernel, arg);
    }

    case KERNEL_TYPE_GENERIC: {
        kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
        uint32_t grid[1] = {arg->num_points};
        return vx_spawn_threads(1, grid, nullptr,
                               (vx_kernel_func_cb)generic_kernel_kernel, arg);
    }

    case KERNEL_TYPE_PREFETCH_CHAIN: {
        // Pointer-chasing prefetch: serial chain walk, install into host LLC
        prefetch_chain_arg_t* arg = (prefetch_chain_arg_t*)csr_read(VX_CSR_MSCRATCH);
        // Single warp, single thread (chain is serial)
        uint32_t grid[1] = {1};
        return vx_spawn_threads(1, grid, nullptr,
                               (vx_kernel_func_cb)prefetch_offload_0_kernel, arg);
    }

    case KERNEL_TYPE_STREAM_PREFETCH: {
        // Sequential/strided prefetch: parallel across threads
        stream_prefetch_arg_t* arg = (stream_prefetch_arg_t*)csr_read(VX_CSR_MSCRATCH);
        uint32_t grid[1] = {arg->count};
        return vx_spawn_threads(1, grid, nullptr,
                               (vx_kernel_func_cb)prefetch_offload_0_kernel, arg);
    }

    case KERNEL_TYPE_CACHELINE_INSTALL: {
        // Bulk cacheline install: parallel across threads
        cacheline_install_arg_t* arg = (cacheline_install_arg_t*)csr_read(VX_CSR_MSCRATCH);
        uint32_t grid[1] = {arg->num_cachelines};
        return vx_spawn_threads(1, grid, nullptr,
                               (vx_kernel_func_cb)prefetch_offload_0_kernel, arg);
    }

    default:
        return -1;
    }
}
)";
    }
};

std::unique_ptr<Pass> createVortexKernelGen() {
    return std::make_unique<VortexKernelGenPass>();
}

} // namespace mlir
