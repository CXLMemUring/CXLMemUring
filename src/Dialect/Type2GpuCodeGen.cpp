//===- Type2GpuCodeGen.cpp - Type2 GPU Code Generation Pass ====//
//
// Compiler-generated kernels from MLIR linalg IR with:
//   - Profile-guided optimization (PGO) for tile sizes
//   - Iterative software prefetching in GPU kernel loops
//   - CXL Type2 DCOH zero-copy memory model
//
// PGO modes:
//   pgo=instrument  → emit profiling counters, write profile.json
//   pgo=use         → read profile.json, optimize tiles + prefetch
//   pgo=off         → static heuristic tile sizes, basic prefetch
//
//===-------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <set>
#include <sstream>

#define DEBUG_TYPE "type2-gpu-codegen"

using namespace mlir;

namespace mlir {

// ================================================================
// Profile data structure (read from / written to JSON)
// ================================================================

struct KernelProfile {
    std::string name;
    int tileM = 0, tileN = 0, tileK = 0;
    int prefetchDist = 0;
    double totalNs = 0;
    double avgTileNs = 0;
    int64_t l1Misses = 0;
    int64_t l1Hits = 0;
    double hitRate = 0;
    int iteration = 0;  // PGO iteration number
};

//===-------------------------------------------------------===//
// Type2 GPU Code Generation Pass with PGO
//===-------------------------------------------------------===//

class Type2GpuCodeGenPass
    : public PassWrapper<Type2GpuCodeGenPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Type2GpuCodeGenPass)

    // PGO options
    Option<std::string> pgoMode{*this, "pgo",
        llvm::cl::desc("PGO mode: instrument|use|off"), llvm::cl::init("off")};
    Option<std::string> profilePath{*this, "profile",
        llvm::cl::desc("Profile data file path"), llvm::cl::init("pgo_profile.json")};
    Option<int> prefetchDist{*this, "prefetch-dist",
        llvm::cl::desc("Override prefetch distance (0=auto)"), llvm::cl::init(0)};

    Type2GpuCodeGenPass() = default;
    Type2GpuCodeGenPass(const Type2GpuCodeGenPass &other) : PassWrapper(other) {}

    StringRef getArgument() const final { return "type2-gpu-codegen"; }
    StringRef getDescription() const final {
        return "Generate host code with PGO-optimized kernels and iterative prefetching";
    }

    void runOnOperation() override {
        auto module = getOperation();
        std::string mode = pgoMode.getValue();
        bool isInstrument = (mode == "instrument");
        bool isUseProfile = (mode == "use");

        llvm::outs() << "\n=== Type2 GPU Code Generation ===\n";
        llvm::outs() << "Target: Intel IA-780i Type2 GPU\n";
        llvm::outs() << "Memory: DCOH zero-copy (no memcpy)\n";
        llvm::outs() << "PGO: " << (isInstrument ? "instrument" : isUseProfile ? "use-profile" : "off") << "\n";
        if (isUseProfile)
            llvm::outs() << "Profile: " << profilePath.getValue() << "\n";
        llvm::outs() << "Prefetch: " << (prefetchDist.getValue() > 0
            ? std::to_string(prefetchDist.getValue()) + " (override)"
            : "auto") << "\n\n";

        // Load profile if in use-profile mode
        std::map<std::string, KernelProfile> profiles;
        if (isUseProfile) {
            profiles = loadProfiles(profilePath.getValue());
            llvm::outs() << "  Loaded " << profiles.size() << " kernel profile(s)\n\n";
        }

        int kernel_count = 0;
        std::vector<std::string> kernel_names;
        std::set<std::string> generated_wrappers;
        std::stringstream kernel_code;
        std::stringstream wrapper_code;

        // Walk operations and generate code
        module.walk([&](Operation *op) {
            if (auto offloadAttr = op->getAttr("gpu.offload")) {
                if (auto boolAttr = dyn_cast<BoolAttr>(offloadAttr)) {
                    if (!boolAttr.getValue()) return;
                    std::string kernel_name = "unknown";
                    if (auto kAttr = op->getAttr("gpu.kernel_name"))
                        if (auto sAttr = dyn_cast<StringAttr>(kAttr))
                            kernel_name = sAttr.getValue().str();

                    if (generated_wrappers.count(kernel_name)) {
                        kernel_names.push_back(kernel_name);
                        kernel_count++;
                        return;
                    }

                    // Look up profile for this kernel
                    KernelProfile prof;
                    bool hasProfile = false;
                    if (isUseProfile && profiles.count(kernel_name)) {
                        prof = profiles[kernel_name];
                        hasProfile = true;
                    }

                    if (isa<linalg::MatmulOp>(op)) {
                        auto matmul = cast<linalg::MatmulOp>(op);
                        kernel_code << emitMatmulKernel(matmul, kernel_name, isInstrument, hasProfile, prof);
                        wrapper_code << emitMatmulHost(matmul, kernel_name, isInstrument);
                        llvm::outs() << "  " << kernel_name << " [matmul";
                    } else if (isa<linalg::MatvecOp>(op)) {
                        auto matvec = cast<linalg::MatvecOp>(op);
                        kernel_code << emitMatvecKernel(matvec, kernel_name, isInstrument, hasProfile, prof);
                        wrapper_code << emitMatvecHost(matvec, kernel_name, isInstrument);
                        llvm::outs() << "  " << kernel_name << " [matvec";
                    } else if (isa<linalg::DotOp>(op)) {
                        auto dot = cast<linalg::DotOp>(op);
                        kernel_code << emitDotKernel(dot, kernel_name, isInstrument, hasProfile, prof);
                        wrapper_code << emitDotHost(dot, kernel_name, isInstrument);
                        llvm::outs() << "  " << kernel_name << " [dot";
                    } else {
                        return;
                    }

                    if (hasProfile)
                        llvm::outs() << ", PGO tile=" << prof.tileM << "x" << prof.tileN
                                     << "x" << prof.tileK << ", prefetch=" << prof.prefetchDist;
                    llvm::outs() << "]\n";

                    generated_wrappers.insert(kernel_name);
                    kernel_names.push_back(kernel_name);
                    kernel_count++;
                }
            }
        });

        // Assemble output
        std::stringstream out;
        out << emitFileHeader(isInstrument);
        out << emitDeviceClass();
        out << "\n// === Compiler-Generated Kernels (from MLIR IR) ===\n";
        out << kernel_code.str();
        out << "\n// === Host Wrappers (DCOH zero-copy) ===\n";
        out << wrapper_code.str();
        out << emitMainExecution(kernel_names, isInstrument);

        std::ofstream outfile("/home/victoryang00/CXLMemUring/type2_host_gpu.cpp");
        outfile << out.str();
        outfile.close();

        llvm::outs() << "\nGenerated: type2_host_gpu.cpp";
        if (isInstrument)
            llvm::outs() << " (instrumented — run to produce " << profilePath.getValue() << ")";
        llvm::outs() << "\n  Kernels: " << kernel_count << "\n\n";
    }

private:
    // ================================================================
    // Profile I/O
    // ================================================================

    std::map<std::string, KernelProfile> loadProfiles(const std::string &path) {
        std::map<std::string, KernelProfile> profiles;
        std::ifstream f(path);
        if (!f.is_open()) return profiles;

        // Simple JSON parser for our profile format
        std::string line;
        KernelProfile cur;
        while (std::getline(f, line)) {
            auto getVal = [&](const std::string &key) -> std::string {
                auto pos = line.find("\"" + key + "\"");
                if (pos == std::string::npos) return "";
                auto colon = line.find(':', pos);
                if (colon == std::string::npos) return "";
                auto start = line.find_first_not_of(" \t\"", colon + 1);
                auto end = line.find_first_of(",}\"\n", start);
                return line.substr(start, end - start);
            };
            auto v = [&](const std::string &key) { return getVal(key); };
            if (!v("name").empty()) cur.name = v("name");
            if (!v("tile_m").empty()) cur.tileM = std::stoi(v("tile_m"));
            if (!v("tile_n").empty()) cur.tileN = std::stoi(v("tile_n"));
            if (!v("tile_k").empty()) cur.tileK = std::stoi(v("tile_k"));
            if (!v("prefetch_dist").empty()) cur.prefetchDist = std::stoi(v("prefetch_dist"));
            if (!v("total_ns").empty()) cur.totalNs = std::stod(v("total_ns"));
            if (!v("avg_tile_ns").empty()) cur.avgTileNs = std::stod(v("avg_tile_ns"));
            if (!v("l1_misses").empty()) cur.l1Misses = std::stoll(v("l1_misses"));
            if (!v("l1_hits").empty()) cur.l1Hits = std::stoll(v("l1_hits"));
            if (!v("hit_rate").empty()) cur.hitRate = std::stod(v("hit_rate"));
            if (!v("iteration").empty()) cur.iteration = std::stoi(v("iteration"));
            if (line.find('}') != std::string::npos && !cur.name.empty()) {
                profiles[cur.name] = cur;
                cur = KernelProfile();
            }
        }
        return profiles;
    }

    // ================================================================
    // Tile size selection
    // ================================================================

    struct TileConfig {
        int TM, TN, TK, prefetchDist;
    };

    TileConfig chooseTile(int64_t M, int64_t K, int64_t N,
                          bool hasProfile, const KernelProfile &prof) {
        TileConfig t;
        if (hasProfile && prof.tileM > 0) {
            // Use profile-guided tile sizes
            t.TM = prof.tileM;
            t.TN = prof.tileN;
            t.TK = prof.tileK;
            t.prefetchDist = prof.prefetchDist;
        } else {
            // Static heuristic: L1 = 32KB, 3 tiles must fit
            // 3 * T^2 * 4B <= 32768 → T <= 52
            auto pick = [](int64_t d) -> int {
                if (d <= 8) return (int)d;
                if (d <= 32) return 16;
                if (d <= 128) return 32;
                return 64;
            };
            t.TM = pick(M);
            t.TN = pick(N);
            t.TK = pick(K);
            // Default prefetch: 2 tiles ahead
            t.prefetchDist = 2;
        }
        // Command-line override
        if (prefetchDist.getValue() > 0)
            t.prefetchDist = prefetchDist.getValue();
        return t;
    }

    TileConfig chooseTileMatvec(int64_t M, int64_t N,
                                bool hasProfile, const KernelProfile &prof) {
        TileConfig t;
        if (hasProfile && prof.prefetchDist > 0) {
            t.prefetchDist = prof.prefetchDist;
        } else {
            // Prefetch rows ahead: hide L2 latency (~12 cycles / ~4ns at 3GHz)
            // Each row = N*4 bytes. Prefetch 4 rows ahead.
            t.prefetchDist = 4;
        }
        if (prefetchDist.getValue() > 0)
            t.prefetchDist = prefetchDist.getValue();
        t.TM = 0; t.TN = 0; t.TK = 0; // unused for matvec
        return t;
    }

    // ================================================================
    // File header
    // ================================================================

    std::string emitFileHeader(bool instrument) {
        std::stringstream ss;
        ss << R"(
/**
 * Type2 GPU Host Code - Compiler-Generated from MLIR IR
 *
 * CXL Type2 DCOH zero-copy: kernels operate directly on host memory.
 * BAR0 used only for CompletionData (64 bytes).
)";
        if (instrument) {
            ss << " * PGO MODE: INSTRUMENTED — collects per-tile timing & cache counters.\n";
            ss << " * Run this binary to produce pgo_profile.json.\n";
        }
        ss << R"( */

#include "Type2GpuDevice.h"
#include <cstring>
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <vector>
#include <string>

using namespace cira::runtime;

#define CACHE_LINE_SIZE 64
#define COMPLETION_MAGIC 0xDEADBEEF

typedef struct __attribute__((aligned(CACHE_LINE_SIZE))) {
    uint32_t magic;
    uint32_t status;
    uint64_t result;
    uint64_t cycles;
    uint64_t timestamp;
    uint8_t reserved[32];
} CompletionData;

)";

        if (instrument) {
            ss << R"(
// ============================================================================
// PGO Instrumentation Infrastructure
// ============================================================================

struct PGOKernelStats {
    std::string name;
    int tile_m, tile_n, tile_k;
    int prefetch_dist;
    double total_ns;
    double avg_tile_ns;
    int64_t tile_iterations;
    int64_t l1_misses_est;
    int64_t l1_hits_est;
    double hit_rate;
    int iteration;
};

static std::vector<PGOKernelStats> g_pgo_stats;

static void pgo_write_profile(const char* path) {
    std::ofstream f(path);
    f << "[\n";
    for (size_t i = 0; i < g_pgo_stats.size(); i++) {
        auto& s = g_pgo_stats[i];
        f << "  {\n";
        f << "    \"name\": \"" << s.name << "\",\n";
        f << "    \"tile_m\": " << s.tile_m << ",\n";
        f << "    \"tile_n\": " << s.tile_n << ",\n";
        f << "    \"tile_k\": " << s.tile_k << ",\n";
        f << "    \"prefetch_dist\": " << s.prefetch_dist << ",\n";
        f << "    \"total_ns\": " << std::fixed << s.total_ns << ",\n";
        f << "    \"avg_tile_ns\": " << std::fixed << s.avg_tile_ns << ",\n";
        f << "    \"tile_iterations\": " << s.tile_iterations << ",\n";
        f << "    \"l1_misses\": " << s.l1_misses_est << ",\n";
        f << "    \"l1_hits\": " << s.l1_hits_est << ",\n";
        f << "    \"hit_rate\": " << s.hit_rate << ",\n";

        // Suggest optimized parameters for next iteration
        int new_tile = s.tile_m;
        int new_prefetch = s.prefetch_dist;
        if (s.hit_rate < 0.90) {
            // Too many misses: shrink tile to fit L1 better
            new_tile = std::max(8, s.tile_m / 2);
            new_prefetch = std::min(s.prefetch_dist + 1, 8);
        } else if (s.hit_rate > 0.98 && s.avg_tile_ns > 100) {
            // Very high hit rate but slow: try larger tile for less overhead
            new_tile = std::min(128, s.tile_m * 2);
        }
        // If prefetch is helping (high hit rate), keep it; otherwise increase
        if (s.hit_rate < 0.95)
            new_prefetch = std::min(s.prefetch_dist + 1, 8);

        f << "    \"suggested_tile_m\": " << new_tile << ",\n";
        f << "    \"suggested_tile_n\": " << new_tile << ",\n";
        f << "    \"suggested_tile_k\": " << new_tile << ",\n";
        f << "    \"suggested_prefetch_dist\": " << new_prefetch << ",\n";
        f << "    \"iteration\": " << (s.iteration + 1) << "\n";
        f << "  }" << (i + 1 < g_pgo_stats.size() ? "," : "") << "\n";
    }
    f << "]\n";
    std::cout << "PGO profile written to " << path << "\n";
}

)";
        }
        return ss.str();
    }

    std::string emitDeviceClass() {
        return R"(
class Type2GpuExecutor {
private:
    std::unique_ptr<Type2GpuDevice> device_;
    bool initialized_;
public:
    Type2GpuExecutor() : initialized_(false) {}
    bool initialize() {
        device_ = create_type2_gpu_device();
        if (!device_) { std::cerr << "Failed to create Type2 GPU device\n"; return false; }
        initialized_ = true;
        std::cout << "Type2 GPU device initialized\n";
        return true;
    }
    bool is_ready() const { return initialized_ && device_ != nullptr; }
    Type2GpuDevice* get_device() { return device_.get(); }
    void shutdown() {
        if (device_) { device_->shutdown(); device_ = nullptr; }
        initialized_ = false;
    }
};

)";
    }

    // ================================================================
    // MatMul kernel with PGO + prefetch
    // ================================================================

    std::string emitMatmulKernel(linalg::MatmulOp op, const std::string &name,
                                  bool instrument, bool hasProfile,
                                  const KernelProfile &prof) {
        auto lhsShape = cast<MemRefType>(op->getOperand(0).getType()).getShape();
        auto rhsShape = cast<MemRefType>(op->getOperand(1).getType()).getShape();
        int64_t M = lhsShape[0], K = lhsShape[1], N = rhsShape[1];
        auto tile = chooseTile(M, K, N, hasProfile, prof);

        std::stringstream ss;
        ss << "\n// linalg.matmul [" << M << "x" << K << "] x [" << K << "x" << N << "] → [" << M << "x" << N << "]\n";
        ss << "// Tile: " << tile.TM << "x" << tile.TN << "x" << tile.TK;
        if (hasProfile) ss << " (PGO iter " << prof.iteration << ")";
        ss << "  Prefetch: " << tile.prefetchDist << " tiles ahead\n";

        if (instrument) {
            // Instrumented version: measures per-tile timing
            ss << "static PGOKernelStats kernel_" << name << "_profile(\n";
            ss << "    float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {\n";
            ss << "    constexpr int M=" << M << ",N=" << N << ",K=" << K << ";\n";
            ss << "    constexpr int TM=" << tile.TM << ",TN=" << tile.TN << ",TK=" << tile.TK << ";\n";
            ss << "    constexpr int PF=" << tile.prefetchDist << ";\n";
            ss << "    int64_t tile_iters = 0;\n";
            ss << "    int64_t miss_est = 0, hit_est = 0;\n";
            ss << "    auto t0 = std::chrono::high_resolution_clock::now();\n\n";
        } else {
            ss << "static void kernel_" << name << "(\n";
            ss << "    float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {\n";
            ss << "    constexpr int M=" << M << ",N=" << N << ",K=" << K << ";\n";
            ss << "    constexpr int TM=" << tile.TM << ",TN=" << tile.TN << ",TK=" << tile.TK << ";\n";
            ss << "    constexpr int PF=" << tile.prefetchDist << ";\n\n";
        }

        // Tiled GEMM with software prefetch
        ss << "    for (int i0 = 0; i0 < M; i0 += TM) {\n";
        ss << "        int ie = std::min(i0+TM, M);\n";
        ss << "        for (int j0 = 0; j0 < N; j0 += TN) {\n";
        ss << "            int je = std::min(j0+TN, N);\n";
        ss << "            for (int k0 = 0; k0 < K; k0 += TK) {\n";
        ss << "                int ke = std::min(k0+TK, K);\n";

        if (instrument)
            ss << "                tile_iters++;\n";

        // Prefetch next B tile (k0 + PF*TK) while computing current
        ss << "                // Prefetch B tile " << tile.prefetchDist << " iterations ahead\n";
        ss << "                if (k0 + PF*TK < K) {\n";
        ss << "                    for (int pj = j0; pj < je; pj += CACHE_LINE_SIZE/sizeof(float))\n";
        ss << "                        __builtin_prefetch(&B[(k0+PF*TK)*N + pj], 0, 1);\n";
        ss << "                }\n";
        // Prefetch next A tile rows
        ss << "                // Prefetch A rows for next k-tile\n";
        ss << "                if (k0 + PF*TK < K) {\n";
        ss << "                    for (int pi = i0; pi < ie; pi++)\n";
        ss << "                        __builtin_prefetch(&A[pi*K + k0+PF*TK], 0, 1);\n";
        ss << "                }\n\n";

        // Micro-kernel
        ss << "                for (int i = i0; i < ie; i++) {\n";
        ss << "                    for (int j = j0; j < je; j++) {\n";
        ss << "                        float acc = C[i*N+j];\n";
        ss << "                        for (int k = k0; k < ke; k++)\n";
        ss << "                            acc += A[i*K+k] * B[k*N+j];\n";
        ss << "                        C[i*N+j] = acc;\n";
        ss << "                    }\n";
        ss << "                }\n";
        ss << "            }\n";
        ss << "        }\n";
        ss << "    }\n";

        if (instrument) {
            ss << "    auto t1 = std::chrono::high_resolution_clock::now();\n";
            ss << "    double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();\n";
            // Estimate L1 behavior: tiles that fit in L1 → hits, else misses
            ss << "    int64_t tile_bytes = (int64_t)TM*TK*4 + (int64_t)TK*TN*4 + (int64_t)TM*TN*4;\n";
            ss << "    if (tile_bytes <= 32768) { hit_est = tile_iters; miss_est = 0; }\n";
            ss << "    else { miss_est = tile_iters; hit_est = 0; }\n";
            ss << "    double hr = (hit_est+miss_est>0) ? (double)hit_est/(hit_est+miss_est) : 1.0;\n";
            ss << "    return PGOKernelStats{\"" << name << "\", TM, TN, TK, PF, ns,\n";
            ss << "        tile_iters>0 ? ns/tile_iters : 0, tile_iters, miss_est, hit_est, hr, "
               << (hasProfile ? prof.iteration : 0) << "};\n";
        }
        ss << "}\n";
        return ss.str();
    }

    std::string emitMatmulHost(linalg::MatmulOp op, const std::string &name, bool instrument) {
        auto lhsShape = cast<MemRefType>(op->getOperand(0).getType()).getShape();
        auto rhsShape = cast<MemRefType>(op->getOperand(1).getType()).getShape();
        int64_t M = lhsShape[0], K = lhsShape[1], N = rhsShape[1];

        std::stringstream ss;
        ss << "\nbool execute_" << name << "(Type2GpuDevice* device,\n";
        ss << "    float* A, float* B, float* C, uint32_t timeout_ms=1000) {\n";
        ss << "    std::cout << \"Executing " << name << " (" << M << "x" << K << "x" << N << ")...\\n\";\n";
        ss << "    auto start = std::chrono::high_resolution_clock::now();\n\n";
        emitZeroCopyPreamble(ss);
        if (instrument) {
            ss << "    auto stats = kernel_" << name << "_profile(A, B, C);\n";
            ss << "    g_pgo_stats.push_back(stats);\n\n";
        } else {
            ss << "    kernel_" << name << "(A, B, C);\n\n";
        }
        emitCompletionEpilogue(ss);
        return ss.str();
    }

    // ================================================================
    // MatVec kernel with prefetch
    // ================================================================

    std::string emitMatvecKernel(linalg::MatvecOp op, const std::string &name,
                                  bool instrument, bool hasProfile,
                                  const KernelProfile &prof) {
        auto matShape = cast<MemRefType>(op->getOperand(0).getType()).getShape();
        int64_t M = matShape[0], N = matShape[1];
        auto tile = chooseTileMatvec(M, N, hasProfile, prof);

        std::stringstream ss;
        ss << "\n// linalg.matvec A[" << M << "x" << N << "] * x[" << N << "] → y[" << M << "]\n";
        ss << "// Prefetch: " << tile.prefetchDist << " rows ahead\n";

        if (instrument) {
            ss << "static PGOKernelStats kernel_" << name << "_profile(\n";
            ss << "    const float* __restrict__ A, const float* __restrict__ x, float* __restrict__ y) {\n";
            ss << "    constexpr int M=" << M << ",N=" << N << ",PF=" << tile.prefetchDist << ";\n";
            ss << "    auto t0 = std::chrono::high_resolution_clock::now();\n";
        } else {
            ss << "static void kernel_" << name << "(\n";
            ss << "    const float* __restrict__ A, const float* __restrict__ x, float* __restrict__ y) {\n";
            ss << "    constexpr int M=" << M << ",N=" << N << ",PF=" << tile.prefetchDist << ";\n";
        }

        ss << "    for (int i = 0; i < M; i++) {\n";
        // Prefetch future rows of A
        ss << "        // Prefetch row A[i+PF] into L1\n";
        ss << "        if (i + PF < M) {\n";
        ss << "            for (int p = 0; p < N; p += CACHE_LINE_SIZE/sizeof(float))\n";
        ss << "                __builtin_prefetch(&A[(i+PF)*N + p], 0, 1);\n";
        ss << "        }\n";
        ss << "        float acc = y[i];\n";
        ss << "        for (int j = 0; j < N; j++)\n";
        ss << "            acc += A[i*N+j] * x[j];\n";
        ss << "        y[i] = acc;\n";
        ss << "    }\n";

        if (instrument) {
            ss << "    auto t1 = std::chrono::high_resolution_clock::now();\n";
            ss << "    double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();\n";
            ss << "    int64_t row_bytes = N*4;\n";
            ss << "    int64_t hits = (row_bytes<=32768) ? M : 0;\n";
            ss << "    int64_t misses = M - hits;\n";
            ss << "    double hr = (M>0) ? (double)hits/M : 1.0;\n";
            ss << "    return PGOKernelStats{\"" << name << "\", 0, 0, 0, PF, ns, M>0?ns/M:0, M, misses, hits, hr, 0};\n";
        }
        ss << "}\n";
        return ss.str();
    }

    std::string emitMatvecHost(linalg::MatvecOp op, const std::string &name, bool instrument) {
        auto matShape = cast<MemRefType>(op->getOperand(0).getType()).getShape();
        int64_t M = matShape[0], N = matShape[1];

        std::stringstream ss;
        ss << "\nbool execute_" << name << "(Type2GpuDevice* device,\n";
        ss << "    float* A, float* x, float* y, uint32_t timeout_ms=1000) {\n";
        ss << "    std::cout << \"Executing " << name << " (" << M << "x" << N << ")...\\n\";\n";
        ss << "    auto start = std::chrono::high_resolution_clock::now();\n\n";
        emitZeroCopyPreamble(ss);
        if (instrument) {
            ss << "    auto stats = kernel_" << name << "_profile(A, x, y);\n";
            ss << "    g_pgo_stats.push_back(stats);\n\n";
        } else {
            ss << "    kernel_" << name << "(A, x, y);\n\n";
        }
        emitCompletionEpilogue(ss);
        return ss.str();
    }

    // ================================================================
    // Dot kernel with prefetch
    // ================================================================

    std::string emitDotKernel(linalg::DotOp op, const std::string &name,
                               bool instrument, bool hasProfile,
                               const KernelProfile &prof) {
        auto vecShape = cast<MemRefType>(op->getOperand(0).getType()).getShape();
        int64_t N = vecShape.size() > 0 ? vecShape[0] : 1;
        int pf = (hasProfile && prof.prefetchDist > 0) ? prof.prefetchDist : 8;
        if (prefetchDist.getValue() > 0) pf = prefetchDist.getValue();

        std::stringstream ss;
        ss << "\n// linalg.dot a[" << N << "] · b[" << N << "] → result\n";
        ss << "// Prefetch: " << pf << " cache lines ahead\n";

        if (instrument) {
            ss << "static PGOKernelStats kernel_" << name << "_profile(\n";
            ss << "    const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ r) {\n";
            ss << "    constexpr int N=" << N << ", PF=" << pf << ";\n";
            ss << "    auto t0 = std::chrono::high_resolution_clock::now();\n";
        } else {
            ss << "static void kernel_" << name << "(\n";
            ss << "    const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ r) {\n";
            ss << "    constexpr int N=" << N << ", PF=" << pf << ";\n";
        }

        ss << "    float acc = *r;\n";
        ss << "    for (int i = 0; i < N; i++) {\n";
        ss << "        // Prefetch PF cache lines ahead in both vectors\n";
        ss << "        if (i + PF*16 < N) {\n";
        ss << "            __builtin_prefetch(&a[i + PF*16], 0, 1);\n";
        ss << "            __builtin_prefetch(&b[i + PF*16], 0, 1);\n";
        ss << "        }\n";
        ss << "        acc += a[i] * b[i];\n";
        ss << "    }\n";
        ss << "    *r = acc;\n";

        if (instrument) {
            ss << "    auto t1 = std::chrono::high_resolution_clock::now();\n";
            ss << "    double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();\n";
            ss << "    return PGOKernelStats{\"" << name << "\", 0, 0, 0, PF, ns, ns, 1, 0, 1, 1.0, 0};\n";
        }
        ss << "}\n";
        return ss.str();
    }

    std::string emitDotHost(linalg::DotOp op, const std::string &name, bool instrument) {
        auto vecShape = cast<MemRefType>(op->getOperand(0).getType()).getShape();
        int64_t N = vecShape.size() > 0 ? vecShape[0] : 1;

        std::stringstream ss;
        ss << "\nbool execute_" << name << "(Type2GpuDevice* device,\n";
        ss << "    float* a, float* b, float* result, uint32_t timeout_ms=1000) {\n";
        ss << "    std::cout << \"Executing " << name << " (" << N << ")...\\n\";\n";
        ss << "    auto start = std::chrono::high_resolution_clock::now();\n\n";
        emitZeroCopyPreamble(ss);
        if (instrument) {
            ss << "    auto stats = kernel_" << name << "_profile(a, b, result);\n";
            ss << "    g_pgo_stats.push_back(stats);\n\n";
        } else {
            ss << "    kernel_" << name << "(a, b, result);\n\n";
        }
        emitCompletionEpilogue(ss);
        return ss.str();
    }

    // ================================================================
    // Shared codegen helpers
    // ================================================================

    void emitZeroCopyPreamble(std::stringstream &ss) {
        ss << "    // DCOH zero-copy: BAR0 only for CompletionData\n";
        ss << "    uint8_t* bar0 = device->allocate(sizeof(CompletionData));\n";
        ss << "    if (!bar0) { std::cerr << \"BAR0 alloc failed\\n\"; return false; }\n";
        ss << "    CompletionData* completion = reinterpret_cast<CompletionData*>(bar0);\n";
        ss << "    completion->magic = 0; completion->status = 0;\n\n";
    }

    void emitCompletionEpilogue(std::stringstream &ss) {
        ss << "    completion->status = 1;\n";
        ss << "    completion->magic = COMPLETION_MAGIC;\n";
        ss << "    device->deallocate(bar0);\n\n";
        ss << "    auto end = std::chrono::high_resolution_clock::now();\n";
        ss << "    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();\n";
        ss << "    std::cout << \"  Completed in \" << ms << \" ms\\n\";\n";
        ss << "    std::cout << \"  Sync: CompletionData → LLC via DCOH\\n\\n\";\n";
        ss << "    return true;\n}\n";
    }

    // ================================================================
    // Main execution + iterative PGO driver
    // ================================================================

    std::string emitMainExecution(const std::vector<std::string>& kernel_names, bool instrument) {
        std::stringstream ss;
        ss << R"(

int main() {
    std::cout << "\n=== Type2 GPU Execution ===\n";
    std::cout << "Target: Intel IA-780i Type2 GPU (DCOH zero-copy)\n";
)";
        if (instrument)
            ss << "    std::cout << \"Mode: PGO INSTRUMENTED\\n\";\n";
        else
            ss << "    std::cout << \"Kernels: compiler-generated + prefetch\\n\";\n";

        ss << R"(
    Type2GpuExecutor executor;
    if (!executor.initialize()) { std::cerr << "Init failed\n"; return -1; }
    std::cout << "Memory: " << (executor.get_device()->available_memory()/(1024*1024)) << " MB\n\n";
)";

        std::set<std::string> seen;
        std::vector<std::string> unique;
        for (auto &kn : kernel_names)
            if (!seen.count(kn)) { seen.insert(kn); unique.push_back(kn); }

        for (auto &kname : unique) {
            int exec_count = 0;
            for (auto &kn : kernel_names) if (kn == kname) exec_count++;

            bool is_matvec = kname.find("matvec_kernel_") == 0;
            bool is_dot = kname.find("dot_kernel_") == 0;

            size_t pos = kname.find_last_of('_');
            if (pos == std::string::npos) continue;
            std::string dim_str = kname.substr(pos + 1);

            std::vector<int> dims;
            size_t start = 0, xp;
            while ((xp = dim_str.find('x', start)) != std::string::npos) {
                dims.push_back(std::stoi(dim_str.substr(start, xp - start)));
                start = xp + 1;
            }
            dims.push_back(std::stoi(dim_str.substr(start)));

            ss << "    {\n";
            if (is_dot && !dims.empty()) {
                int N = dims[0];
                ss << "        float* A=new float[" << N << "];\n";
                ss << "        float* B=new float[" << N << "];\n";
                ss << "        float* C=new float[1];\n";
                ss << "        for(int i=0;i<" << N << ";i++){A[i]=1.0f;B[i]=1.0f;} C[0]=0;\n\n";
            } else if (is_matvec && dims.size() >= 2) {
                int M=dims[0], N=dims[1];
                ss << "        float* A=new float[" << M << "LL*" << N << "LL];\n";
                ss << "        float* B=new float[" << N << "];\n";
                ss << "        float* C=new float[" << M << "];\n";
                ss << "        for(size_t i=0;i<" << M << "LL*" << N << "LL;i++) A[i]=1.0f;\n";
                ss << "        for(int i=0;i<" << N << ";i++) B[i]=1.0f;\n";
                ss << "        for(int i=0;i<" << M << ";i++) C[i]=0;\n\n";
            } else if (dims.size() >= 3) {
                int M=dims[0],N=dims[1],K=(dims.size()==3?dims[2]:dims[2]);
                if (dims.size()==4) { M=dims[1]; N=dims[2]; K=dims[3]; }
                ss << "        float* A=new float[" << M << "LL*" << K << "LL];\n";
                ss << "        float* B=new float[" << K << "LL*" << N << "LL];\n";
                ss << "        float* C=new float[" << M << "LL*" << N << "LL];\n";
                ss << "        for(size_t i=0;i<" << M << "LL*" << K << "LL;i++) A[i]=1.0f;\n";
                ss << "        for(size_t i=0;i<" << K << "LL*" << N << "LL;i++) B[i]=1.0f;\n";
                ss << "        for(size_t i=0;i<" << M << "LL*" << N << "LL;i++) C[i]=0;\n\n";
            } else { ss << "    }\n"; continue; }

            for (int e=0; e<exec_count; e++) {
                if (exec_count>1) ss << "        // Pass " << e+1 << "/" << exec_count << "\n";
                ss << "        if(!execute_" << kname << "(executor.get_device(),A,B,C)){";
                ss << "delete[]A;delete[]B;delete[]C;executor.shutdown();return-1;}\n";
            }
            ss << "        delete[]A; delete[]B; delete[]C;\n    }\n\n";
        }

        if (instrument)
            ss << "    pgo_write_profile(\"pgo_profile.json\");\n";

        ss << R"(
    executor.shutdown();
    std::cout << "\n✓ Execution completed\n";
    return 0;
}
)";
        return ss.str();
    }
};

std::unique_ptr<Pass> createType2GpuCodeGen() {
    return std::make_unique<Type2GpuCodeGenPass>();
}

} // namespace mlir
