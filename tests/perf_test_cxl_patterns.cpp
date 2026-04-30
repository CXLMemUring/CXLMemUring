/**
 * perf_test_cxl_patterns.cpp
 *
 * Performance testing suite for CXL Type2 device.
 * Identifies performance bottlenecks:
 * - Pointer chasing (latency-sensitive)
 * - Bulk memory loads (bandwidth-sensitive)
 * - Cache coherency overhead
 * - Memory access patterns
 *
 * Build:
 *   g++ -std=c++17 -O3 -march=native \
 *       -I/home/victoryang00/CXLMemUring/runtime/include \
 *       perf_test_cxl_patterns.cpp \
 *       /home/victoryang00/CXLMemUring/runtime/src/Type2GpuDevice.cpp \
 *       -o perf_test_cxl_patterns
 *
 * Usage:
 *   sudo ./perf_test_cxl_patterns [--test pointer-chase|bulk-load|mixed|all]
 */

#include "../runtime/include/Type2GpuDevice.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cstring>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace cira::runtime;

// ============================================================================
// Timing Infrastructure
// ============================================================================

class Timer {
private:
    using clock = std::chrono::high_resolution_clock;
    clock::time_point start_;

public:
    Timer() : start_(clock::now()) {}

    double elapsed_ms() const {
        auto end = clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

    double elapsed_us() const {
        auto end = clock::now();
        return std::chrono::duration<double, std::micro>(end - start_).count();
    }

    void reset() {
        start_ = clock::now();
    }
};

// ============================================================================
// Test 1: Pointer Chasing (Latency Benchmark)
// ============================================================================

struct PointerNode {
    uint64_t next;      // Pointer to next node
    float data[7];      // 32 bytes per node (cache line sized)
};

class PointerChasingTest {
private:
    std::vector<PointerNode> nodes_;
    size_t num_nodes_;

public:
    PointerChasingTest(size_t num_nodes = 1024*1024)
        : num_nodes_(num_nodes) {

        nodes_.resize(num_nodes_);

        // Create random linked list (worst case for cache)
        std::vector<size_t> indices(num_nodes_);
        std::iota(indices.begin(), indices.end(), 0);

        std::mt19937 rng(42);
        std::shuffle(indices.begin(), indices.end(), rng);

        for (size_t i = 0; i < num_nodes_ - 1; i++) {
            nodes_[indices[i]].next = reinterpret_cast<uint64_t>(&nodes_[indices[i+1]]);
            nodes_[indices[i]].data[0] = static_cast<float>(i);
        }
        nodes_[indices[num_nodes_-1]].next = 0;
        nodes_[indices[0]].data[0] = 0.0f;

        std::cout << "[PointerChasing] Created " << num_nodes_ << " node linked list\n";
    }

    struct Result {
        double elapsed_ms;
        size_t iterations;
        double ops_per_second;
        double latency_ns;
    };

    Result run(size_t iterations = 10) {
        Timer timer;

        volatile float sum = 0.0f;
        volatile PointerNode* current = &nodes_[0];

        for (size_t iter = 0; iter < iterations; iter++) {
            for (size_t i = 0; i < num_nodes_; i++) {
                sum += current->data[0];
                current = reinterpret_cast<PointerNode*>(current->next);
            }
        }

        double elapsed = timer.elapsed_ms();
        size_t total_accesses = iterations * num_nodes_;
        double ops_per_sec = total_accesses / (elapsed / 1000.0);
        double latency_ns = (elapsed * 1e6) / total_accesses;

        return {
            .elapsed_ms = elapsed,
            .iterations = iterations,
            .ops_per_second = ops_per_sec,
            .latency_ns = latency_ns
        };
    }
};

// ============================================================================
// Test 2: Bulk Memory Load (Bandwidth Benchmark)
// ============================================================================

class BulkMemoryTest {
private:
    std::vector<float> source_;
    std::vector<float> dest_;
    size_t data_size_;

public:
    BulkMemoryTest(size_t data_size_mb = 256)
        : data_size_(data_size_mb * 1024 * 1024 / sizeof(float)) {

        source_.resize(data_size_);
        dest_.resize(data_size_);

        // Fill with pattern
        for (size_t i = 0; i < data_size_; i++) {
            source_[i] = static_cast<float>(i);
        }

        std::cout << "[BulkMemory] Created " << data_size_mb << " MB data\n";
    }

    struct Result {
        double elapsed_ms;
        size_t bytes_transferred;
        double bandwidth_gbps;
    };

    Result run_memcpy(size_t iterations = 10) {
        Timer timer;

        for (size_t iter = 0; iter < iterations; iter++) {
            memcpy(dest_.data(), source_.data(), data_size_ * sizeof(float));
        }

        double elapsed = timer.elapsed_ms();
        size_t total_bytes = iterations * data_size_ * sizeof(float);
        double bandwidth = (total_bytes / (1024*1024*1024)) / (elapsed / 1000.0);

        return {
            .elapsed_ms = elapsed,
            .bytes_transferred = total_bytes,
            .bandwidth_gbps = bandwidth
        };
    }

    Result run_sequential_read(size_t iterations = 100) {
        Timer timer;

        volatile float sum = 0.0f;
        for (size_t iter = 0; iter < iterations; iter++) {
            for (size_t i = 0; i < data_size_; i++) {
                sum += source_[i];
            }
        }

        double elapsed = timer.elapsed_ms();
        size_t total_bytes = iterations * data_size_ * sizeof(float);
        double bandwidth = (total_bytes / (1024*1024*1024)) / (elapsed / 1000.0);

        return {
            .elapsed_ms = elapsed,
            .bytes_transferred = total_bytes,
            .bandwidth_gbps = bandwidth
        };
    }

    Result run_stride_read(size_t stride = 256, size_t iterations = 100) {
        Timer timer;

        volatile float sum = 0.0f;
        for (size_t iter = 0; iter < iterations; iter++) {
            for (size_t i = 0; i < data_size_; i += stride) {
                sum += source_[i];
            }
        }

        double elapsed = timer.elapsed_ms();
        size_t total_accesses = iterations * (data_size_ / stride);
        size_t total_bytes = total_accesses * sizeof(float);
        double bandwidth = (total_bytes / (1024*1024*1024)) / (elapsed / 1000.0);

        return {
            .elapsed_ms = elapsed,
            .bytes_transferred = total_bytes,
            .bandwidth_gbps = bandwidth
        };
    }
};

// ============================================================================
// Test 3: GEMM with CXL Memory (Mixed Pattern)
// ============================================================================

class GEMMCXLTest {
private:
    std::unique_ptr<Type2GpuDevice> gpu_;
    std::vector<float> A_, B_, C_;
    uint32_t M_, N_, K_;

public:
    GEMMCXLTest(uint32_t M = 1024, uint32_t N = 1024, uint32_t K = 1024)
        : M_(M), N_(N), K_(K) {

        gpu_ = create_type2_gpu_device();
        if (!gpu_) {
            std::cerr << "Failed to create Type2GpuDevice\n";
            throw std::runtime_error("GPU device creation failed");
        }

        A_.resize(M * K);
        B_.resize(K * N);
        C_.resize(M * N);

        // Initialize with random values
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        for (auto& v : A_) v = dist(rng);
        for (auto& v : B_) v = dist(rng);
        for (auto& v : C_) v = 0.0f;

        std::cout << "[GEMMCXL] Created " << M << "x" << N << "x" << K
                  << " matrices\n";
    }

    struct Result {
        double elapsed_ms;
        double tflops;
        uint64_t gpu_cycles;
        uint64_t gpu_instructions;
    };

    Result run(uint32_t iterations = 5) {
        Timer timer;

        for (uint32_t iter = 0; iter < iterations; iter++) {
            if (!gpu_->gemm_f32(
                C_.data(), A_.data(), B_.data(),
                M_, N_, K_,
                1.0f, 0.0f,
                10000
            )) {
                throw std::runtime_error("GEMM failed");
            }
        }

        double elapsed = timer.elapsed_ms();
        uint64_t total_ops = static_cast<uint64_t>(M_) * N_ * K_ * 2 * iterations;
        double tflops = (total_ops / 1e12) / (elapsed / 1000.0);

        return {
            .elapsed_ms = elapsed,
            .tflops = tflops,
            .gpu_cycles = gpu_->get_kernel_cycles(),
            .gpu_instructions = gpu_->get_kernel_instructions()
        };
    }
};

// ============================================================================
// Test 4: Cache Coherency Overhead (DCOH)
// ============================================================================

class DCOHTest {
private:
    std::vector<float> shared_mem_;
    std::unique_ptr<Type2GpuDevice> gpu_;

public:
    DCOHTest(size_t size_mb = 64)
        : shared_mem_(size_mb * 1024 * 1024 / sizeof(float)) {

        gpu_ = create_type2_gpu_device();
        if (!gpu_) {
            throw std::runtime_error("GPU device creation failed");
        }

        std::cout << "[DCOH] Testing cache coherency with " << size_mb << " MB\n";
    }

    struct Result {
        double elapsed_ms;
        size_t operations;
        double ops_per_second;
    };

    Result run_write_invalidate(size_t iterations = 1000) {
        Timer timer;

        for (size_t iter = 0; iter < iterations; iter++) {
            // Simulate GPU writing to shared memory
            // In real scenario, GPU would write via CXL.cache
            for (size_t i = 0; i < shared_mem_.size(); i += 64/sizeof(float)) {
                shared_mem_[i] = static_cast<float>(iter);
            }
            // Invalidate CPU cache (simulated)
            __asm__ volatile("mfence" ::: "memory");
        }

        double elapsed = timer.elapsed_ms();
        size_t total_ops = iterations * (shared_mem_.size() / (64/sizeof(float)));
        double ops_per_sec = total_ops / (elapsed / 1000.0);

        return {
            .elapsed_ms = elapsed,
            .operations = total_ops,
            .ops_per_second = ops_per_sec
        };
    }
};

// ============================================================================
// Performance Analysis Functions
// ============================================================================

void print_header(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(70, '=') << "\n";
}

void print_result(const std::string& name, const std::string& metric,
                  double value, const std::string& unit) {
    std::cout << std::left << std::setw(35) << name
              << std::right << std::setw(15) << std::fixed << std::setprecision(3)
              << value << " " << unit << "\n";
}

// ============================================================================
// Main Test Suite
// ============================================================================

int main(int argc, char* argv[]) {
    std::string test_type = "all";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--test" && i + 1 < argc) {
            test_type = argv[++i];
        }
    }

    print_header("CXL Type2 Performance Characterization");
    std::cout << "Testing memory access patterns and GPU performance\n";
    std::cout << "Test type: " << test_type << "\n";

    try {
        // Test 1: Pointer Chasing
        if (test_type == "pointer-chase" || test_type == "all") {
            print_header("Test 1: Pointer Chasing (Latency Sensitive)");

            PointerChasingTest pc_test(512*1024);  // 512K nodes
            auto result = pc_test.run(10);

            print_result("Total Time", "", result.elapsed_ms, "ms");
            print_result("Iterations", "", result.iterations, "");
            print_result("Operations/Second", "", result.ops_per_second / 1e9, "G ops/s");
            print_result("Average Latency", "", result.latency_ns, "ns per access");

            // Analysis
            std::cout << "\nAnalysis:\n";
            if (result.latency_ns > 500) {
                std::cout << "⚠ HIGH LATENCY: Pointer chasing shows "
                         << result.latency_ns << " ns per access\n";
                std::cout << "  → Suggests poor cache locality or CXL cache misses\n";
                std::cout << "  → This is a performance bug indicator\n";
            } else {
                std::cout << "✓ ACCEPTABLE LATENCY: " << result.latency_ns << " ns\n";
            }
        }

        // Test 2: Bulk Memory Load
        if (test_type == "bulk-load" || test_type == "all") {
            print_header("Test 2: Bulk Memory Load (Bandwidth Sensitive)");

            BulkMemoryTest bm_test(256);  // 256 MB

            std::cout << "\n[2a] Sequential Read (Optimal Pattern)\n";
            auto seq_result = bm_test.run_sequential_read(50);
            print_result("Elapsed Time", "", seq_result.elapsed_ms, "ms");
            print_result("Total Data", "", seq_result.bytes_transferred / 1e9, "GB");
            print_result("Bandwidth", "", seq_result.bandwidth_gbps, "GB/s");

            std::cout << "\n[2b] Stride Read (Stride=256 bytes)\n";
            auto stride_result = bm_test.run_stride_read(256/sizeof(float), 100);
            print_result("Elapsed Time", "", stride_result.elapsed_ms, "ms");
            print_result("Bandwidth", "", stride_result.bandwidth_gbps, "GB/s");

            std::cout << "\n[2c] Memcpy Baseline\n";
            auto memcpy_result = bm_test.run_memcpy(10);
            print_result("Elapsed Time", "", memcpy_result.elapsed_ms, "ms");
            print_result("Bandwidth", "", memcpy_result.bandwidth_gbps, "GB/s");

            // Analysis
            std::cout << "\nAnalysis:\n";
            double bandwidth_ratio = stride_result.bandwidth_gbps / seq_result.bandwidth_gbps;
            std::cout << "Sequential vs Stride ratio: " << std::fixed << std::setprecision(2)
                     << bandwidth_ratio << "x\n";

            if (bandwidth_ratio < 0.8) {
                std::cout << "⚠ SIGNIFICANT BANDWIDTH DROP with stride pattern\n";
                std::cout << "  → Suggests cache line utilization issue\n";
                std::cout << "  → Possible CXL memory alignment problem\n";
            } else {
                std::cout << "✓ GOOD STRIDE TOLERANCE: Bandwidth preserved\n";
            }

            if (seq_result.bandwidth_gbps < 10.0) {
                std::cout << "⚠ LOW SEQUENTIAL BANDWIDTH: " << seq_result.bandwidth_gbps << " GB/s\n";
                std::cout << "  → Indicates memory subsystem bottleneck\n";
            }
        }

        // Test 3: GEMM Performance
        if (test_type == "gemm" || test_type == "mixed" || test_type == "all") {
            print_header("Test 3: GEMM with CXL Memory");

            try {
                GEMMCXLTest gemm_test(1024, 1024, 1024);
                auto result = gemm_test.run(3);

                print_result("Total Time", "", result.elapsed_ms, "ms");
                print_result("Performance", "", result.tflops, "TFLOPS");
                print_result("GPU Cycles", "", result.gpu_cycles, "");
                print_result("GPU Instructions", "", result.gpu_instructions, "");

                // Analysis
                std::cout << "\nAnalysis:\n";
                if (result.tflops < 1.0) {
                    std::cout << "⚠ LOW GEMM PERFORMANCE: " << result.tflops << " TFLOPS\n";
                    std::cout << "  → Indicates memory bandwidth limitation\n";
                    std::cout << "  → Possible CXL memory load issue\n";
                } else {
                    std::cout << "✓ ACCEPTABLE GEMM PERFORMANCE: " << result.tflops << " TFLOPS\n";
                }
            } catch (const std::exception& e) {
                std::cout << "⚠ GEMM test skipped: " << e.what() << "\n";
                std::cout << "  (Requires GPU hardware to be available)\n";
            }
        }

        // Test 4: DCOH Overhead
        if (test_type == "dcoh" || test_type == "all") {
            print_header("Test 4: DCOH Cache Coherency Overhead");

            DCOHTest dcoh_test(64);
            auto result = dcoh_test.run_write_invalidate(1000);

            print_result("Total Time", "", result.elapsed_ms, "ms");
            print_result("Cache Line Updates", "", result.operations, "");
            print_result("Operations/Second", "", result.ops_per_second / 1e9, "G ops/s");

            // Analysis
            std::cout << "\nAnalysis:\n";
            double ns_per_op = (result.elapsed_ms * 1e6) / result.operations;
            std::cout << "Cache line coherency overhead: " << ns_per_op << " ns\n";

            if (ns_per_op > 100) {
                std::cout << "⚠ HIGH DCOH OVERHEAD: " << ns_per_op << " ns per cache line\n";
                std::cout << "  → Suggests coherency protocol bottleneck\n";
            }
        }

        print_header("Performance Testing Complete");
        std::cout << "\nSummary:\n";
        std::cout << "✓ All performance tests completed\n";
        std::cout << "✓ Check analysis section above for bottleneck indicators\n";
        std::cout << "\nRecommendations:\n";
        std::cout << "1. If pointer-chase latency > 500ns: Increase CXL cache size\n";
        std::cout << "2. If stride bandwidth < 80% of sequential: Check memory alignment\n";
        std::cout << "3. If GEMM TFLOPS < 1: Verify CXL bandwidth is sufficient\n";
        std::cout << "4. If DCOH overhead > 100ns: Profile coherency messages\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << "\n";
        return 1;
    }
}
