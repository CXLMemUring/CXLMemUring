//
// OffloadProfile.h - Profile-guided offload decision support
// Reads timing profiles from JSON and provides hints to compiler passes
//

#ifndef CIRA_OFFLOADPROFILE_H
#define CIRA_OFFLOADPROFILE_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "Dialect/RemoteMem.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

#include <map>
#include <string>
#include <vector>

namespace mlir {
namespace cira {

// Timing data for a single function offload profile
struct FunctionOffloadProfile {
    std::string name;
    uint64_t calls;
    uint64_t total_ns;
    double avg_ns;
    uint64_t elements_processed;
    double throughput;
    bool offload_candidate;
    std::string parallelism_type; // "embarrassingly_parallel", "reduction", "data_dependent", "tree_traversal"
};

// Overall offload timing profile
struct OffloadTimingProfile {
    std::string profile_type;
    std::string target;

    // Per-function profiles
    std::map<std::string, FunctionOffloadProfile> functions;

    // Overall execution stats
    uint64_t total_simplex_iterations;
    uint64_t total_execution_ns;

    // Offload hints from profiler
    std::string primary_offload_target;
    std::string secondary_offload_target;
    double expected_speedup;
    uint64_t min_elements_for_offload;
    double data_transfer_cost_factor;

    // Transfer timing (from Vortex profiling)
    uint64_t h2d_latency_ns;
    uint64_t kernel_latency_ns;
    uint64_t d2h_latency_ns;
    double h2d_bandwidth_gbps;
    double d2h_bandwidth_gbps;
    uint64_t optimal_prefetch_distance;
};

// Profile-guided offload decision
enum class OffloadDecision {
    CPU_ONLY,           // Keep on CPU
    GPU_ALWAYS,         // Always offload to GPU
    GPU_CONDITIONAL,    // Offload based on data size
    HYBRID              // Split between CPU and GPU
};

struct OffloadStrategy {
    OffloadDecision decision;
    uint64_t min_elements;      // Minimum elements for GPU offload
    double expected_speedup;
    std::string target_device;  // "vortex", "cpu", "hetero"
};

// Load offload timing profile from JSON file
LogicalResult loadOffloadProfile(StringRef filename, OffloadTimingProfile &profile);

// Compute offload strategy based on profile data
OffloadStrategy computeOffloadStrategy(
    const OffloadTimingProfile &profile,
    StringRef functionName,
    uint64_t estimatedElements,
    uint64_t dataTransferBytes);

// Check if a function should be offloaded based on profile
bool shouldOffloadFunction(
    const OffloadTimingProfile &profile,
    StringRef functionName,
    uint64_t estimatedElements);

// Get expected speedup for offloading a function
double getExpectedSpeedup(
    const OffloadTimingProfile &profile,
    StringRef functionName);

//===----------------------------------------------------------------------===//
// Profile-Guided Offload Annotation Pass
//===----------------------------------------------------------------------===//

struct ProfileGuidedOffloadPass
    : public PassWrapper<ProfileGuidedOffloadPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ProfileGuidedOffloadPass)

    ProfileGuidedOffloadPass() = default;
    ProfileGuidedOffloadPass(const ProfileGuidedOffloadPass &pass) {}

    void runOnOperation() override;

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<cira::RemoteMemDialect>();
    }

    StringRef getArgument() const final { return "profile-guided-offload"; }
    StringRef getDescription() const final {
        return "Apply profile-guided offload decisions to functions";
    }

    // Command-line options
    Option<std::string> profilePathOption{
        *this, "offload-profile",
        llvm::cl::desc("Path to offload timing profile JSON file")};

    Option<std::string> targetOption{
        *this, "offload-target",
        llvm::cl::desc("Target device for offloading (vortex, cpu, auto)"),
        llvm::cl::init("auto")};

    Option<uint64_t> minElementsOption{
        *this, "min-offload-elements",
        llvm::cl::desc("Minimum elements to trigger offloading"),
        llvm::cl::init(1000)};

    Option<double> speedupThresholdOption{
        *this, "speedup-threshold",
        llvm::cl::desc("Minimum expected speedup to offload"),
        llvm::cl::init(1.5)};

    Option<bool> forceOffloadOption{
        *this, "force-offload",
        llvm::cl::desc("Force offload regardless of profile"),
        llvm::cl::init(false)};

private:
    OffloadTimingProfile profile;

    void annotateFunction(func::FuncOp funcOp, const OffloadStrategy &strategy);
    void annotateLoop(Operation *loopOp, const OffloadStrategy &strategy);
    uint64_t estimateLoopTripCount(Operation *loopOp);
    uint64_t estimateDataTransfer(func::FuncOp funcOp);
};

std::unique_ptr<Pass> createProfileGuidedOffloadPass();

inline void registerProfileGuidedOffloadPass() {
    PassManager::registerPass([]() -> std::unique_ptr<Pass> {
        return createProfileGuidedOffloadPass();
    });
}

//===----------------------------------------------------------------------===//
// Offload Cost Model
//===----------------------------------------------------------------------===//

// Cost model for offload decisions
class OffloadCostModel {
public:
    OffloadCostModel(const OffloadTimingProfile &profile);

    // Estimate execution time on CPU
    double estimateCPUTime(StringRef funcName, uint64_t elements) const;

    // Estimate execution time on GPU (including transfers)
    double estimateGPUTime(StringRef funcName, uint64_t elements,
                          uint64_t h2dBytes, uint64_t d2hBytes) const;

    // Get recommended offload decision
    OffloadDecision getRecommendation(StringRef funcName, uint64_t elements,
                                     uint64_t h2dBytes, uint64_t d2hBytes) const;

    // Get crossover point (elements where GPU becomes faster)
    uint64_t getCrossoverPoint(StringRef funcName,
                              uint64_t bytesPerElement) const;

private:
    const OffloadTimingProfile &profile_;

    // Default constants if profile doesn't have data
    static constexpr double kDefaultH2DBandwidth = 10.0;  // GB/s
    static constexpr double kDefaultD2HBandwidth = 10.0;  // GB/s
    static constexpr double kDefaultKernelOverhead = 10000.0;  // ns
    static constexpr double kDefaultGPUSpeedup = 10.0;  // x faster per element
};

} // namespace cira
} // namespace mlir

#endif // CIRA_OFFLOADPROFILE_H
