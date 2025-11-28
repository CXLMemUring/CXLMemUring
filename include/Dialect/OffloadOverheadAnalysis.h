//
// OffloadOverheadAnalysis.h - Enhanced overhead analysis with dominator tree and liveness
//
// Uses dominator tree analysis to find optimal H2D/D2H placement and
// liveness analysis to minimize data transfer overhead.
//

#ifndef CIRA_OFFLOAD_OVERHEAD_ANALYSIS_H
#define CIRA_OFFLOAD_OVERHEAD_ANALYSIS_H

#include "Dialect/OffloadProfile.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"

#include <map>

namespace mlir {
namespace cira {

//===----------------------------------------------------------------------===//
// Value Liveness for Offload Regions
//===----------------------------------------------------------------------===//

// Represents a value that needs to be transferred
struct TransferValue {
    Value value;
    Type type;
    uint64_t sizeBytes;
    bool isInput;   // True for H2D, false for D2H
    bool isOutput;  // True if value is used after offload region

    // Dominator info
    Operation *definingOp;
    Operation *lastUseOp;
};

// Liveness state for offload analysis
struct OffloadLivenessState {
    llvm::DenseSet<Value> liveIn;   // Values live at entry (need H2D)
    llvm::DenseSet<Value> liveOut;  // Values live at exit (need D2H)
    llvm::DenseSet<Value> defined;  // Values defined in region
    llvm::DenseSet<Value> used;     // Values used in region
};

//===----------------------------------------------------------------------===//
// Enhanced Overhead Analysis
//===----------------------------------------------------------------------===//

class OffloadOverheadAnalysis {
public:
    OffloadOverheadAnalysis(const OffloadTimingProfile &profile);

    // Analyze a function for offload overhead
    void analyzeFunction(func::FuncOp funcOp);

    // Get transfer values for an offload region
    void computeTransferValues(Operation *regionOp,
                               SmallVectorImpl<TransferValue> &h2dValues,
                               SmallVectorImpl<TransferValue> &d2hValues);

    // Compute overhead using dominator tree optimization
    struct OverheadResult {
        uint64_t h2dBytes;
        uint64_t d2hBytes;
        uint64_t h2dLatencyNs;
        uint64_t d2hLatencyNs;
        uint64_t kernelCycles;
        uint64_t totalOverheadNs;
        double expectedSpeedup;
        bool shouldOffload;
        std::string reason;
    };

    OverheadResult computeOverhead(Operation *regionOp,
                                   uint64_t estimatedElements);

    // Get optimal H2D/D2H placement using dominator tree
    struct TransferPlacement {
        Operation *h2dInsertPoint;  // Where to insert H2D transfer
        Operation *d2hInsertPoint;  // Where to insert D2H transfer
        bool canHoistH2D;           // Can move H2D out of loop
        bool canSinkD2H;            // Can move D2H out of loop
    };

    TransferPlacement getOptimalPlacement(Operation *regionOp);

private:
    const OffloadTimingProfile &profile_;
    std::unique_ptr<DominanceInfo> domInfo_;
    std::unique_ptr<PostDominanceInfo> postDomInfo_;

    // Compute liveness for a region
    OffloadLivenessState computeLiveness(Operation *regionOp);

    // Estimate size of a value in bytes
    uint64_t estimateValueSize(Value value);

    // Check if value is used after a given operation
    bool isUsedAfter(Value value, Operation *op);

    // Check if value is defined before a given operation
    bool isDefinedBefore(Value value, Operation *op);

    // Find the immediate dominator of an operation
    Operation *getImmediateDominator(Operation *op);
};

//===----------------------------------------------------------------------===//
// Enhanced Profile-Guided Offload Pass with Overhead Analysis
//===----------------------------------------------------------------------===//

struct EnhancedOffloadAnalysisPass
    : public PassWrapper<EnhancedOffloadAnalysisPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EnhancedOffloadAnalysisPass)

    EnhancedOffloadAnalysisPass() = default;
    EnhancedOffloadAnalysisPass(const EnhancedOffloadAnalysisPass &pass) {}

    void runOnOperation() override;

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<cira::RemoteMemDialect>();
    }

    StringRef getArgument() const final { return "enhanced-offload-analysis"; }
    StringRef getDescription() const final {
        return "Enhanced offload analysis with dominator tree and liveness";
    }

    // Command-line options
    Option<std::string> profilePathOption{
        *this, "offload-profile",
        llvm::cl::desc("Path to offload timing profile JSON file")};

    Option<std::string> experimentResultsOption{
        *this, "experiment-results",
        llvm::cl::desc("Path to experiment results JSON file")};

    Option<double> speedupThresholdOption{
        *this, "speedup-threshold",
        llvm::cl::desc("Minimum expected speedup to offload"),
        llvm::cl::init(1.2)};

    Option<bool> verboseOption{
        *this, "verbose",
        llvm::cl::desc("Print detailed analysis results"),
        llvm::cl::init(false)};

private:
    OffloadTimingProfile profile_;

    void analyzeAndAnnotate(func::FuncOp funcOp,
                           OffloadOverheadAnalysis &analysis);
};

std::unique_ptr<Pass> createEnhancedOffloadAnalysisPass();

//===----------------------------------------------------------------------===//
// Profile Export for Compiler Integration
//===----------------------------------------------------------------------===//

// Export experiment results in compiler-compatible format
LogicalResult exportProfileForCompiler(
    StringRef experimentResultsPath,
    StringRef outputPath,
    const OffloadOverheadAnalysis::OverheadResult &analysisResult);

// Load experiment results and convert to OffloadTimingProfile
LogicalResult loadExperimentResults(
    StringRef experimentResultsPath,
    OffloadTimingProfile &profile);

} // namespace cira
} // namespace mlir

#endif // CIRA_OFFLOAD_OVERHEAD_ANALYSIS_H
