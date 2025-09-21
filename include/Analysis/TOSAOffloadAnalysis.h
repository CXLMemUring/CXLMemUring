//===- TOSAOffloadAnalysis.h - TOSA Offload Analysis ---------------===//
//
// This file declares analysis passes for identifying TOSA operations
// that would benefit from remote memory offloading.
//
//===----------------------------------------------------------------------===//

#ifndef ANALYSIS_TOSAOFFLOADANALYSIS_H
#define ANALYSIS_TOSAOFFLOADANALYSIS_H

#include "mlir/Analysis/CallGraph.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/AnalysisManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace cira {

//===----------------------------------------------------------------------===//
// Memory Access Pattern Classification
//===----------------------------------------------------------------------===//

enum class MemoryAccessPattern {
  SEQUENTIAL,     // Linear access through memory
  STRIDED,        // Regular stride access
  RANDOM,         // Irregular access pattern
  BLOCK_SPARSE,   // Block-wise sparse access
  GATHER_SCATTER  // Indirect memory access
};

struct AccessInfo {
  MemoryAccessPattern pattern;
  int64_t accessCount;      // Number of memory accesses
  int64_t dataSize;         // Size of data accessed in bytes
  double intensity;         // Compute to memory access ratio
  bool isReusable;          // Whether data can be reused
};

//===----------------------------------------------------------------------===//
// Tensor Operation Profiling
//===----------------------------------------------------------------------===//

struct TensorOpProfile {
  Operation *op;
  int64_t flopCount;                    // Floating point operations
  int64_t memoryFootprint;              // Total memory required
  SmallVector<AccessInfo> inputAccess;  // Input access patterns
  SmallVector<AccessInfo> outputAccess; // Output access patterns
  double computeIntensity;              // FLOPs per byte ratio
  bool hasDataReuse;                    // Cross-operation data reuse
  int64_t criticalPathLength;           // Operations in critical path
};

//===----------------------------------------------------------------------===//
// Graph-Level Analysis
//===----------------------------------------------------------------------===//

class TOSAGraphAnalysis {
private:
  SmallVector<TensorOpProfile> opProfiles_;
  DenseMap<Value, SmallVector<Operation*>> dataFlowGraph_;
  DenseMap<Operation*, int64_t> criticalPath_;

public:
  /// Analyze a function for TOSA operation characteristics
  void analyzeFunction(func::FuncOp func);

  /// Get the profile for a specific operation
  const TensorOpProfile* getProfile(Operation *op) const;

  /// Find operations that form computation clusters
  SmallVector<SmallVector<Operation*>> findComputeClusters() const;

  /// Identify memory-bound vs compute-bound operations
  SmallVector<Operation*> getMemoryBoundOps() const;
  SmallVector<Operation*> getComputeBoundOps() const;

  /// Find operations suitable for different memory tiers
  SmallVector<Operation*> getCandidatesForTier(int memoryTier) const;

  /// Estimate the benefit of offloading a set of operations
  double estimateOffloadBenefit(ArrayRef<Operation*> ops) const;

private:
  /// Profile individual TOSA operations
  TensorOpProfile profileMatMul(tosa::MatMulOp op);
  TensorOpProfile profileConv2D(tosa::Conv2DOp op);
  TensorOpProfile profileReduce(tosa::ReduceSumOp op);
  TensorOpProfile profileElementwise(Operation *op);

  /// Analyze data flow between operations
  void buildDataFlowGraph(func::FuncOp func);

  /// Compute critical path through the computation graph
  void computeCriticalPath();

  /// Estimate memory access patterns
  AccessInfo analyzeMemoryAccess(Value tensor, Operation *consumer);
};

//===----------------------------------------------------------------------===//
// Offloading Cost Model
//===----------------------------------------------------------------------===//

struct OffloadCostModel {
  // Hardware characteristics
  struct {
    double localBandwidth;     // Local memory bandwidth (GB/s)
    double cxlBandwidth;       // CXL memory bandwidth (GB/s)
    double farBandwidth;       // Far memory bandwidth (GB/s)
    double localLatency;       // Local memory latency (ns)
    double cxlLatency;         // CXL memory latency (ns)
    double farLatency;         // Far memory latency (ns)
  } hardware;

  // Cost estimation functions
  double estimateLocalCost(const TensorOpProfile &profile) const;
  double estimateRemoteCost(const TensorOpProfile &profile, int memoryTier) const;
  double estimateOffloadOverhead(ArrayRef<Operation*> ops) const;

  /// Get the optimal memory tier for an operation
  int selectOptimalTier(const TensorOpProfile &profile) const;

  /// Calculate the speedup from offloading
  double calculateSpeedup(ArrayRef<Operation*> ops, int targetTier) const;
};

//===----------------------------------------------------------------------===//
// Pattern-Based Offload Recommendations
//===----------------------------------------------------------------------===//

struct OffloadRecommendation {
  Operation *op;
  int recommendedTier;      // Target memory tier
  double expectedSpeedup;   // Predicted performance improvement
  SmallVector<Operation*> dependentOps; // Operations that should move together
  StringRef rationale;      // Human-readable explanation
};

class TOSAOffloadRecommender {
private:
  TOSAGraphAnalysis analysis_;
  OffloadCostModel costModel_;

public:
  /// Generate offloading recommendations for a function
  SmallVector<OffloadRecommendation> recommendOffloads(func::FuncOp func);

  /// Validate recommendations against constraints
  LogicalResult validateRecommendations(ArrayRef<OffloadRecommendation> recs);

  /// Apply recommendations to transform the IR
  LogicalResult applyRecommendations(func::FuncOp func,
                                   ArrayRef<OffloadRecommendation> recs);

private:
  /// Find optimal groupings of operations for offloading
  SmallVector<SmallVector<Operation*>> findOffloadGroups();

  /// Check if operations can be safely offloaded together
  bool canOffloadTogether(ArrayRef<Operation*> ops);

  /// Generate explanation for offload decisions
  StringRef explainRecommendation(const OffloadRecommendation &rec);
};

//===----------------------------------------------------------------------===//
// Analysis Pass Integration
//===----------------------------------------------------------------------===//

/// Analysis pass that identifies TOSA offloading opportunities
class TOSAOffloadAnalysisPass : public AnalysisInfoMixin<TOSAOffloadAnalysisPass> {
  friend AnalysisInfoMixin<TOSAOffloadAnalysisPass>;
  static AnalysisKey key;

public:
  using Result = SmallVector<OffloadRecommendation>;

  TOSAOffloadAnalysisPass(Operation *op, AnalysisManager &am);

  /// Get the analysis results
  const Result& getRecommendations() const { return recommendations_; }

private:
  Result recommendations_;
};

} // namespace cira
} // namespace mlir

#endif // ANALYSIS_TOSAOFFLOADANALYSIS_H