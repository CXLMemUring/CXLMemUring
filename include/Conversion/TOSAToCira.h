//===- TOSAToCira.h - TOSA to Cira conversion ----------------------===//
//
// This file declares the conversion from TOSA dialect to Cira dialect
// for offloading tensor operations to remote memory systems.
//
//===----------------------------------------------------------------------===//

#ifndef CONVERSION_TOSATOCIRA_H
#define CONVERSION_TOSATOCIRA_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class MLIRContext;
class RewritePatternSet;
class TypeConverter;

namespace cira {

//===----------------------------------------------------------------------===//
// TOSA to Cira Conversion Pass
//===----------------------------------------------------------------------===//

/// Create a pass to convert TOSA operations to Cira offload operations
std::unique_ptr<OperationPass<ModuleOp>> createTOSAToCiraPass();

/// Populate TOSA to Cira conversion patterns
void populateTOSAToCiraPatterns(MLIRContext *ctx, RewritePatternSet &patterns);

//===----------------------------------------------------------------------===//
// TOSA Offloading Analysis
//===----------------------------------------------------------------------===//

/// Analyze TOSA operations for remote memory access patterns
struct TOSAOffloadAnalysis {
  // Operations that benefit from remote memory access
  SmallVector<Operation*> remoteAccessOps;

  // Memory-intensive operations
  SmallVector<Operation*> memoryIntensiveOps;

  // Graph-like operations (matmul chains, convolutions)
  SmallVector<Operation*> graphOps;

  /// Analyze a function for TOSA offloading opportunities
  static TOSAOffloadAnalysis analyze(func::FuncOp func);

  /// Check if an operation should be offloaded
  bool shouldOffload(Operation *op) const;
};

//===----------------------------------------------------------------------===//
// TOSA Memory Tier Selection
//===----------------------------------------------------------------------===//

enum class TensorMemoryTier {
  LOCAL_CACHE,    // Hot data, frequently accessed
  LOCAL_DRAM,     // Warm data, moderately accessed
  CXL_ATTACHED,   // Large tensors, sequential access
  CXL_POOLED,     // Shared tensors across processes
  FAR_MEMORY      // Cold data, rarely accessed
};

/// Select appropriate memory tier for a tensor based on access patterns
TensorMemoryTier selectMemoryTier(Value tensor, const TOSAOffloadAnalysis &analysis);

} // namespace cira
} // namespace mlir

#endif // CONVERSION_TOSATOCIRA_H