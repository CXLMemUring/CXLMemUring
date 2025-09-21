//===- CIRToCira.h - ClangIR to Cira conversion --------------------===//
//
// This file declares the conversion from ClangIR dialect to Cira dialect
// for offloading C/C++ operations to remote memory systems.
//
//===----------------------------------------------------------------------===//

#ifndef CONVERSION_CIRTOCIRA_H
#define CONVERSION_CIRTOCIRA_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
class MLIRContext;
class RewritePatternSet;

namespace cira {

//===----------------------------------------------------------------------===//
// ClangIR to Cira Conversion Pass
//===----------------------------------------------------------------------===//

/// Create a pass to convert ClangIR operations to Cira offload operations
std::unique_ptr<OperationPass<ModuleOp>> createCIRToCiraPass();

/// Populate ClangIR to Cira conversion patterns
void populateCIRToCiraPatterns(MLIRContext *ctx, RewritePatternSet &patterns);

//===----------------------------------------------------------------------===//
// ClangIR Loop Analysis for Graph Processing
//===----------------------------------------------------------------------===//

/// Analyze ClangIR for/while loops for graph processing patterns
struct CIRLoopAnalysis {
  // Nested loops suitable for graph traversal
  SmallVector<Operation*> graphTraversalLoops;

  // Memory-intensive loops
  SmallVector<Operation*> memoryIntensiveLoops;

  // Loops with regular access patterns
  SmallVector<Operation*> regularAccessLoops;

  /// Analyze a function for loop-based offloading opportunities
  static CIRLoopAnalysis analyze(func::FuncOp func);

  /// Check if a loop should be offloaded to remote memory
  bool shouldOffloadLoop(Operation *loop) const;
};

//===----------------------------------------------------------------------===//
// C/C++ Memory Access Pattern Detection
//===----------------------------------------------------------------------===//

enum class CMemoryPattern {
  ARRAY_SCAN,        // for(i=0; i<n; i++) a[i] = ...
  NESTED_LOOP,       // for(i=0; i<m; i++) for(j=0; j<n; j++) ...
  POINTER_CHASE,     // while(ptr) ptr = ptr->next
  MATRIX_MULTIPLY,   // Triple nested loop with a[i][k] * b[k][j]
  REDUCTION,         // Accumulation pattern
  STREAMING          // Sequential access with large data
};

/// Detect memory access patterns in ClangIR operations
CMemoryPattern detectMemoryPattern(Operation *op);

/// Estimate the benefit of offloading a C/C++ construct
double estimateCOffloadBenefit(Operation *op, CMemoryPattern pattern);

} // namespace cira
} // namespace mlir

#endif // CONVERSION_CIRTOCIRA_H