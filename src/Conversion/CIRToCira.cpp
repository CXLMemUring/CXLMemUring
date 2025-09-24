//===- CIRToCira.cpp - ClangIR to Cira conversion -----------------===//
//
// This file implements the conversion from ClangIR dialect to Cira dialect
// for offloading C/C++ operations to remote memory systems.
//
//===----------------------------------------------------------------------===//

#include "Conversion/CIRToCira.h"
#include "Dialect/CiraOps.h"
#include "Dialect/RemoteMem.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

// Enable direct matching on CIR operations
#include <clang/CIR/Dialect/IR/CIRDialect.h>

using namespace mlir;
using namespace mlir::cira;

namespace {

//===----------------------------------------------------------------------===//
// ClangIR For Loop to Cira Offload Pattern
//===----------------------------------------------------------------------===//

/// Convert ClangIR for loops with graph patterns to Cira offload operations
struct CIRForLoopToCiraPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    Location loc = forOp.getLoc();

    // Analyze the loop for graph processing patterns
    if (!isGraphProcessingLoop(forOp)) {
      return failure();
    }

    // Estimate the loop workload
    auto tripCount = estimateTripCount(forOp);
    if (!tripCount || *tripCount < 10000) {
      return failure(); // Only offload large loops
    }

    // Create offload operation for the loop
    auto offloadOp = rewriter.create<OffloadOp>(
        loc, TypeRange{}, // No results for now
        rewriter.getStringAttr("graph_traversal"),
        forOp.getInitArgs()
    );

    // Build the offload body with optimized remote memory access
    Block *offloadBody = rewriter.createBlock(&offloadOp.getBody());
    rewriter.setInsertionPointToStart(offloadBody);

    // Clone the loop body with remote memory optimizations
    IRMapping mapping;
    for (auto &op : forOp.getBody()->getOperations()) {
      if (!isa<scf::YieldOp>(op)) {
        rewriter.clone(op, mapping);
      }
    }

    rewriter.replaceOp(forOp, offloadOp.getResults());
    return success();
  }

private:
  /// Check if a for loop exhibits graph processing patterns
  bool isGraphProcessingLoop(scf::ForOp forOp) const {
    // Look for patterns like:
    // for (i = 0; i < n; i++) {
    //   process_edges(graph[i]);
    //   access_neighbors(graph[i]);
    // }

    bool hasIndirectAccess = false;
    bool hasLargeMemoryFootprint = false;

    forOp.getBody()->walk([&](Operation *op) {
      // Check for load operations (potential graph access)
      if (isa<memref::LoadOp>(op)) {
        hasIndirectAccess = true;
      }

      // Check for nested loops (common in graph algorithms)
      if (isa<scf::ForOp, scf::WhileOp>(op)) {
        hasLargeMemoryFootprint = true;
      }
    });

    return hasIndirectAccess && hasLargeMemoryFootprint;
  }

  /// Estimate the trip count of a loop
  std::optional<int64_t> estimateTripCount(scf::ForOp forOp) const {
    // Simple heuristic: check if bounds are constants
    auto lowerBound = forOp.getLowerBound().getDefiningOp<arith::ConstantOp>();
    auto upperBound = forOp.getUpperBound().getDefiningOp<arith::ConstantOp>();
    auto step = forOp.getStep().getDefiningOp<arith::ConstantOp>();

    if (lowerBound && upperBound && step) {
      auto lbValue = llvm::cast<IntegerAttr>(lowerBound.getValue()).getInt();
      auto ubValue = llvm::cast<IntegerAttr>(upperBound.getValue()).getInt();
      auto stepValue = llvm::cast<IntegerAttr>(step.getValue()).getInt();

      if (stepValue > 0) {
        return (ubValue - lbValue) / stepValue;
      }
    }

    return std::nullopt;
  }
};

//===----------------------------------------------------------------------===//
// ClangIR While Loop to Cira Pattern (Pointer Chasing)
//===----------------------------------------------------------------------===//

/// Convert pointer-chasing while loops to Cira offload operations
struct CIRWhileLoopToCiraPattern : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern<scf::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
    Location loc = whileOp.getLoc();

    // Check if this is a pointer-chasing pattern
    if (!isPointerChasingLoop(whileOp)) {
      return failure();
    }

    // Create offload operation for pointer chasing
    auto offloadOp = rewriter.create<OffloadOp>(
        loc, whileOp.getResultTypes(),
        rewriter.getStringAttr("pointer_chase"),
        whileOp.getInits()
    );

    // Build optimized pointer chasing with prefetching
    Block *offloadBody = rewriter.createBlock(&offloadOp.getBody());
    rewriter.setInsertionPointToStart(offloadBody);

    // Placeholder for optimized pointer chasing implementation
    // Would include:
    // - Prefetching next pointers
    // - Batching pointer accesses
    // - Remote memory streaming

    rewriter.replaceOp(whileOp, offloadOp.getResults());
    return success();
  }

private:
  /// Check if a while loop is doing pointer chasing
  bool isPointerChasingLoop(scf::WhileOp whileOp) const {
    // Look for patterns like:
    // while (ptr != null) {
    //   process(ptr->data);
    //   ptr = ptr->next;
    // }

    bool hasPointerLoad = false;
    bool hasPointerUpdate = false;

    // Check the before region for pointer comparison
    if (!whileOp.getBefore().empty()) {
      whileOp.getBefore().walk([&](Operation *op) {
        if (isa<memref::LoadOp>(op)) {
          hasPointerLoad = true;
        }
      });
    }

    // Check the after region for pointer updates
    if (!whileOp.getAfter().empty()) {
      whileOp.getAfter().walk([&](Operation *op) {
        if (isa<memref::LoadOp, memref::StoreOp>(op)) {
          hasPointerUpdate = true;
        }
      });
    }

    return hasPointerLoad && hasPointerUpdate;
  }
};

//===----------------------------------------------------------------------===//
// Direct CIR -> Cira fallback patterns (when SCF lowering isn't available)
//===----------------------------------------------------------------------===//
struct CIRForLoopOpToCiraFallback : public OpRewritePattern<cir::ForOp> {
  using OpRewritePattern<cir::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(cir::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    auto off = rewriter.create<OffloadOp>(
        forOp.getLoc(), TypeRange{}, rewriter.getStringAttr("graph_traversal"),
        ValueRange{});
    (void)rewriter.createBlock(&off.getBody());
    rewriter.replaceOp(forOp, off.getResults());
    return success();
  }
};

struct CIRWhileOpToCiraFallback : public OpRewritePattern<cir::WhileOp> {
  using OpRewritePattern<cir::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(cir::WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
    auto off = rewriter.create<OffloadOp>(
        whileOp.getLoc(), TypeRange{},
        rewriter.getStringAttr("pointer_chase"), ValueRange{});
    (void)rewriter.createBlock(&off.getBody());
    rewriter.replaceOp(whileOp, off.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ClangIR to Cira Conversion Pass
//===----------------------------------------------------------------------===//

struct CIRToCiraPass : public PassWrapper<CIRToCiraPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CIRToCiraPass)

  StringRef getArgument() const override { return "cir-to-cira"; }

  StringRef getDescription() const override {
    return "Convert ClangIR operations to Cira offload operations for graph processing";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<cira::RemoteMemDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    populateCIRToCiraPatterns(context, patterns);

    ConversionTarget target(*context);
    target.addLegalDialect<cira::RemoteMemDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();

    // Mark large loops as illegal to force conversion
    target.addDynamicallyLegalOp<scf::ForOp>([](scf::ForOp op) {
      // Keep small loops as legal
      return true; // Simplified for now
    });

    target.addDynamicallyLegalOp<scf::WhileOp>([](scf::WhileOp op) {
      // Keep non-pointer-chasing loops as legal
      return true; // Simplified for now
    });

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public API Implementation
//===----------------------------------------------------------------------===//

void mlir::cira::populateCIRToCiraPatterns(MLIRContext *ctx, RewritePatternSet &patterns) {
  patterns.add<CIRForLoopToCiraPattern, CIRWhileLoopToCiraPattern>(ctx);
  // Also register CIR fallback patterns to make progress on raw CIR inputs.
  patterns.add<CIRForLoopOpToCiraFallback, CIRWhileOpToCiraFallback>(ctx);
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::cira::createCIRToCiraPass() {
  return std::make_unique<CIRToCiraPass>();
}

//===----------------------------------------------------------------------===//
// ClangIR Loop Analysis Implementation
//===----------------------------------------------------------------------===//

CIRLoopAnalysis CIRLoopAnalysis::analyze(func::FuncOp func) {
  CIRLoopAnalysis analysis;

  func.walk([&](Operation *op) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      // Analyze for loop characteristics
      bool isGraphTraversal = false;
      bool isMemoryIntensive = false;
      bool hasRegularAccess = true;

      forOp.getBody()->walk([&](Operation *innerOp) {
        if (isa<memref::LoadOp, memref::StoreOp>(innerOp)) {
          isMemoryIntensive = true;

          // Check for indirect access patterns (graph-like)
          for (auto operand : innerOp->getOperands()) {
            if (auto loadOp = operand.getDefiningOp<memref::LoadOp>()) {
              isGraphTraversal = true;
              hasRegularAccess = false;
            }
          }
        }
      });

      if (isGraphTraversal) {
        analysis.graphTraversalLoops.push_back(op);
      }
      if (isMemoryIntensive) {
        analysis.memoryIntensiveLoops.push_back(op);
      }
      if (hasRegularAccess) {
        analysis.regularAccessLoops.push_back(op);
      }
    }
    else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
      // While loops are often graph traversal (pointer chasing)
      analysis.graphTraversalLoops.push_back(op);
    }
  });

  return analysis;
}

bool CIRLoopAnalysis::shouldOffloadLoop(Operation *loop) const {
  return llvm::find(graphTraversalLoops, loop) != graphTraversalLoops.end() ||
         llvm::find(memoryIntensiveLoops, loop) != memoryIntensiveLoops.end();
}

//===----------------------------------------------------------------------===//
// Memory Pattern Detection Implementation
//===----------------------------------------------------------------------===//

CMemoryPattern mlir::cira::detectMemoryPattern(Operation *op) {
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    // Check for nested loops (matrix operations)
    bool hasNestedLoop = false;
    int loopDepth = 0;

    forOp.getBody()->walk([&](scf::ForOp nestedFor) {
      hasNestedLoop = true;
      loopDepth++;
    });

    if (loopDepth >= 3) {
      return CMemoryPattern::MATRIX_MULTIPLY;
    } else if (hasNestedLoop) {
      return CMemoryPattern::NESTED_LOOP;
    } else {
      return CMemoryPattern::ARRAY_SCAN;
    }
  }
  else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    return CMemoryPattern::POINTER_CHASE;
  }

  return CMemoryPattern::STREAMING;
}

double mlir::cira::estimateCOffloadBenefit(Operation *op, CMemoryPattern pattern) {
  // Simple heuristic based on pattern type
  switch (pattern) {
    case CMemoryPattern::MATRIX_MULTIPLY:
      return 10.0; // High benefit for matrix operations
    case CMemoryPattern::POINTER_CHASE:
      return 8.0;  // High benefit for irregular access
    case CMemoryPattern::NESTED_LOOP:
      return 6.0;  // Medium benefit for nested operations
    case CMemoryPattern::ARRAY_SCAN:
      return 4.0;  // Medium benefit for regular access
    case CMemoryPattern::REDUCTION:
      return 5.0;  // Medium-high benefit for reductions
    case CMemoryPattern::STREAMING:
      return 3.0;  // Lower benefit for streaming
  }
  return 1.0; // Default minimal benefit
}
