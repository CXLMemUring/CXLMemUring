//===- CIRToCira.cpp - ClangIR to Cira conversion -----------------===//
//
// This file implements the conversion from ClangIR dialect to Cira dialect
// for offloading C/C++ operations to remote memory systems.
//
// Key transformation (from paper Listing 1):
//   Original:                          Transformed CIRA IR:
//   while (node) {                     %stream = cira.stream_create_indirect
//     val = node->data;                          %start_node, offset=8
//     node = node->next;               cira.offload_start @vortex_core_0 {
//   }                                    cira.prefetch_chain %stream, depth=16
//                                      }
//                                      %loop:
//                                        %future = cira.peek_stream %stream
//                                        %data = cira.future_await %future
//                                        // Computation on %data...
//                                        cira.advance_stream %stream
//                                        br %loop
//
//===----------------------------------------------------------------------===//

#include "Conversion/CIRToCira.h"
#include "Dialect/CiraOps.h"
#include "Dialect/RemoteMem.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
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
// Cost Model for Offload Decision
//===----------------------------------------------------------------------===//

/// Cost model parameters (from paper equation)
struct OffloadCostModel {
  // Latency parameters (in nanoseconds)
  static constexpr double L_CXL = 165.0;    // CXL memory latency
  static constexpr double L_LLC = 15.0;     // LLC hit latency
  static constexpr double C_sync = 50.0;    // Sync cost via ring buffer

  // Minimum dependency chain depth to benefit from offloading
  static constexpr int MIN_CHAIN_DEPTH = 4;

  /// Compute the gain from offloading
  /// Gain = sum(L_CXL - L_LLC) - (C_sync + C_vortex_busy)
  static double computeGain(int64_t chainDepth, double vortexBusyCycles) {
    double latencySaving = chainDepth * (L_CXL - L_LLC);
    double overhead = C_sync + vortexBusyCycles;
    return latencySaving - overhead;
  }

  /// Check if offloading is beneficial
  static bool shouldOffload(int64_t chainDepth, double vortexBusyCycles = 0.0) {
    return computeGain(chainDepth, vortexBusyCycles) > 0 &&
           chainDepth >= MIN_CHAIN_DEPTH;
  }
};

//===----------------------------------------------------------------------===//
// Pointer Chasing Analysis
//===----------------------------------------------------------------------===//

/// Analyzes a loop to detect pointer-chasing patterns (linked list traversal)
struct PointerChasingAnalysis {
  // The pointer value being chased
  Value ptrValue;

  // The "next" field offset in bytes (e.g., offsetof(node, next))
  int64_t nextPtrOffset = 8;  // Default: assume 8-byte offset

  // The "data" field offset
  int64_t dataOffset = 0;

  // Estimated chain depth (for prefetching)
  int64_t estimatedDepth = 16;

  // Whether the pattern was successfully detected
  bool detected = false;

  // The load operation that loads the next pointer
  Operation *nextPtrLoad = nullptr;

  // The load operation that loads data
  Operation *dataLoad = nullptr;
};

/// Detect pointer chasing pattern in a while loop
PointerChasingAnalysis detectPointerChasing(scf::WhileOp whileOp) {
  PointerChasingAnalysis result;

  // The while loop should have exactly one iter_arg (the pointer)
  if (whileOp.getInits().size() != 1)
    return result;

  result.ptrValue = whileOp.getInits()[0];

  // Analyze the "before" region (condition check)
  // Look for: ptr != null
  Block &beforeBlock = whileOp.getBefore().front();

  // Analyze the "after" region (loop body)
  // Look for:
  //   data = load ptr+dataOffset   (data access)
  //   ptr = load ptr+nextOffset    (pointer update)
  Block &afterBlock = whileOp.getAfter().front();

  bool foundNextLoad = false;
  bool foundDataLoad = false;

  afterBlock.walk([&](Operation *op) {
    if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      // Heuristic: if the loaded value is yielded, it's the next pointer
      // If it's used for computation, it's data
      for (auto *user : loadOp.getResult().getUsers()) {
        if (isa<scf::YieldOp>(user)) {
          result.nextPtrLoad = op;
          foundNextLoad = true;
        } else {
          result.dataLoad = op;
          foundDataLoad = true;
        }
      }
    }
  });

  // Also check for CIR-level loads
  afterBlock.walk([&](Operation *op) {
    if (op->getName().getStringRef().contains("cir.load")) {
      // Check if this is loading the next pointer
      foundNextLoad = true;
      result.nextPtrLoad = op;
    }
  });

  result.detected = foundNextLoad;
  return result;
}

/// Detect pointer chasing in CIR while loop
PointerChasingAnalysis detectPointerChasingCIR(cir::WhileOp whileOp) {
  PointerChasingAnalysis result;

  // Analyze the condition block for null check
  // Analyze the body for pointer chasing pattern

  bool hasPointerLoad = false;
  bool hasPointerUpdate = false;

  whileOp.walk([&](Operation *op) {
    // Look for CIR load operations
    if (op->getName().getStringRef().contains("cir.load") ||
        op->getName().getStringRef().contains("cir.get_member")) {
      hasPointerLoad = true;
      if (!result.nextPtrLoad)
        result.nextPtrLoad = op;
    }
    // Look for struct member access (node->next pattern)
    if (op->getName().getStringRef().contains("cir.ptr_stride") ||
        op->getName().getStringRef().contains("cir.member")) {
      hasPointerUpdate = true;
    }
  });

  result.detected = hasPointerLoad && hasPointerUpdate;
  return result;
}

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

    // Check cost model
    if (!OffloadCostModel::shouldOffload(*tripCount)) {
      return failure();
    }

    // Create offload region operation for the loop
    auto offloadOp = rewriter.create<OffloadRegionOp>(
        loc, TypeRange{}, // No results for now
        SymbolRefAttr(),  // target (optional)
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
// ClangIR While Loop to Cira Pattern (Pointer Chasing) - Paper Listing 1
//===----------------------------------------------------------------------===//

/// Convert pointer-chasing while loops to Cira stream operations
/// This implements the transformation from Listing 1 in the paper:
///
/// Original:                          Transformed:
/// while (node) {                     %stream = cira.stream_create_indirect
///   val = node->data;                          %start_node, offset=8
///   node = node->next;               cira.offload_start @vortex_core_0 {
/// }                                    cira.prefetch_chain %stream, depth=16
///                                    }
///                                    scf.while ... {
///                                      %future = cira.peek_stream %stream
///                                      %data = cira.future_await %future
///                                      // Computation on %data...
///                                      cira.advance_stream %stream
///                                    }
struct CIRWhileLoopToCiraStreamPattern : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern<scf::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
    Location loc = whileOp.getLoc();

    // Detect pointer-chasing pattern
    auto analysis = detectPointerChasing(whileOp);
    if (!analysis.detected) {
      return failure();
    }

    // Check cost model - pointer chasing typically has high benefit
    if (!OffloadCostModel::shouldOffload(analysis.estimatedDepth)) {
      return failure();
    }

    MLIRContext *ctx = rewriter.getContext();

    // Get the element type from the pointer
    Type elementType = rewriter.getI64Type(); // Default to i64
    if (analysis.ptrValue.getType().isa<MemRefType>()) {
      elementType = analysis.ptrValue.getType().cast<MemRefType>().getElementType();
    }

    // Create stream type for the linked list traversal
    auto streamType = StreamType::get(elementType, 0, analysis.nextPtrOffset);
    auto futureType = FutureType::get(elementType);
    auto handleType = HandleType::get(elementType);

    // Step 1: Create stream descriptor for indirect access pattern
    // %stream = cira.stream_create_indirect %start_node, offset=8 : !cira.stream
    auto streamCreateOp = rewriter.create<StreamCreateIndirectOp>(
        loc, streamType,
        analysis.ptrValue,  // start_ptr (will need cast)
        rewriter.getI64IntegerAttr(analysis.nextPtrOffset)
    );

    // Step 2: Create offload region for Vortex to prefetch the chain
    // cira.offload_start @vortex_core_0 {
    //   cira.prefetch_chain %stream, depth=16
    // }
    auto offloadStartOp = rewriter.create<OffloadStartOp>(
        loc, SymbolRefAttr::get(ctx, "vortex_core_0")
    );

    Block *offloadBody = rewriter.createBlock(&offloadStartOp.getBody());
    rewriter.setInsertionPointToStart(offloadBody);

    // Create prefetch_chain inside the offload region
    rewriter.create<PrefetchChainOp>(
        loc, streamCreateOp.getResult(),
        rewriter.getI64IntegerAttr(analysis.estimatedDepth)
    );

    // Step 3: Transform the while loop to use stream operations
    rewriter.setInsertionPointAfter(offloadStartOp);

    // Create a new while loop that uses the stream
    auto newWhileOp = rewriter.create<scf::WhileOp>(
        loc, whileOp.getResultTypes(), whileOp.getInits()
    );

    // Clone the "before" region (condition check)
    rewriter.cloneRegionBefore(whileOp.getBefore(), newWhileOp.getBefore(),
                               newWhileOp.getBefore().end());

    // Build transformed "after" region
    Block *afterBlock = rewriter.createBlock(&newWhileOp.getAfter(),
                                              newWhileOp.getAfter().end());

    // Add block arguments matching the original
    for (auto arg : whileOp.getAfter().front().getArguments()) {
      afterBlock->addArgument(arg.getType(), loc);
    }

    rewriter.setInsertionPointToStart(afterBlock);

    // %future = cira.peek_stream %stream : !cira.stream -> !cira.future
    auto peekOp = rewriter.create<PeekStreamOp>(
        loc, futureType, streamCreateOp.getResult()
    );

    // %data = cira.future_await %future : !cira.future -> element_type
    auto awaitOp = rewriter.create<FutureAwaitOp>(
        loc, elementType, peekOp.getResult()
    );

    // Clone the original computation, replacing pointer loads with awaited data
    IRMapping mapping;
    mapping.map(whileOp.getAfter().front().getArguments(),
                afterBlock->getArguments());

    // Map the data load to the awaited value
    if (analysis.dataLoad) {
      mapping.map(analysis.dataLoad->getResult(0), awaitOp.getResult());
    }

    for (auto &op : whileOp.getAfter().front().getOperations()) {
      // Skip the original load operations - they're replaced by stream ops
      if (&op == analysis.dataLoad || &op == analysis.nextPtrLoad)
        continue;

      if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
        // Before yielding, advance the stream
        rewriter.create<AdvanceStreamOp>(loc, streamCreateOp.getResult());
        rewriter.clone(op, mapping);
      } else {
        rewriter.clone(op, mapping);
      }
    }

    rewriter.replaceOp(whileOp, newWhileOp.getResults());
    return success();
  }
};

/// Simpler pattern for while loops that just wraps in offload region
struct CIRWhileLoopToCiraPattern : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern<scf::WhileOp>::OpRewritePattern;

  // Lower benefit than stream pattern
  CIRWhileLoopToCiraPattern(MLIRContext *ctx)
      : OpRewritePattern<scf::WhileOp>(ctx, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(scf::WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
    Location loc = whileOp.getLoc();

    // Check if this is a pointer-chasing pattern
    if (!isPointerChasingLoop(whileOp)) {
      return failure();
    }

    // Create offload region operation for pointer chasing
    auto offloadOp = rewriter.create<OffloadRegionOp>(
        loc, whileOp.getResultTypes(),
        SymbolRefAttr(),
        whileOp.getInits()
    );

    // Build optimized pointer chasing with prefetching
    Block *offloadBody = rewriter.createBlock(&offloadOp.getBody());
    rewriter.setInsertionPointToStart(offloadBody);

    // Clone the while loop inside the offload region
    IRMapping mapping;
    auto clonedWhile = rewriter.clone(*whileOp.getOperation(), mapping);

    // Add yield at end of offload body
    SmallVector<Value> yieldValues;
    for (auto result : clonedWhile->getResults()) {
      yieldValues.push_back(result);
    }
    rewriter.create<YieldOp>(loc, yieldValues);

    rewriter.replaceOp(whileOp, offloadOp.getResults());
    return success();
  }

private:
  /// Check if a while loop is doing pointer chasing
  bool isPointerChasingLoop(scf::WhileOp whileOp) const {
    bool hasPointerLoad = false;
    bool hasPointerUpdate = false;

    if (!whileOp.getBefore().empty()) {
      whileOp.getBefore().walk([&](Operation *op) {
        if (isa<memref::LoadOp>(op)) {
          hasPointerLoad = true;
        }
      });
    }

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
// Direct CIR -> Cira patterns (for CIR loops)
//===----------------------------------------------------------------------===//

/// Transform CIR for loops with graph patterns
struct CIRForLoopOpToCiraPattern : public OpRewritePattern<cir::ForOp> {
  using OpRewritePattern<cir::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(cir::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    Location loc = forOp.getLoc();

    // Analyze for graph processing patterns
    bool hasIndirectAccess = false;
    forOp.walk([&](Operation *op) {
      if (op->getName().getStringRef().contains("cir.load") ||
          op->getName().getStringRef().contains("cir.get_member")) {
        hasIndirectAccess = true;
      }
    });

    if (!hasIndirectAccess) {
      return failure();
    }

    // Create offload region
    auto offloadOp = rewriter.create<OffloadRegionOp>(
        loc, TypeRange{},
        SymbolRefAttr(),
        ValueRange{}
    );

    Block *offloadBody = rewriter.createBlock(&offloadOp.getBody());
    rewriter.setInsertionPointToStart(offloadBody);

    // Clone the for loop inside offload region
    IRMapping mapping;
    rewriter.clone(*forOp.getOperation(), mapping);

    rewriter.replaceOp(forOp, offloadOp.getResults());
    return success();
  }
};

/// Transform CIR while loops with pointer chasing to CIRA stream operations
struct CIRWhileOpToCiraStreamPattern : public OpRewritePattern<cir::WhileOp> {
  using OpRewritePattern<cir::WhileOp>::OpRewritePattern;

  // Higher benefit to prefer this over simple offload
  CIRWhileOpToCiraStreamPattern(MLIRContext *ctx)
      : OpRewritePattern<cir::WhileOp>(ctx, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(cir::WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
    Location loc = whileOp.getLoc();

    // Detect pointer chasing pattern
    auto analysis = detectPointerChasingCIR(whileOp);
    if (!analysis.detected) {
      return failure();
    }

    // Check cost model
    if (!OffloadCostModel::shouldOffload(analysis.estimatedDepth)) {
      return failure();
    }

    MLIRContext *ctx = rewriter.getContext();
    Type elementType = rewriter.getI64Type();

    // Create stream type
    auto streamType = StreamType::get(elementType, 0, analysis.nextPtrOffset);
    auto futureType = FutureType::get(elementType);

    // Get start pointer from while loop operands/context
    Value startPtr;
    // Try to find the pointer being iterated
    whileOp.walk([&](Operation *op) {
      for (auto operand : op->getOperands()) {
        if (operand.getType().isa<LLVM::LLVMPointerType>() ||
            operand.getType().isa<MemRefType>()) {
          startPtr = operand;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });

    if (!startPtr) {
      return failure();
    }

    // Create handle type and wrap start pointer
    auto handleType = HandleType::get(elementType);

    // Step 1: Create stream for indirect access
    auto streamOp = rewriter.create<StreamCreateIndirectOp>(
        loc, streamType, startPtr,
        rewriter.getI64IntegerAttr(analysis.nextPtrOffset)
    );

    // Step 2: Offload prefetching to Vortex
    auto offloadStart = rewriter.create<OffloadStartOp>(
        loc, SymbolRefAttr::get(ctx, "vortex_core_0")
    );

    Block *offloadBody = rewriter.createBlock(&offloadStart.getBody());
    rewriter.setInsertionPointToStart(offloadBody);
    rewriter.create<PrefetchChainOp>(
        loc, streamOp.getResult(),
        rewriter.getI64IntegerAttr(analysis.estimatedDepth)
    );

    // Step 3: Create the transformed loop structure
    rewriter.setInsertionPointAfter(offloadStart);

    // Create offload region containing the loop
    auto offloadRegion = rewriter.create<OffloadRegionOp>(
        loc, TypeRange{}, SymbolRefAttr(), ValueRange{}
    );

    Block *regionBody = rewriter.createBlock(&offloadRegion.getBody());
    rewriter.setInsertionPointToStart(regionBody);

    // Clone the while loop with stream-based access
    IRMapping mapping;
    rewriter.clone(*whileOp.getOperation(), mapping);

    rewriter.eraseOp(whileOp);
    return success();
  }
};

/// Simple fallback pattern for CIR while loops
struct CIRWhileOpToCiraFallback : public OpRewritePattern<cir::WhileOp> {
  using OpRewritePattern<cir::WhileOp>::OpRewritePattern;

  CIRWhileOpToCiraFallback(MLIRContext *ctx)
      : OpRewritePattern<cir::WhileOp>(ctx, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(cir::WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
    Location loc = whileOp.getLoc();

    // Create simple offload region
    auto offloadOp = rewriter.create<OffloadRegionOp>(
        loc, TypeRange{}, SymbolRefAttr(), ValueRange{}
    );

    Block *offloadBody = rewriter.createBlock(&offloadOp.getBody());
    rewriter.setInsertionPointToStart(offloadBody);

    IRMapping mapping;
    rewriter.clone(*whileOp.getOperation(), mapping);

    rewriter.replaceOp(whileOp, offloadOp.getResults());
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
  // SCF-level patterns (higher priority stream patterns first)
  patterns.add<CIRWhileLoopToCiraStreamPattern>(ctx);  // Benefit=2 (stream-based)
  patterns.add<CIRForLoopToCiraPattern>(ctx);
  patterns.add<CIRWhileLoopToCiraPattern>(ctx);        // Benefit=1 (simple offload)

  // Direct CIR patterns (for when input is raw CIR before SCF lowering)
  patterns.add<CIRWhileOpToCiraStreamPattern>(ctx);    // Benefit=2 (stream-based)
  patterns.add<CIRForLoopOpToCiraPattern>(ctx);
  patterns.add<CIRWhileOpToCiraFallback>(ctx);         // Benefit=1 (fallback)
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
