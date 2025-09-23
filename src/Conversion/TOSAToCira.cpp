//===- TOSAToCira.cpp - TOSA to Cira conversion --------------------===//
//
// This file implements the conversion from TOSA dialect to Cira dialect
// for offloading tensor operations to remote memory systems.
//
//===----------------------------------------------------------------------===//

#include "Conversion/TOSAToCira.h"
#include "Dialect/CiraOps.h"
#include "Dialect/RemoteMem.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::cira;

namespace {

//===----------------------------------------------------------------------===//
// TOSA MatMul to Cira Offload Pattern
//===----------------------------------------------------------------------===//

/// Convert TOSA MatMul operations to Cira offload operations
struct TOSAMatMulToCiraPattern : public OpRewritePattern<tosa::MatMulOp> {
  using OpRewritePattern<tosa::MatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::MatMulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    Location loc = matmulOp.getLoc();

    // Analyze the operation for offloading potential
    auto lhs = matmulOp.getA();
    auto rhs = matmulOp.getB();
    auto result = matmulOp.getResult();

    // Check if this matmul should be offloaded based on size
    auto lhsType = llvm::cast<TensorType>(lhs.getType());
    auto rhsType = llvm::cast<TensorType>(rhs.getType());

    if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape()) {
      return failure(); // Skip dynamic shapes for now
    }

    // Calculate operation intensity (FLOPs)
    int64_t m = lhsType.getShape()[0];
    int64_t k = lhsType.getShape()[1];
    int64_t n = rhsType.getShape()[1];
    int64_t flops = 2 * m * k * n;

    // Only offload large operations (>1M FLOPs)
    if (flops < 1000000) {
      return failure();
    }

    // Create offload operation for matrix multiplication
    auto offloadOp = rewriter.create<OffloadOp>(
        loc, result.getType(),
        rewriter.getStringAttr("matmul"),
        ValueRange{lhs, rhs}
    );

    // Build the offload body with optimized memory access patterns
    Block *offloadBody = rewriter.createBlock(&offloadOp.getBody());
    rewriter.setInsertionPointToStart(offloadBody);

    // For now, create a simple placeholder implementation
    // In a real implementation, this would generate optimized loops
    // with remote memory access patterns

    // Convert tensors to memrefs for memory operations
    auto lhsMemRef = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(lhsType.getShape(), lhsType.getElementType()));
    auto rhsMemRef = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(rhsType.getShape(), rhsType.getElementType()));
    auto resultMemRef = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(llvm::cast<TensorType>(result.getType()).getShape(),
                           llvm::cast<TensorType>(result.getType()).getElementType()));

    rewriter.replaceOp(matmulOp, offloadOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TOSA Conv2D to Cira Offload Pattern
//===----------------------------------------------------------------------===//

/// Convert TOSA Conv2D operations to Cira offload operations
struct TOSAConv2DToCiraPattern : public OpRewritePattern<tosa::Conv2DOp> {
  using OpRewritePattern<tosa::Conv2DOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::Conv2DOp convOp,
                                PatternRewriter &rewriter) const override {
    Location loc = convOp.getLoc();

    auto input = convOp.getInput();
    auto weight = convOp.getWeight();
    auto bias = convOp.getBias();
    auto result = convOp.getResult();

    auto inputType = llvm::cast<TensorType>(input.getType());
    auto weightType = llvm::cast<TensorType>(weight.getType());

    if (!inputType.hasStaticShape() || !weightType.hasStaticShape()) {
      return failure();
    }

    // Calculate convolution workload
    auto inputShape = inputType.getShape();
    auto weightShape = weightType.getShape();

    // Estimate FLOPs for convolution
    int64_t batch = inputShape[0];
    int64_t inChannels = inputShape[3];
    int64_t outChannels = weightShape[0];
    int64_t kernelH = weightShape[1];
    int64_t kernelW = weightShape[2];
    int64_t outputH = inputShape[1]; // Simplified - should account for padding/stride
    int64_t outputW = inputShape[2];

    int64_t flops = batch * outputH * outputW * outChannels * inChannels * kernelH * kernelW;

    // Only offload compute-intensive convolutions
    if (flops < 5000000) {
      return failure();
    }

    // Create offload operation for convolution
    ValueRange operands = bias ? ValueRange{input, weight, bias} : ValueRange{input, weight};
    auto offloadOp = rewriter.create<OffloadOp>(
        loc, result.getType(),
        rewriter.getStringAttr("conv2d"),
        operands
    );

    // Build optimized convolution body with tiling and prefetching
    Block *offloadBody = rewriter.createBlock(&offloadOp.getBody());
    rewriter.setInsertionPointToStart(offloadBody);

    // Placeholder for optimized convolution implementation
    // Would include:
    // - Input/weight tiling for cache efficiency
    // - Prefetching for remote memory access
    // - Vectorized inner loops

    rewriter.replaceOp(convOp, offloadOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TOSA Reduce Operations to Cira Pattern
//===----------------------------------------------------------------------===//

/// Convert TOSA reduction operations to Cira offload operations
struct TOSAReduceToCiraPattern : public OpRewritePattern<tosa::ReduceSumOp> {
  using OpRewritePattern<tosa::ReduceSumOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ReduceSumOp reduceOp,
                                PatternRewriter &rewriter) const override {
    Location loc = reduceOp.getLoc();

    auto input = reduceOp.getInput();
    auto inputType = llvm::cast<TensorType>(input.getType());

    if (!inputType.hasStaticShape()) {
      return failure();
    }

    // Calculate reduction size
    int64_t numElements = inputType.getNumElements();

    // Only offload large reductions
    if (numElements < 100000) {
      return failure();
    }

    // Create offload operation for reduction
    auto offloadOp = rewriter.create<OffloadOp>(
        loc, reduceOp.getResult().getType(),
        rewriter.getStringAttr("reduce_sum"),
        ValueRange{input}
    );

    // Build reduction body with streaming access patterns
    Block *offloadBody = rewriter.createBlock(&offloadOp.getBody());
    rewriter.setInsertionPointToStart(offloadBody);

    // Placeholder for optimized reduction implementation
    // Would include:
    // - Hierarchical reduction across memory tiers
    // - Streaming access for large tensors
    // - Parallel reduction trees

    rewriter.replaceOp(reduceOp, offloadOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TOSA to Cira Conversion Pass
//===----------------------------------------------------------------------===//

struct TOSAToCiraPass : public PassWrapper<TOSAToCiraPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TOSAToCiraPass)

  StringRef getArgument() const override { return "tosa-to-cira"; }

  StringRef getDescription() const override {
    return "Convert TOSA operations to Cira offload operations";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<cira::RemoteMemDialect>();
    registry.insert<tosa::TosaDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<tensor::TensorDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    populateTOSAToCiraPatterns(context, patterns);

    ConversionTarget target(*context);
    target.addLegalDialect<cira::RemoteMemDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<tensor::TensorDialect>();

    // Mark large TOSA operations as illegal to force conversion
    target.addDynamicallyLegalOp<tosa::MatMulOp>([](tosa::MatMulOp op) {
      auto lhsType = llvm::cast<TensorType>(op.getA().getType());
      auto rhsType = llvm::cast<TensorType>(op.getB().getType());

      if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape()) {
        return true; // Keep dynamic shapes as legal
      }

      int64_t m = lhsType.getShape()[0];
      int64_t k = lhsType.getShape()[1];
      int64_t n = rhsType.getShape()[1];
      int64_t flops = 2 * m * k * n;

      return flops < 1000000; // Small operations remain legal
    });

    target.addDynamicallyLegalOp<tosa::Conv2DOp>([](tosa::Conv2DOp op) {
      auto inputType = llvm::cast<TensorType>(op.getInput().getType());
      auto weightType = llvm::cast<TensorType>(op.getWeight().getType());

      if (!inputType.hasStaticShape() || !weightType.hasStaticShape()) {
        return true;
      }

      auto inputShape = inputType.getShape();
      auto weightShape = weightType.getShape();

      int64_t batch = inputShape[0];
      int64_t inChannels = inputShape[3];
      int64_t outChannels = weightShape[0];
      int64_t kernelH = weightShape[1];
      int64_t kernelW = weightShape[2];
      int64_t outputH = inputShape[1];
      int64_t outputW = inputShape[2];

      int64_t flops = batch * outputH * outputW * outChannels * inChannels * kernelH * kernelW;

      return flops < 5000000; // Small convolutions remain legal
    });

    target.addDynamicallyLegalOp<tosa::ReduceSumOp>([](tosa::ReduceSumOp op) {
      auto inputType = llvm::cast<TensorType>(op.getInput().getType());

      if (!inputType.hasStaticShape()) {
        return true;
      }

      return inputType.getNumElements() < 100000; // Small reductions remain legal
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

void mlir::cira::populateTOSAToCiraPatterns(MLIRContext *ctx, RewritePatternSet &patterns) {
  patterns.add<TOSAMatMulToCiraPattern, TOSAConv2DToCiraPattern, TOSAReduceToCiraPattern>(ctx);
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::cira::createTOSAToCiraPass() {
  return std::make_unique<TOSAToCiraPass>();
}

//===----------------------------------------------------------------------===//
// TOSA Offloading Analysis Implementation
//===----------------------------------------------------------------------===//

TOSAOffloadAnalysis TOSAOffloadAnalysis::analyze(func::FuncOp func) {
  TOSAOffloadAnalysis analysis;

  func.walk([&](Operation *op) {
    if (auto matmulOp = dyn_cast<tosa::MatMulOp>(op)) {
      analysis.memoryIntensiveOps.push_back(op);
      analysis.graphOps.push_back(op);

      auto lhsType = llvm::cast<TensorType>(matmulOp.getA().getType());
      if (lhsType.hasStaticShape() && lhsType.getNumElements() > 10000) {
        analysis.remoteAccessOps.push_back(op);
      }
    }
    else if (auto convOp = dyn_cast<tosa::Conv2DOp>(op)) {
      analysis.memoryIntensiveOps.push_back(op);
      analysis.graphOps.push_back(op);

      auto inputType = llvm::cast<TensorType>(convOp.getInput().getType());
      if (inputType.hasStaticShape() && inputType.getNumElements() > 50000) {
        analysis.remoteAccessOps.push_back(op);
      }
    }
    else if (auto reduceOp = dyn_cast<tosa::ReduceSumOp>(op)) {
      analysis.memoryIntensiveOps.push_back(op);

      auto inputType = llvm::cast<TensorType>(reduceOp.getInput().getType());
      if (inputType.hasStaticShape() && inputType.getNumElements() > 100000) {
        analysis.remoteAccessOps.push_back(op);
      }
    }
  });

  return analysis;
}

bool TOSAOffloadAnalysis::shouldOffload(Operation *op) const {
  return llvm::find(remoteAccessOps, op) != remoteAccessOps.end();
}

//===----------------------------------------------------------------------===//
// Memory Tier Selection Implementation
//===----------------------------------------------------------------------===//

TensorMemoryTier mlir::cira::selectMemoryTier(Value tensor, const TOSAOffloadAnalysis &analysis) {
  auto tensorType = llvm::cast<TensorType>(tensor.getType());

  if (!tensorType.hasStaticShape()) {
    return TensorMemoryTier::LOCAL_DRAM; // Conservative choice for dynamic shapes
  }

  int64_t numElements = tensorType.getNumElements();
  int64_t sizeInBytes = numElements * tensorType.getElementTypeBitWidth() / 8;

  // Memory tier selection based on size and access patterns
  if (sizeInBytes < 1024 * 1024) { // < 1MB
    return TensorMemoryTier::LOCAL_CACHE;
  }
  else if (sizeInBytes < 16 * 1024 * 1024) { // < 16MB
    return TensorMemoryTier::LOCAL_DRAM;
  }
  else if (sizeInBytes < 256 * 1024 * 1024) { // < 256MB
    return TensorMemoryTier::CXL_ATTACHED;
  }
  else if (sizeInBytes < 1024 * 1024 * 1024) { // < 1GB
    return TensorMemoryTier::CXL_POOLED;
  }
  else {
    return TensorMemoryTier::FAR_MEMORY;
  }
}