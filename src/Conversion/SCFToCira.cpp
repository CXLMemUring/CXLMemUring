#include "Conversion/CIRA.h"
#include "Dialect/CiraOps.h"
#include "Dialect/RemoteMem.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::cira;

namespace {

/// Pattern to detect graph traversal patterns in SCF loops and annotate for CIRA
///
/// Note: Keep this pattern non-destructive for now. We conservatively add
/// attributes to mark candidates instead of restructuring IR. This makes the
/// frontend verifiable without breaking downstream lowering.
struct GraphTraversalPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    // Check if this loop exhibits typical graph traversal characteristics:
    // - At least one nested loop or multiple memory accesses
    // - Contains indirect access patterns (loads whose indices depend on prior loads)

    bool hasNestedLoop = false;
    bool hasLoads = false;
    bool hasIndirect = false;

    forOp.getBody()->walk([&](Operation *op) {
      if (isa<scf::ForOp, scf::WhileOp>(op))
        hasNestedLoop = true;
      if (auto ld = dyn_cast<memref::LoadOp>(op)) {
        hasLoads = true;
        // Heuristic: any index coming from a non-constant defining op
        // indicates potential indirect access; if that defining op is
        // itself a load/index cast/fptosi chain, mark indirect.
        for (Value idx : ld.getIndices()) {
          if (!idx.getDefiningOp())
            continue;
          Operation *def = idx.getDefiningOp();
          if (!isa<arith::ConstantIndexOp, arith::ConstantOp>(def))
            hasIndirect = true;
        }
      }
    });

    if (!(hasLoads && (hasNestedLoop || hasIndirect)))
      return failure();

    // Annotate this loop as a remote/offload candidate. Keep IR unchanged.
    rewriter.modifyOpInPlace(forOp, [&] {
      forOp->setAttr("cira.remote_candidate", rewriter.getUnitAttr());
      // Optional: store a simple hint for prefetch distance.
      forOp->setAttr("cira.prefetch_dist", rewriter.getI64IntegerAttr(8));
    });
    return success();
  }
};

/// Pattern to tag memory loads that likely hit remote data paths
struct MemRefLoadToCiraLoad : public RewritePattern {
  MemRefLoadToCiraLoad(MLIRContext *context)
      : RewritePattern(memref::LoadOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto loadOp = cast<memref::LoadOp>(op);
    auto memrefType = dyn_cast<MemRefType>(loadOp.getMemRef().getType());
    if (!memrefType)
      return failure();

    // Simple heuristic: tag 2D+ loads (e.g., edge tables) and 1D loads where
    // the index is computed from previous loads (node arrays) as remote.
    bool candidate = false;
    if (memrefType.getRank() >= 2) {
      candidate = true;
    } else {
      // Check for indirect index
      for (Value idx : loadOp.getIndices()) {
        if (Operation *def = idx.getDefiningOp()) {
          if (!isa<arith::ConstantIndexOp, arith::ConstantOp>(def)) {
            candidate = true;
            break;
          }
        }
      }
    }

    if (!candidate)
      return failure();

    rewriter.modifyOpInPlace(loadOp, [&] {
      loadOp->setAttr("cira.remote_load", rewriter.getUnitAttr());
    });
    return success();
  }
};

} // namespace

namespace mlir {
namespace cira {

void populateSCFToCiraPatterns(MLIRContext *ctx, RewritePatternSet &patterns) {
  patterns.add<GraphTraversalPattern, MemRefLoadToCiraLoad>(ctx);
}

} // namespace cira
} // namespace mlir
