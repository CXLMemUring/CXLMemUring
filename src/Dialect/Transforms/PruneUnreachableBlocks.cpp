#include "Dialect/Transforms/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir {
#define GEN_PASS_DEF_CIRAPRUNEUNREACHABLEBLOCKS
#include "Dialect/Transforms/Passes.h.inc"
} // namespace mlir

namespace {

struct CiraPruneUnreachableBlocksPass
    : public mlir::impl::CiraPruneUnreachableBlocksBase<
          CiraPruneUnreachableBlocksPass> {
  void runOnOperation() override {
    mlir::IRRewriter rewriter(&getContext());
    (void)mlir::eraseUnreachableBlocks(rewriter,
                                       getOperation()->getRegions());
  }
};

} // namespace
