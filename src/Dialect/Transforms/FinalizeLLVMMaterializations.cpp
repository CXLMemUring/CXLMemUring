#include "Dialect/Transforms/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
#define GEN_PASS_DEF_CIRAFINALIZELLVMMATERIALIZATIONS
#include "Dialect/Transforms/Passes.h.inc"
} // namespace mlir

namespace {

struct CiraFinalizeLLVMMaterializationsPass
    : public mlir::impl::CiraFinalizeLLVMMaterializationsBase<
          CiraFinalizeLLVMMaterializationsPass> {
  void runOnOperation() override {
    llvm::SmallVector<mlir::UnrealizedConversionCastOp> casts;
    getOperation().walk(
        [&](mlir::UnrealizedConversionCastOp op) { casts.push_back(op); });

    mlir::IRRewriter rewriter(&getContext());
    bool failed = false;
    for (mlir::UnrealizedConversionCastOp op : casts) {
      rewriter.setInsertionPoint(op);
      if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
        op.emitError("unsupported non-scalar LLVM materialization");
        failed = true;
        continue;
      }

      mlir::Value input = op.getOperand(0);
      mlir::Value result = op.getResult(0);
      if (result.use_empty()) {
        rewriter.eraseOp(op);
        continue;
      }
      if (input.getType() == result.getType()) {
        rewriter.replaceOp(op, input);
        continue;
      }

      auto sourcePointer =
          mlir::dyn_cast<mlir::LLVM::LLVMPointerType>(input.getType());
      auto resultPointer =
          mlir::dyn_cast<mlir::LLVM::LLVMPointerType>(result.getType());
      if (sourcePointer && resultPointer) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::AddrSpaceCastOp>(
            op, resultPointer, input);
        continue;
      }

      op.emitError("unsupported LLVM materialization from ")
          << input.getType() << " to " << result.getType();
      failed = true;
    }

    if (failed)
      signalPassFailure();
  }
};

} // namespace
