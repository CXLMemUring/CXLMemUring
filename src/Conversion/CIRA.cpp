#include "Conversion/CIRA.h"
#include "Dialect/RemoteMem.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/IR/DataLayout.h"

namespace mlir {
#define GEN_PASS_DEF_CIRA
#include "Conversion/Passes.h.inc"

// =================================================================================================================

namespace cira {
class CIRAPass : public ::mlir::impl::CIRABase<CIRAPass> {
public:
    CIRAPass() = default;
    void runOnOperation() override {
        Operation *op = getOperation();
        RewritePatternSet patterns(&getContext());

        populateCIRAPatterns(&getContext(), patterns);

        ConversionTarget target(getContext());

        if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

void populateCIRAPatterns(MLIRContext *ctx, RewritePatternSet &patterns) { populateSCFCIRAPatterns(ctx, patterns); }

std::unique_ptr<Pass> createCIRAPass() { return std::make_unique<CIRAPass>(); }

} // namespace cira
} // namespace mlir