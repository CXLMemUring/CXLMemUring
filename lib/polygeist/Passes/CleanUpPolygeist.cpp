#include "PassDetails.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "polygeist/Ops.h"

using namespace mlir;
using namespace polygeist;

namespace {
struct LLVMOpLowering : public ConversionPattern {
  explicit LLVMOpLowering(LLVMTypeConverter &converter)
      : ConversionPattern(converter, Pattern::MatchAnyOpTypeTag(), 1,
                          &converter.getContext()) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    TypeConverter *converter = getTypeConverter();
    SmallVector<Type> convertedResultTypes;
    if (failed(converter->convertTypes(op->getResultTypes(),
                                       convertedResultTypes))) {
      return failure();
    }
    SmallVector<Type> convertedOperandTypes;
    if (failed(converter->convertTypes(op->getOperandTypes(),
                                       convertedOperandTypes))) {
      return failure();
    }
    if (convertedResultTypes == op->getResultTypes() &&
        convertedOperandTypes == op->getOperandTypes()) {
      return failure();
    }
    if (isa<UnrealizedConversionCastOp>(op))
      return failure();

    OperationState state(op->getLoc(), op->getName());
    state.addOperands(operands);
    state.addTypes(convertedResultTypes);
    state.addAttributes(op->getAttrs());
    state.addSuccessors(op->getSuccessors());
    for (unsigned i = 0, e = op->getNumRegions(); i < e; ++i)
      state.addRegion();

    Operation *rewritten = rewriter.create(state);
    rewriter.replaceOp(op, rewritten->getResults());

    for (unsigned i = 0, e = op->getNumRegions(); i < e; ++i)
      rewriter.inlineRegionBefore(op->getRegion(i), rewritten->getRegion(i),
                                  rewritten->getRegion(i).begin());

    return success();
  }
};
}

namespace {
class CleanUpPolygeistOpPass : public CleanUpPolygeistBase<CleanUpPolygeistOpPass> {
public:
  CleanUpPolygeistOpPass() = default;
  CleanUpPolygeistOpPass(bool useBarePtrCallConv, unsigned indexBitwidth,
                             bool useAlignedAlloc,
                             const llvm::DataLayout &dataLayout,
                             bool useCStyleMemRef) {
    this->useBarePtrCallConv = useBarePtrCallConv;
    this->indexBitwidth = indexBitwidth;
    this->dataLayout = dataLayout.getStringRepresentation();
    this->useCStyleMemRef = useCStyleMemRef;
  }
  void runOnOperation() override {
    ModuleOp m = getOperation();
    const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();

    if (useCStyleMemRef && useBarePtrCallConv) {
      emitError(m.getLoc()) << "C-style memref lowering is not compatible with "
                               "bare-pointer calling convention";
      signalPassFailure();
      return;
    }

    LowerToLLVMOptions options(&getContext(),
                               dataLayoutAnalysis.getAtOrAbove(m));
    options.useBarePtrCallConv = useBarePtrCallConv;
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);

    options.dataLayout = llvm::DataLayout(this->dataLayout);

    LLVMTypeConverter converter(&getContext(), options, &dataLayoutAnalysis);
    RewritePatternSet patterns(&getContext());
    populatePolygeistToLLVMConversionPatterns(converter, patterns);
    populateMemRefToLLVMConversionPatterns(converter, patterns);
    populateFuncToLLVMConversionPatterns(converter, patterns);
    patterns.add<LLVMOpLowering>(converter);

    auto areAllTypesConverted = [&](Operation *op) -> Optional<bool> {
        SmallVector<Type> convertedResultTypes;
        if (failed(converter.convertTypes(op->getResultTypes(),
                                          convertedResultTypes)))
          return llvm::None;
        SmallVector<Type> convertedOperandTypes;
        if (failed(converter.convertTypes(op->getOperandTypes(),
                                          convertedOperandTypes)))
          return llvm::None;
        return convertedResultTypes == op->getResultTypes() &&
               convertedOperandTypes == op->getOperandTypes();
      };
    LLVMConversionTarget target(getContext());
    target.addDynamicallyLegalDialect<LLVM::LLVMDialect>(
          areAllTypesConverted);
    target.addDynamicallyLegalOp<LLVM::GlobalOp>(
        [&](LLVM::GlobalOp op) -> Optional<bool> {
          if (converter.convertType(op.getGlobalType()) == op.getGlobalType())
            return true;
          return llvm::None;
        });
    target.addDynamicallyLegalOp<LLVM::ReturnOp>(
        [&](LLVM::ReturnOp op) -> Optional<bool> {
          // Outside global ops, defer to the normal type-based check. Note
          // that the infrastructure will not do it automatically because
          // per-op checks override dialect-level checks unconditionally.
          if (!isa<LLVM::GlobalOp>(op->getParentOp()))
            return areAllTypesConverted(op);

          SmallVector<Type> convertedOperandTypes;
          if (failed(converter.convertTypes(op->getOperandTypes(),
                                            convertedOperandTypes)))
            return llvm::None;
          return convertedOperandTypes == op->getOperandTypes();
        });
    target.addIllegalOp<UnrealizedConversionCastOp>();
    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      signalPassFailure();
  }
};
}

std::unique_ptr<Pass> mlir::polygeist::createCleanUpPolygeistOpPass(const LowerToLLVMOptions &options, bool useCStyleMemRef) {
  auto allocLowering = options.allocLowering;
  // There is no way to provide additional patterns for pass, so
  // AllocLowering::None will always fail.
  assert(allocLowering != LowerToLLVMOptions::AllocLowering::None &&
         "LLVMLoweringPass doesn't support AllocLowering::None");
  bool useAlignedAlloc =
      (allocLowering == LowerToLLVMOptions::AllocLowering::AlignedAlloc);
  return std::make_unique<CleanUpPolygeistOpPass>(
      options.useBarePtrCallConv, options.getIndexBitwidth(), useAlignedAlloc,
      options.dataLayout, useCStyleMemRef);
}

std::unique_ptr<Pass> mlir::polygeist::createCleanUpPolygeistOpPass() {
  auto dl = llvm::DataLayout("");
  return std::make_unique<CleanUpPolygeistOpPass>(false, 64u, false, dl, true);
}