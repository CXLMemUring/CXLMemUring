#include "Dialect/FunctionUtils.h"
#include "Dialect/RemoteMem.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/IR/DataLayout.h"
#include "Lowering/EmitLLVM.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"

namespace mlir {
#define GEN_PASS_DEF_EMITLLVM
#include "Lowering/Passes.h.inc"
 using namespace mlir::cira;

namespace {
class EmitLLVMPass : public ::mlir::impl::EmitLLVMBase<EmitLLVMPass> {
public:
    EmitLLVMPass() = default;
    void runOnOperation() override {
        ModuleOp m = getOperation();

        const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
        LowerToLLVMOptions options(&getContext(), dataLayoutAnalysis.getAtOrAbove(m));

        LLVMTypeConverter llvmTypeConverter(&getContext(), options);
        RewritePatternSet patterns(&getContext());
        mlir::populateEmitLLVMPatterns(llvmTypeConverter, patterns);

        ConversionTarget target(getContext());
        target.addLegalDialect<LLVM::LLVMDialect>();
        target.addLegalOp<ModuleOp>();
        target.addLegalOp<UnrealizedConversionCastOp>();
        target.addIllegalDialect<cira::RemoteMemDialect>();
        // Generic target that fileter out most of operations
        // target.markUnknownOpDynamicallyLegal([](Operation *op) {
        //   return !(llvm::any_of(op->getOperandTypes(), rmem::hasRemoteTarget) || llvm::any_of(op->getResultTypes(),
        //   rmem::hasRemoteTarget));
        // });
        // target.addDynamicallyLegalOp<func::FuncOp>([](func::FuncOp op) {
        //   return (!llvm::any_of(op.getArgumentTypes(), rmem::hasRemoteTarget)) && (!llvm::any_of(op.getResultTypes(),
        //   rmem::hasRemoteTarget));
        // });

        if (failed(applyFullConversion(m, target, std::move(patterns))))
            signalPassFailure();

        /* add disagg env init */
//        auto mainFunc = m.lookupSymbol<func::FuncOp>("main");
//        if (mainFunc) {
//            /* call inits and shutdown */
//            auto initClientOp = cira::lookupOrCreateInitClientFn(m);
//            // auto initCacheOp = rmem::lookupOrCreateCacheInitFn(m);
//            // auto initChannelOp = rmem::lookupOrCreateChannelInitFn(m);
//
//            OpBuilder b(mainFunc.getBody());
//            cira::createLLVMCall(b, mainFunc.getLoc(), initClientOp);
//            // rmem::createLLVMCall(b, mainFunc.getLoc(), initCacheOp);
//            // rmem::createLLVMCall(b, mainFunc.getLoc(), initChannelOp);
//        }
    }
};

} // namespace

void populateEmitLLVMPatterns(LLVMTypeConverter &llvmTypeConverter, RewritePatternSet &patterns) {
//    populateRemoteMemToLLVMPatterns(patterns);

    vector::populateVectorToVectorCanonicalizationPatterns(patterns);
//    vector::populateVectorBroadcastLoweringPatterns(patterns);
    // vector::populateVectorContractLoweringPatterns(patterns);
    // vector::populateVectorTransposeLoweringPatterns(patterns);

    populateAffineToStdConversionPatterns(patterns);
    populateSCFToControlFlowConversionPatterns(patterns);

    populateShapeToStandardConversionPatterns(patterns);
    populateVectorToLLVMMatrixConversionPatterns(llvmTypeConverter, patterns);
    populateVectorToLLVMConversionPatterns(llvmTypeConverter, patterns);
    populateVectorToLLVMMatrixConversionPatterns(llvmTypeConverter, patterns);
//    memref::populateExpandOpsPatterns(patterns);
    // Use polynomial approximation for math.{tanh, sin, cos and exp} for better
    // performance.
    populateMathPolynomialApproximationPatterns(patterns);
    arith::populateArithExpandOpsPatterns(patterns);
    populateMathToLLVMConversionPatterns(llvmTypeConverter, patterns);
    populateFuncToLLVMConversionPatterns(llvmTypeConverter, patterns);
    populateVectorToLLVMConversionPatterns(llvmTypeConverter, patterns);
    // populateFinalizeMemRefToLLVMConversionPatterns(llvmTypeConverter, patterns);
    arith::populateArithToLLVMConversionPatterns(llvmTypeConverter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(llvmTypeConverter, patterns);

    // populateReconcileUnrealizedCastsPatterns(patterns); // Not available in this MLIR version
}

std::unique_ptr<Pass> createEmitLLVMPass() { return std::make_unique<EmitLLVMPass>(); }
} // namespace mlir
