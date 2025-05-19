#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "Dialect/Transforms/Passes.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Conversion/Passes.h"
#include "Dialect/RemoteMem.h"
#include "Lowering/Passes.h"
#include "Lowering/RemoteMemToLLVM.h"
//#include "Utils/Passes.h"

using namespace mlir;

class MemRefInsider : public mlir::MemRefElementTypeInterface::FallbackModel<MemRefInsider> {};

template <typename T>
struct PtrElementModel : public mlir::LLVM::PointerElementTypeInterface::ExternalModel<PtrElementModel<T>, T> {};

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    registerAllDialects(registry);
    registry.insert<mlir::cira::RemoteMemDialect>();

    // register remote mem related passes
    mlir::registerCIRAConversionPasses();
    mlir::registerCIRALoweringPasses();
    mlir::registerRemoteMemPasses();

    // register normal passes
    mlir::registerAllPasses();
    // mlir::registerCSEPass();
    // mlir::registerInlinerPass();
    // mlir::registerCanonicalizerPass();
    // mlir::registerSymbolDCEPass();
    // mlir::registerLoopInvariantCodeMotionPass();

    // register util passes
    //  mlir::registerAllocationAnnotationPass();

    // interface perpare
    registry.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
        LLVM::LLVMFunctionType::attachInterface<MemRefInsider>(*ctx);
    });
    registry.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
        LLVM::LLVMArrayType::attachInterface<MemRefInsider>(*ctx);
    });
    registry.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
        LLVM::LLVMPointerType::attachInterface<MemRefInsider>(*ctx);
    });
    registry.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
        LLVM::LLVMStructType::attachInterface<MemRefInsider>(*ctx);
    });
    registry.addExtension(+[](MLIRContext *ctx, memref::MemRefDialect *dialect) {
        MemRefType::attachInterface<PtrElementModel<MemRefType>>(*ctx);
    });

    registry.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
        LLVM::LLVMStructType::attachInterface<PtrElementModel<LLVM::LLVMStructType>>(*ctx);
    });

    registry.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
        LLVM::LLVMPointerType::attachInterface<PtrElementModel<LLVM::LLVMPointerType>>(*ctx);
    });

    registry.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
        LLVM::LLVMArrayType::attachInterface<PtrElementModel<LLVM::LLVMArrayType>>(*ctx);
    });

    return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "Remote Mem opt driver", registry));
}