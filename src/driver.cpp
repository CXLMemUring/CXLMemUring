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

// ClangIR: CIR passes and pipelines
#include <clang/CIR/Passes.h>
#include <clang/CIR/Dialect/Passes.h>

using namespace mlir;

#include <clang/CIR/Dialect/IR/CIRDialect.h>

class MemRefInsider : public mlir::MemRefElementTypeInterface::FallbackModel<MemRefInsider> {};

template <typename T>
struct PtrElementModel : public mlir::LLVM::PointerElementTypeInterface::ExternalModel<PtrElementModel<T>, T> {};

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    registerAllDialects(registry);
    registry.insert<mlir::cira::RemoteMemDialect>();

    registry.insert<cir::CIRDialect>();
    // Register CIR dialect-level passes (canonicalize/simplify)
    {
        // Functions are inline in Passes.h.inc under namespace mlir
        // Brings pass names like `cir-canonicalize` into the registry
        mlir::registerCIRPasses();
    }

    // Provide a pipeline to lower CIR -> core MLIR for use in -pass-pipeline
    static mlir::PassPipelineRegistration<> cirLowerToMlirPipeline(
        "cir-lower-to-mlir",
        "Lower ClangIR (CIR) to core MLIR dialects",
        [](mlir::OpPassManager &pm) {
            pm.addPass(::cir::createConvertCIRToMLIRPass());
        });

    // Optional: Direct CIR -> LLVM dialect lowering (bypasses core MLIR)
    static mlir::PassPipelineRegistration<> cirDirectToLLVMPipeline(
        "cir-direct-to-llvm",
        "Lower ClangIR (CIR) directly to LLVM dialect",
        [](mlir::OpPassManager &pm) {
            pm.addPass(::cir::direct::createConvertCIRToLLVMPass());
        });

    static mlir::PassPipelineRegistration<> cirDirectToLLVMPipeline2(
        "cir-direct-to-llvm-pipeline",
        "Run CIR->LLVM lowering pipeline (with CC lowering)",
        [](mlir::OpPassManager &pm) {
            ::cir::direct::populateCIRToLLVMPasses(pm, /*useCCLowering=*/true);
        });
    // register remote mem related passes
    mlir::registerCIRAConversionPasses();
    mlir::registerCIRALoweringPasses();
    mlir::registerRemoteMemPasses();

    // register normal passes
    mlir::registerAllPasses();
    mlir::registerCSEPass();
    mlir::registerInlinerPass();
    mlir::registerCanonicalizerPass();
    mlir::registerSymbolDCEPass();
    mlir::registerLoopInvariantCodeMotionPass();

    // register util passes (none)

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
