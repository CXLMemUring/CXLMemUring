//
// Created by yangyw on 8/4/24.
//
#include "compat/LLVM.h"
#include "Dialect/FunctionUtils.h"
#include "Dialect/RemoteMem.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::cira;

static const char * kRemoteAccess = "remote_access_%s";
static int kRemoteAccessNum = 0;
static constexpr llvm::StringRef kInstrProfInc = "llvm.instrprof.increment";
static constexpr llvm::StringRef kInstrProfIncStep = "llvm.instrprof.increment.step";
static constexpr llvm::StringRef kDumpProfile = "__llvm_profile_write_file";

Value mlir::cira::createIntConstant(OpBuilder &builder, Location loc, int64_t value, Type resultType) {
    return builder.create<LLVM::ConstantOp>(
        loc, resultType, builder.getIntegerAttr(resultType, value)
    );
}
// llvm.mlir.global external @tokens() {addr_space = 0 : i32} : !llvm.array<33554432 x struct<"struct.Token", (i64, i8, i8, i16, i32)>>
LLVM::LLVMStructType mlir::cira::getStructTokenType(MLIRContext *ctx) {
    /*
    typedef struct cache_token_t {
      uint64_t tag;
      uint8_t flags;
      uint8_t pad0;
      uint16_t seq;
      pthread_spinlock_t lock;
    } cache_token_t;
    */
    return LLVM::LLVMStructType::getLiteral(ctx,
                                            {
                                                getIntBitType(ctx, 64),
                                                getIntBitType(ctx, 8),
                                                getIntBitType(ctx, 8),
                                                getIntBitType(ctx, 16),
                                                getIntBitType(ctx, 32),
                                            }
    );
}

LLVM::LLVMVoidType mlir::cira::getVoidType(MLIRContext *ctx) {
    return LLVM::LLVMVoidType::get(ctx);
}

LLVM::LLVMPointerType mlir::cira::getVoidPtrType(MLIRContext *ctx) {
    return LLVM::LLVMPointerType::get(getIntBitType(ctx, 8));
}

llvm::StringRef mlir::cira::getNextRemoteAccessName() {
    static char buffer[50];  // Make buffer static to ensure the string persists
    snprintf(buffer, sizeof(buffer), kRemoteAccess, kRemoteAccessNum++);
    return llvm::StringRef(buffer);
}
LLVM::LLVMFuncOp mlir::cira::lookupOrCreatRemoteAccess(ModuleOp moduleOp) {
    auto ctx = moduleOp.getContext();

    return cira::lookupOrCreateFn(
        moduleOp, getNextRemoteAccessName(),
        {getVoidPtrType(ctx), getIntBitType(ctx, 64), getIntBitType(ctx, 32), getIntBitType(ctx, 32)}, {});
}

LLVM::LLVMFuncOp mlir::cira::lookupOrCreateInstrInc(ModuleOp moduleOp) {
    auto ctx = moduleOp.getContext();
    return cira::lookupOrCreateFn(
        moduleOp, kInstrProfInc,
        {getVoidPtrType(ctx), getIntBitType(ctx, 64), getIntBitType(ctx, 32), getIntBitType(ctx, 32)}, {});
}

LLVM::LLVMFuncOp mlir::cira::lookupOrCreateInstrIncStep(ModuleOp moduleOp) {
    auto ctx = moduleOp.getContext();
    return cira::lookupOrCreateFn(
        moduleOp, kInstrProfIncStep,
        {getVoidPtrType(ctx), getIntBitType(ctx, 64), getIntBitType(ctx, 32), getIntBitType(ctx, 32)}, {});
}

LLVM::LLVMFuncOp mlir::cira::lookupOrCreateProfileWriteFn(ModuleOp moduleOp) {
    auto ctx = moduleOp.getContext();
    return cira::lookupOrCreateFn(moduleOp, kDumpProfile, {}, {getIntBitType(ctx, 32)});
}
LLVM::LLVMFuncOp mlir::cira::lookupOrCreateFn(ModuleOp moduleOp, StringRef name, ArrayRef<Type> inputTypes,
                                              Type resultType) {
    auto func = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(name);
    if (func)
        return func;
    OpBuilder b(moduleOp.getBodyRegion());
    return b.create<LLVM::LLVMFuncOp>(
        moduleOp->getLoc(), name,
        LLVM::LLVMFunctionType::get(resultType ? resultType : getVoidType(moduleOp.getContext()), inputTypes));
}
Operation::result_range mlir::cira::createLLVMCall(OpBuilder &builder, Location loc, LLVM::LLVMFuncOp fn,
                                                   ValueRange inputs) {

    return builder.create<LLVM::CallOp>(loc, fn, inputs)->getResults();
}

LLVM::LLVMFuncOp mlir::cira::lookupOrCreateCallOffloadService(ModuleOp moduleOp) {
    auto ctx = moduleOp.getContext();
    return cira::lookupOrCreateFn(
        moduleOp, 
        "call_offload_service",
        {getIntBitType(ctx, 64), getIntBitType(ctx, 64), getIntBitType(ctx, 64)}, 
        getVoidPtrType(ctx));
}

Type mlir::cira::getIntBitType(MLIRContext *ctx, unsigned bitwidth) {
    return IntegerType::get(ctx, bitwidth);
}
