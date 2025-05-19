//
// Created by yangyw on 8/4/24.
//

#ifndef CIRA_FUNCTIONUTILS_H
#define CIRA_FUNCTIONUTILS_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class Location;
class ModuleOp;
class OpBuilder;
class Operation;
class Type;
class Value;
class ValueRange;

namespace LLVM {
class LLVMFuncOp;
class LLVMVoidType;
class LLVMPointerType;
class LLVMStructType;
}

namespace cira {
Type getIntBitType(MLIRContext *ctx, unsigned bitwidth);
Value createIntConstant(OpBuilder &builder, Location loc, int64_t value, Type resultType);
LLVM::LLVMStructType getStructTokenType(MLIRContext *ctx);
LLVM::LLVMVoidType getVoidType(MLIRContext*);
LLVM::LLVMPointerType getVoidPtrType(MLIRContext *ctx);
llvm::StringRef getNextRemoteAccessName();
// calcualte sizeof(elemType) * arraySize in bytes
// not considering
Value calculateBufferSize(OpBuilder &builder, Location loc, Type elemType, Value arraySize);
LLVM::LLVMFuncOp lookupOrCreateRawMallocFn(ModuleOp moduleOp);
LLVM::LLVMFuncOp lookupOrCreateRawCallocFn(ModuleOp moduleOp);

LLVM::LLVMFuncOp lookupOrCreatRemoteAccess(ModuleOp moduleOp);
// int __llvm_profile_write_file();
LLVM::LLVMFuncOp lookupOrCreateProfileWriteFn(ModuleOp moduleOp);

// unsigned channel_create(
//   uint64_t original_start_vaddr,
//   uint64_t upper_bound,
//   size_t size_each,
//   unsigned num_slots,
//   unsigned batch,
//   unsigned dist,
//   int kind);
LLVM::LLVMFuncOp lookupOrCreateChannelCreateFn(ModuleOp moduleOp);

// void * channel_access(unsigned channel, unsigned i);
LLVM::LLVMFuncOp lookupOrCreateChannelAccessFn(ModuleOp moduleOp);

// void channel_destroy(unsigned channel);
LLVM::LLVMFuncOp lookupOrCreateChannelDestroyFn(ModuleOp moduleOp);

// void rdma(i64 buf, i64 size, i64 raddr, i64 id, i32 code);
LLVM::LLVMFuncOp lookupOrCreateRDMAFn(ModuleOp moduleOp);

// void rring_sync(ptr<i64> s/r, i64 t)
LLVM::LLVMFuncOp lookupOrCreateRRingSync(ModuleOp moduleOp);

// Function for remote offloading
LLVM::LLVMFuncOp lookupOrCreateCallOffloadService(ModuleOp moduleOp);

/* instrumentation intrinsics
1. void @llvm.instrprof.increment(ptr <n>, i64 <hash>,
                                       i32 <num-counters>, i32 <index>)

2. void @llvm.instrprof.increment.step(ptr <name>, i64 <hash>,
                                            i32 <num-counters>,
                                            i32 <index>, i64 <step>)
*/
LLVM::LLVMFuncOp lookupOrCreateInstrInc(ModuleOp moduleOp);
LLVM::LLVMFuncOp lookupOrCreateInstrIncStep(ModuleOp moduleOp);

Value cacheRequestCallWrapper(OpBuilder &builder, Location loc, LLVM::LLVMFuncOp reqFn, Value ptr);

Value cacheAccessCallWrapper(OpBuilder &builder, Location loc,
                             LLVM::LLVMFuncOp accFn,
                             Value token_ptr /* int128_t* */,
                             Type castPtrType);

LLVM::LLVMFuncOp lookupOrCreateFn(ModuleOp moduleOp,
                                  StringRef name,
                                  ArrayRef<Type> inputTypes = {},
                                  Type resultType = {});
Operation::result_range createLLVMCall(OpBuilder &builder,
                                       Location loc,
                                       LLVM::LLVMFuncOp fn,
                                       ValueRange inputs = {});

}
}
#endif // CIRA_FUNCTIONUTILS_H
