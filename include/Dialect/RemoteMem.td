#ifndef REMOTEMEMREF_BASE
#define REMOTEMEMREF_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinDialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

// basic block top to remote memory
def RemoteMemDialect : Dialect {
  let name = "cira";
  let cppNamespace = "::mlir::cira";
  let description = [{The `cira` dialect provides a unified interface to the remote memory access. The dialect will be converted to raw pointer or remotable access eventually}];
  let extraClassDeclaration = [{
  private:
    void registerTypes();

  public:
  }];
  // let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}
class RemoteMem_Op<string mnemonic, list<Trait> traits = []>:
  Op<RemoteMemDialect, mnemonic, traits>;

/* RemoteMemRef Type Definitions */
def IsRemoteMemRefTypePred : CPred<"llvm::isa<::mlir::cira::RemoteMemRefType>($_self)">;

class RemoteMemRefOf<list<Type> allowedTypes> :
  ContainerType<
    AnyTypeOf<allowedTypes>,
    IsRemoteMemRefTypePred,
    "$_self.cast<::mlir::cira::RemoteMemRefType>().getElementType()",
    "Remote mem ref of accepted type",
    "::mlir::cira::RemoteMemRefType">;

// For future extensions
def AnyRemoteMemRef : RemoteMemRefOf<[AnyType]>;

// Used for now
// AllowedRemoteElement = [AnyMemRef, LLVM_AnyPointer]
def RemoteMemRef : RemoteMemRefOf<[AnyMemRef, LLVM_AnyPointer]>;

class Detail_RemoteMemType<string name, string typeMnemonic, list<Trait> traits = []>
  :TypeDef<RemoteMemDialect, name, traits> {
    let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// CIRA Type System - !cira.handle<T>, !cira.future<T>, !cira.stream
//===----------------------------------------------------------------------===//

// Handle type - a pointer guaranteed to reside in CXL space
def Cira_HandleType : Detail_RemoteMemType<"Handle", "handle"> {
  let summary = "A pointer guaranteed to reside in CXL memory space";
  let description = [{
    The `!cira.handle<T>` type represents a pointer to data that is guaranteed
    to reside in CXL-attached memory. This type enables the compiler to reason
    about memory locality and generate appropriate prefetch/offload operations.

    Example:
    ```mlir
    %ptr = cira.alloc_cxl : !cira.handle<i64>
    ```
  }];
  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}

// Type predicate for Handle type
def IsCiraHandleTypePred : CPred<"llvm::isa<::mlir::cira::HandleType>($_self)">;
def CiraHandle : Type<IsCiraHandleTypePred, "cira handle type">;
def AnyCiraHandle : Type<IsCiraHandleTypePred, "any cira handle type">;

// Future type - a token representing an in-flight asynchronous load
def Cira_FutureType : Detail_RemoteMemType<"Future", "future"> {
  let summary = "A token representing an in-flight asynchronous memory operation";
  let description = [{
    The `!cira.future<T>` type represents a handle to an asynchronous memory
    operation (load or store). The result can be obtained by awaiting the future.
    This enables decoupling of memory request initiation from data consumption.

    Example:
    ```mlir
    %fut = cira.load_async %ptr : !cira.handle<i64> -> !cira.future<i64>
    // ... independent computation ...
    %val = cira.future_await %fut : !cira.future<i64> -> i64
    ```
  }];
  let parameters = (ins "Type":$valueType);
  let assemblyFormat = "`<` $valueType `>`";
}

// Type predicate for Future type
def IsCiraFutureTypePred : CPred<"llvm::isa<::mlir::cira::FutureType>($_self)">;
def CiraFuture : Type<IsCiraFutureTypePred, "cira future type">;
def AnyCiraFuture : Type<IsCiraFutureTypePred, "any cira future type">;

// Stream type - a descriptor for a recurring memory pattern
def Cira_StreamType : Detail_RemoteMemType<"Stream", "stream"> {
  let summary = "A descriptor for a recurring memory access pattern";
  let description = [{
    The `!cira.stream` type represents a memory access pattern that can be
    prefetched by the near-memory accelerator. Streams can be:
    - Sequential: linear address progression
    - Strided: fixed stride between accesses
    - Indirect: pointer chasing (e.g., linked list traversal)

    The stream descriptor contains metadata about the pattern that enables
    the Vortex accelerator to speculatively prefetch data ahead of consumption.

    Example:
    ```mlir
    %stream = cira.stream_create_indirect %start_node, offset=8 : !cira.stream
    cira.prefetch_chain %stream, depth=16
    ```
  }];
  let parameters = (ins
    "Type":$elementType,
    OptionalParameter<"int64_t">:$stride,
    OptionalParameter<"int64_t">:$offsetToNext
  );
  let assemblyFormat = "`<` $elementType (`,` `stride` `=` $stride^)? (`,` `offset` `=` $offsetToNext^)? `>`";
}

// Type predicate for Stream type
def IsCiraStreamTypePred : CPred<"llvm::isa<::mlir::cira::StreamType>($_self)">;
def CiraStream : Type<IsCiraStreamTypePred, "cira stream type">;
def AnyCiraStream : Type<IsCiraStreamTypePred, "any cira stream type">;

#endif
