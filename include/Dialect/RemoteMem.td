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
def IsRemoteMemRefTypePred : CPred<"$_self.isa<::mlir::cira::RemoteMemRefType>()">;

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

#endif
