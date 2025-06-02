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

def RemoteMem_OffloadOp : RemoteMem_Op<"offload", [
  AffineScope, AutomaticAllocationScope,
   IsolatedFromAbove
]> {
  let summary = "An operation with a name containing a single `SSACFG` region";
  let description = [{
    Operations within the function cannot implicitly capture values defined
    outside of the function, i.e. Functions are `IsolatedFromAbove`. All
    external references must use function arguments or attributes that establish
    a symbolic connection (e.g. symbols referenced by name via a string
    attribute like SymbolRefAttr). An external function declaration (used when
    referring to a function declared in some other module) has no body. While
    the MLIR textual form provides a nice inline syntax for function arguments,
    they are internally represented as "block arguments" to the first block in
    the region.

    Only dialect attribute names may be specified in the attribute dictionaries
    for function arguments, results, or the function itself.

    Example:

    ```mlir
    // External function definitions.
    func.func @abort()
    func.func @scribble(i32, i64, memref<? x 128 x f32, #layout_map0>) -> f64

    // A function that returns its argument twice:
    func.func @count(%x: i64) -> (i64, i64)
      attributes {fruit: "banana"} {
      return %x, %x: i64, i64
    }

    // A function with an argument attribute
    func.func @example_fn_arg(%x: i32 {swift.self = unit})

    // A function with a result attribute
    func.func @example_fn_result() -> (f64 {dialectName.attrName = 0 : i64})

    // A function with an attribute
    func.func @example_fn_attr() attributes {dialectName.attrName = false}
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<StrAttr>:$sym_visibility,
                       OptionalAttr<ArrayAttr>:$arg_attrs,
                       OptionalAttr<ArrayAttr>:$res_attrs);
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
  >];
  let extraClassDeclaration = [{
    static mlir::func::FuncOp create(Location location, StringRef name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs = {});
    static mlir::func::FuncOp create(Location location, StringRef name, FunctionType type,
                         Operation::dialect_attr_range attrs);
    static mlir::func::FuncOp create(Location location, StringRef name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs,
                         ArrayRef<DictionaryAttr> argAttrs);

    /// Returns true if this function is external, i.e. it has no body.
    bool isExternal() { return getBody().empty(); }

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Allow the dialect prefix to be omitted.
    static StringRef getDefaultDialect() { return "offload"; }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration() { return isExternal(); }
  }];
  let hasCustomAssemblyFormat = 1;
}
#endif
