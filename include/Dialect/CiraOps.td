#ifndef CIRA_OPS_TD
#define CIRA_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "Dialect/RemoteMem.td"

//===----------------------------------------------------------------------===//
// Cira dialect operation definitions
//===----------------------------------------------------------------------===//

// Base class for Cira operations that take an offload command
class OffloadOpBase<string mnemonic, list<Trait> traits = []> :
    RemoteMem_Op<"offload." # mnemonic, traits> {
  let summary = "Offload " # mnemonic # " operation";
}

// Load edge operation
def LoadEdgeOp : OffloadOpBase<"load_edge", [Pure]> {
  let summary = "Load edge data from remote memory";
  let description = [{
    Loads edge data from remote memory. Takes a remotable struct pointer,
    an index, and an optional prefetch distance.
  }];

  let arguments = (ins AnyRemoteMemRef:$edge_ptr,
                       Index:$index,
                       Optional<Index>:$prefetch_distance);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    `(` $edge_ptr `,` $index (`,` $prefetch_distance^)? `)` attr-dict `:` type($edge_ptr) `->` type($result)
  }];
}

// Load node operation  
def LoadNodeOp : OffloadOpBase<"load_node", [Pure]> {
  let summary = "Load node data from remote memory";
  let description = [{
    Loads node data from remote memory. Takes an edge element,
    a field name (from/to), and an optional prefetch distance.
  }];

  let arguments = (ins AnyType:$edge_element,
                       StrAttr:$field_name,
                       Optional<Index>:$prefetch_distance);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    `(` $edge_element `,` $field_name (`,` $prefetch_distance^)? `)` attr-dict `:` type($edge_element) `->` type($result)
  }];
}

// Get physical address operation
def GetPaddrOp : RemoteMem_Op<"offload.get_paddr", [Pure]> {
  let summary = "Get physical address from offloaded data";
  let description = [{
    Obtains the physical address of previously offloaded node data.
  }];

  let arguments = (ins StrAttr:$field_name,
                       AnyType:$node_data);
  let results = (outs Index:$paddr);

  let assemblyFormat = [{
    `(` $field_name `,` $node_data `)` attr-dict `:` type($node_data) `->` type($paddr)
  }];
}

// Evict edge operation
def EvictEdgeOp : OffloadOpBase<"evict_edge", []> {
  let summary = "Evict edge data from cache";
  let description = [{
    Provides an eviction hint for processed edge cache line.
  }];

  let arguments = (ins AnyRemoteMemRef:$edge_ptr,
                       Index:$index);

  let assemblyFormat = [{
    `(` $edge_ptr `,` $index `)` attr-dict `:` type($edge_ptr)
  }];
}

// Generic offload operation wrapper
def OffloadOp : RemoteMem_Op<"offload", [SingleBlock, NoTerminator]> {
  let summary = "Generic offload operation";
  let description = [{
    Generic offload operation that can contain different offload commands.
    This is used for operations that don't fit the specific patterns above.
  }];

  let arguments = (ins SymbolNameAttr:$command,
                       Variadic<AnyType>:$operands);
  let results = (outs Optional<AnyType>:$result);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $command `(` $operands `)` attr-dict `:` functional-type($operands, $result) $body
  }];
}

// Call operation
def CallOp : RemoteMem_Op<"call", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Call a function with physical addresses";
  let description = [{
    Calls a function with the given arguments. This is typically used
    to call update functions that work on node data using physical addresses.
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee,
                       Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let builders = [
    OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
    }]>,
    OpBuilder<(ins "StringAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, SymbolRefAttr::get(callee), results, operands);
    }]>,
    OpBuilder<(ins "StringRef":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, StringAttr::get($_builder.getContext(), callee),
            results, operands);
    }]>
  ];

  let extraClassDeclaration = [{
    FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

#endif // CIRA_OPS_TD