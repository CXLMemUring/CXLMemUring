#ifndef CIRA_OPS_TD
#define CIRA_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
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

//===----------------------------------------------------------------------===//
// Memory Operations
//===----------------------------------------------------------------------===//

// Allocate memory in CXL space
def AllocCxlOp : RemoteMem_Op<"alloc_cxl", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocate memory in CXL-attached memory space";
  let description = [{
    Allocates memory in CXL-attached memory and returns a handle to it.
    The returned handle is guaranteed to point to CXL memory space.

    Example:
    ```mlir
    %ptr = cira.alloc_cxl %size : i64 -> !cira.handle<f32>
    ```
  }];

  let arguments = (ins Index:$size,
                       OptionalAttr<I64Attr>:$alignment);
  let results = (outs AnyCiraHandle:$result);

  let assemblyFormat = [{
    $size (`align` $alignment^)? attr-dict `:` type($result)
  }];
}

// Asynchronous load from CXL memory
def LoadAsyncOp : RemoteMem_Op<"load_async", [MemoryEffects<[MemRead]>]> {
  let summary = "Initiate asynchronous load from CXL memory";
  let description = [{
    Initiates an asynchronous load from CXL memory. Returns a future that
    can be awaited to get the actual value. This decouples data request
    from data consumption, enabling latency hiding.

    Example:
    ```mlir
    %fut = cira.load_async %ptr : !cira.handle<i64> -> !cira.future<i64>
    // ... independent computation ...
    %val = cira.future_await %fut : !cira.future<i64> -> i64
    ```
  }];

  let arguments = (ins AnyCiraHandle:$ptr,
                       Optional<Index>:$index);
  let results = (outs AnyCiraFuture:$future);

  let assemblyFormat = [{
    $ptr (`[` $index^ `]`)? attr-dict `:` type($ptr) `->` type($future)
  }];
}

// Asynchronous store to CXL memory
def StoreAsyncOp : RemoteMem_Op<"store_async", [MemoryEffects<[MemWrite]>]> {
  let summary = "Initiate asynchronous store to CXL memory";
  let description = [{
    Initiates an asynchronous store to CXL memory. Returns a future that
    can be awaited to ensure the store has completed.

    Example:
    ```mlir
    %fut = cira.store_async %val, %ptr : i64, !cira.handle<i64> -> !cira.future<none>
    ```
  }];

  let arguments = (ins AnyType:$value,
                       AnyCiraHandle:$ptr,
                       Optional<Index>:$index);
  let results = (outs AnyCiraFuture:$future);

  let assemblyFormat = [{
    $value `,` $ptr (`[` $index^ `]`)? attr-dict `:` type($value) `,` type($ptr) `->` type($future)
  }];
}

// Flush pending writes to CXL memory
def FlushOp : RemoteMem_Op<"flush", []> {
  let summary = "Flush pending writes to CXL memory";
  let description = [{
    Ensures all pending asynchronous writes to CXL memory have completed.
    Acts as a memory barrier for CXL operations.

    Example:
    ```mlir
    cira.flush
    ```
  }];

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Cache Management Operations
//===----------------------------------------------------------------------===//

// Install cache line
def InstallCachelineOp : RemoteMem_Op<"install_cacheline", []> {
  let summary = "Install data into local cache hierarchy";
  let description = [{
    Explicitly installs data from CXL memory into the local cache hierarchy.
    This is used when the compiler knows data will be accessed repeatedly
    and wants to ensure it resides in fast cache.

    Example:
    ```mlir
    cira.install_cacheline %ptr, %size : !cira.handle<i64>, i64
    ```
  }];

  let arguments = (ins AnyCiraHandle:$ptr,
                       Index:$size,
                       OptionalAttr<I64Attr>:$cache_level);

  let assemblyFormat = [{
    $ptr `,` $size (`level` `=` $cache_level^)? attr-dict `:` type($ptr) `,` type($size)
  }];
}

// Eviction hint
def EvictHintOp : RemoteMem_Op<"evict_hint", []> {
  let summary = "Provide eviction hint for cache line";
  let description = [{
    Provides a hint to the cache system that the specified data is no longer
    needed and can be evicted to make room for other data. This is non-binding.

    Example:
    ```mlir
    cira.evict_hint %ptr : !cira.handle<i64>
    ```
  }];

  let arguments = (ins AnyCiraHandle:$ptr,
                       Optional<Index>:$size);

  let assemblyFormat = [{
    $ptr (`,` $size^)? attr-dict `:` type($ptr)
  }];
}

// Prefetch stream (sequential/strided pattern)
def PrefetchStreamOp : RemoteMem_Op<"prefetch_stream", []> {
  let summary = "Initiate prefetching for sequential/strided access pattern";
  let description = [{
    Sets up prefetching for a sequential or strided memory access pattern.
    The hardware/runtime will prefetch data ahead of consumption.

    Example:
    ```mlir
    cira.prefetch_stream %ptr, %count, stride=%stride : !cira.handle<f32>, i64
    ```
  }];

  let arguments = (ins AnyCiraHandle:$ptr,
                       Index:$count,
                       Optional<Index>:$stride,
                       OptionalAttr<I64Attr>:$distance);

  let assemblyFormat = [{
    $ptr `,` $count (`,` `stride` `=` $stride^)? (`distance` `=` $distance^)? attr-dict `:` type($ptr) `,` type($count)
  }];
}

// Prefetch indirect (pointer chasing pattern)
def PrefetchIndirectOp : RemoteMem_Op<"prefetch_indirect", []> {
  let summary = "Initiate prefetching for pointer-chasing access pattern";
  let description = [{
    Sets up prefetching for an indirect/pointer-chasing memory access pattern.
    This is used for linked lists, trees, graphs where the next address depends
    on loading the current element. The accelerator will chase pointers ahead.

    Example:
    ```mlir
    cira.prefetch_indirect %start_node, offset=8, depth=16 : !cira.handle<node>
    ```
  }];

  let arguments = (ins AnyCiraHandle:$start_ptr,
                       I64Attr:$next_ptr_offset,
                       I64Attr:$depth);

  let assemblyFormat = [{
    $start_ptr `,` `offset` `=` $next_ptr_offset `,` `depth` `=` $depth attr-dict `:` type($start_ptr)
  }];
}

//===----------------------------------------------------------------------===//
// Synchronization Operations
//===----------------------------------------------------------------------===//

// Create a future for tracking async operations
def FutureCreateOp : RemoteMem_Op<"future_create", [Pure]> {
  let summary = "Create a future for tracking asynchronous operations";
  let description = [{
    Creates an empty future that can be used to track completion of
    asynchronous operations.

    Example:
    ```mlir
    %fut = cira.future_create : !cira.future<i64>
    ```
  }];

  let results = (outs AnyCiraFuture:$future);

  let assemblyFormat = [{
    attr-dict `:` type($future)
  }];
}

// Await a future
def FutureAwaitOp : RemoteMem_Op<"future_await", []> {
  let summary = "Await completion of an asynchronous operation";
  let description = [{
    Blocks until the asynchronous operation represented by the future
    completes, then returns the result value.

    Example:
    ```mlir
    %val = cira.future_await %fut : !cira.future<i64> -> i64
    ```
  }];

  let arguments = (ins AnyCiraFuture:$future);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $future attr-dict `:` type($future) `->` type($result)
  }];
}

// Barrier synchronization
def BarrierOp : RemoteMem_Op<"barrier", []> {
  let summary = "Synchronization barrier for CXL operations";
  let description = [{
    Ensures all pending CXL operations (loads, stores, prefetches) have
    completed before proceeding. This is a full memory barrier.

    Example:
    ```mlir
    cira.barrier
    ```
  }];

  let assemblyFormat = "attr-dict";
}

// Release resources
def ReleaseOp : RemoteMem_Op<"release", []> {
  let summary = "Release CXL memory resources";
  let description = [{
    Releases the specified CXL memory allocation. After this operation,
    the handle is no longer valid.

    Example:
    ```mlir
    cira.release %ptr : !cira.handle<i64>
    ```
  }];

  let arguments = (ins AnyCiraHandle:$ptr);

  let assemblyFormat = [{
    $ptr attr-dict `:` type($ptr)
  }];
}

//===----------------------------------------------------------------------===//
// Control Operations
//===----------------------------------------------------------------------===//

// Offload region to near-memory accelerator
def OffloadRegionOp : RemoteMem_Op<"offload", [SingleBlock, NoTerminator,
                                               IsolatedFromAbove]> {
  let summary = "Offload a region of computation to near-memory accelerator";
  let description = [{
    Marks a region of computation to be offloaded to the near-memory
    accelerator (Vortex cores). The region should contain memory operations
    that benefit from near-memory execution.

    Example:
    ```mlir
    cira.offload @vortex_core_0 {
      cira.prefetch_chain %stream, depth=16
    }
    ```
  }];

  let arguments = (ins OptionalAttr<SymbolRefAttr>:$target,
                       Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    ($target^)? (`(` $operands^ `)`)? attr-dict-with-keyword (`:` functional-type($operands, $results)^)? $body
  }];
}

// Speculative execution region
def SpeculateOp : RemoteMem_Op<"speculate", [SingleBlock, NoTerminator]> {
  let summary = "Mark a region for speculative execution";
  let description = [{
    Marks a region of computation as speculative. The operations inside
    may be executed speculatively by the accelerator. If speculation
    is incorrect, results are discarded.

    Example:
    ```mlir
    %result = cira.speculate {
      %val = cira.load_async %predicted_ptr : !cira.handle<i64> -> !cira.future<i64>
      cira.yield %val : !cira.future<i64>
    } : !cira.future<i64>
    ```
  }];

  let arguments = (ins OptionalAttr<F64Attr>:$confidence);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    (`confidence` `=` $confidence^)? attr-dict-with-keyword `:` type($results) $body
  }];
}

// Phase boundary marker
def PhaseBoundaryOp : RemoteMem_Op<"phase_boundary", []> {
  let summary = "Mark a phase boundary in the computation";
  let description = [{
    Marks a boundary between computation phases. This helps the compiler
    and runtime optimize data placement and prefetching strategies.
    At phase boundaries, the system may reorganize data or adjust
    prefetching parameters.

    Example:
    ```mlir
    cira.phase_boundary "scatter_to_gather"
    ```
  }];

  let arguments = (ins OptionalAttr<StrAttr>:$phase_name);

  let assemblyFormat = [{
    ($phase_name^)? attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Stream Operations (for pointer chasing / linked list patterns)
//===----------------------------------------------------------------------===//

// Create a stream for indirect access patterns
def StreamCreateIndirectOp : RemoteMem_Op<"stream_create_indirect", [Pure]> {
  let summary = "Create a stream descriptor for pointer-chasing patterns";
  let description = [{
    Creates a stream descriptor for indirect/pointer-chasing access patterns.
    The stream tracks the current position and enables the accelerator to
    prefetch ahead by following pointers.

    Example:
    ```mlir
    %stream = cira.stream_create_indirect %start_node, offset=8 : !cira.stream<node>
    ```
  }];

  let arguments = (ins AnyCiraHandle:$start_ptr,
                       I64Attr:$next_ptr_offset);
  let results = (outs AnyCiraStream:$stream);

  let assemblyFormat = [{
    $start_ptr `,` `offset` `=` $next_ptr_offset attr-dict `:` type($start_ptr) `->` type($stream)
  }];
}

// Start offloaded prefetch chain on stream
def OffloadStartOp : RemoteMem_Op<"offload_start", []> {
  let summary = "Start offloaded prefetching on a stream";
  let description = [{
    Initiates offloaded prefetching on the given stream. The accelerator
    will chase pointers to the specified depth.

    Example:
    ```mlir
    cira.offload_start %vortex_core {
      cira.prefetch_chain %stream, depth=16
    }
    ```
  }];

  let arguments = (ins OptionalAttr<SymbolRefAttr>:$target);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    ($target^)? attr-dict-with-keyword $body
  }];
}

// Prefetch chain operation (runs on accelerator)
def PrefetchChainOp : RemoteMem_Op<"prefetch_chain", []> {
  let summary = "Chase pointer chain and prefetch to LLC";
  let description = [{
    Chases the pointer chain in the stream to the specified depth,
    prefetching each element into the LLC. This operation typically
    runs on the near-memory accelerator.

    Example:
    ```mlir
    cira.prefetch_chain %stream, depth=16 : !cira.stream<node>
    ```
  }];

  let arguments = (ins AnyCiraStream:$stream,
                       I64Attr:$depth);

  let assemblyFormat = [{
    $stream `,` `depth` `=` $depth attr-dict `:` type($stream)
  }];
}

// Peek at next element in stream (returns future)
def PeekStreamOp : RemoteMem_Op<"peek_stream", [Pure]> {
  let summary = "Peek at the next element in the stream";
  let description = [{
    Returns a future representing the next element in the stream.
    The element should already be prefetched into the LLC.

    Example:
    ```mlir
    %future = cira.peek_stream %stream : !cira.stream<node> -> !cira.future<node>
    ```
  }];

  let arguments = (ins AnyCiraStream:$stream);
  let results = (outs AnyCiraFuture:$future);

  let assemblyFormat = [{
    $stream attr-dict `:` type($stream) `->` type($future)
  }];
}

// Advance stream position
def AdvanceStreamOp : RemoteMem_Op<"advance_stream", []> {
  let summary = "Advance the stream to the next element";
  let description = [{
    Advances the stream position to the next element. This typically
    follows a peek_stream + await sequence.

    Example:
    ```mlir
    cira.advance_stream %stream : !cira.stream<node>
    ```
  }];

  let arguments = (ins AnyCiraStream:$stream);

  let assemblyFormat = [{
    $stream attr-dict `:` type($stream)
  }];
}

// Yield from speculative region
def YieldOp : RemoteMem_Op<"yield", [Pure, Terminator,
                                     ParentOneOf<["SpeculateOp", "OffloadRegionOp"]>]> {
  let summary = "Yield values from a region";
  let description = [{
    Yields values from a speculative or offload region.
  }];

  let arguments = (ins Variadic<AnyType>:$values);

  let assemblyFormat = [{
    ($values^ `:` type($values))? attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Legacy Operations (for backward compatibility)
//===----------------------------------------------------------------------===//

// Load edge operation (legacy)
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

// Load node operation (legacy)
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

// Get physical address operation (legacy)
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

// Evict edge operation (legacy)
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

// Generic offload operation wrapper (legacy)
def OffloadOp : RemoteMem_Op<"offload_legacy", [SingleBlock, NoTerminator]> {
  let summary = "Generic offload operation (legacy)";
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