# Cira Dialect for MLIR

This document describes the new Cira dialect added to the MLIR framework for supporting graph traversal operations with offloading capabilities.

## Overview

The Cira dialect provides operations for efficient graph processing with support for:
- Remote memory access
- Prefetching and cache management
- Physical address management
- Offloading operations

## Operations

### `cira.offload.load_edge`

Loads edge data from remote memory with optional prefetching.

```mlir
%edge = cira.offload.load_edge(%edge_ptr, %index, %prefetch_distance) : !edge_type
```

- `%edge_ptr`: Remotable pointer to edge array
- `%index`: Index of the edge to load
- `%prefetch_distance`: Optional prefetch distance for cache optimization

### `cira.offload.load_node`

Loads node data based on edge element fields.

```mlir
%node = cira.offload.load_node(%edge_element, "field_name", %prefetch_distance) : !node_type
```

- `%edge_element`: Previously loaded edge element
- `"field_name"`: String attribute ("from" or "to")
- `%prefetch_distance`: Optional prefetch distance

### `cira.offload.get_paddr`

Obtains physical address from offloaded data.

```mlir
%paddr = cira.offload.get_paddr("field_name", %node_data) : !llvm.ptr
```

- `"field_name"`: Field identifier
- `%node_data`: Node data from previous load

### `cira.offload.evict_edge`

Provides cache eviction hints for processed data.

```mlir
cira.offload.evict_edge(%edge_ptr, %index)
```

### `cira.call`

Calls functions with physical addresses.

```mlir
cira.call @function_name(%arg1, %arg2, ...) : (type1, type2, ...) -> (result_types...)
```

## Example Usage

```mlir
func.func @trvs_graph_opt(%arg0: !remotable<struct<edge>>) {
    scf.for %i = %0 to %num_edges step %elements_per_line {
        // Offload prefetching of a cache line from far memory
        cira.offload.load_edge(%arg0, %i, %n_ahead) : !edge_type
        
        scf.for %j = %0 to %elements_per_line {
            // Load the edge element
            %edge_element = cira.offload.load_edge(%arg0, %i) : !edge_type
            
            // Offload node prefetching
            %node_from = cira.offload.load_node(%edge_element, "from", %n_ahead_node) : !node_type
            %node_to = cira.offload.load_node(%edge_element, "to", %n_ahead_node) : !node_type
            
            // Get physical addresses
            %paddr_from = cira.offload.get_paddr("from", %node_from) : !llvm.ptr
            %paddr_to = cira.offload.get_paddr("to", %node_to) : !llvm.ptr
            
            // Call update function
            cira.call @update_node(%edge_element, %paddr_from, %paddr_to)
        }
        
        // Evict processed cache line
        cira.offload.evict_edge(%arg0, %i)
    }
}
```

## Conversion to LLVM

The dialect includes a conversion pass to lower Cira operations to LLVM:

```bash
polygeist-opt --convert-cira-to-llvm input.mlir -o output.mlir
```

The conversion handles:
- Load operations → LLVM GEP and load instructions
- Node field extraction → LLVM extractvalue operations
- Call operations → LLVM call instructions
- Eviction hints → Platform-specific cache management (or removed if not supported)

## Implementation Details

### Files Added/Modified

1. **TableGen Definitions**:
   - `include/Dialect/CiraOps.td` - Operation definitions
   - `include/Dialect/CiraOps.h` - C++ header for operations

2. **Implementation**:
   - `src/Dialect/CiraOps.cpp` - Operation implementations
   - `src/Conversion/CiraToLLVM.cpp` - Conversion patterns

3. **Build System**:
   - Updated `include/Dialect/CMakeLists.txt`
   - Updated `src/Dialect/CMakeLists.txt`
   - Updated `src/Conversion/CMakeLists.txt`

4. **Pass Registration**:
   - Added to `include/Conversion/Passes.td`
   - Updated `include/Conversion/Passes.h`

## Building

```bash
./build_cira.sh
```

Or manually:

```bash
mkdir build
cd build
cmake .. -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir
make -j$(nproc)
```

## Testing

Run the example test:

```bash
./build/tools/polygeist-opt/polygeist-opt test/cira-graph-example.mlir --convert-cira-to-llvm
```

## Future Work

- Implement actual runtime support for remote memory operations
- Add more sophisticated prefetching strategies
- Support for different cache architectures
- Integration with hardware offload engines