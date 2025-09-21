# Cira Runtime System Design

## Overview
The Cira runtime provides the execution engine for offloaded graph processing operations on CXL memory systems.

## Architecture Components

### 1. Offload Engine
Manages hardware offload operations for remote memory access.

```cpp
class OffloadEngine {
public:
    // Prefetch edge data from remote memory
    void* loadEdge(RemoteMemRef* edge_ptr, size_t index, size_t prefetch_distance);

    // Prefetch node data based on edge element
    void* loadNode(void* edge_element, const char* field_name, size_t prefetch_distance);

    // Get physical address of offloaded data
    uintptr_t getPhysicalAddr(const char* field_name, void* node_data);

    // Evict edge from cache
    void evictEdge(RemoteMemRef* edge_ptr, size_t index);
};
```

### 2. Memory Management
Handles allocation and management of remote memory regions.

```cpp
class RemoteMemoryManager {
    // Allocate remote memory region
    RemoteMemRef* allocateRemote(size_t size, MemoryTier tier);

    // Map remote memory to local address space
    void* mapRemote(RemoteMemRef* ref);

    // Manage memory tiers (local DRAM, CXL, far memory)
    void setMemoryTier(RemoteMemRef* ref, MemoryTier tier);
};
```

### 3. Prefetch Controller
Optimizes data prefetching based on access patterns.

```cpp
class PrefetchController {
    // Adaptive prefetch based on access pattern
    void adaptivePrefetch(void* base_addr, size_t stride, size_t distance);

    // Batch prefetch for multiple elements
    void batchPrefetch(void** addresses, size_t count);

    // Profile-guided prefetch optimization
    void profileGuidedPrefetch(AccessProfile* profile);
};
```

### 4. Graph Processing Runtime
Specialized runtime for graph algorithms.

```cpp
class GraphRuntime {
    // Initialize graph structure in remote memory
    void initializeGraph(Graph* graph, RemoteMemoryManager* mem_mgr);

    // Execute vertex-centric computation
    void executeVertexProgram(VertexProgram* program, Graph* graph);

    // Execute edge-centric computation
    void executeEdgeProgram(EdgeProgram* program, Graph* graph);
};
```

## Lowering Strategy

### MLIR to Runtime Mapping

1. **cira.offload.load_edge** → `OffloadEngine::loadEdge()`
2. **cira.offload.load_node** → `OffloadEngine::loadNode()`
3. **cira.offload.get_paddr** → `OffloadEngine::getPhysicalAddr()`
4. **cira.offload.evict_edge** → `OffloadEngine::evictEdge()`
5. **cira.call** → Direct function call with physical addresses

### Optimization Passes

1. **Prefetch Distance Analysis**: Determine optimal prefetch distance
2. **Access Pattern Recognition**: Identify strided vs random access
3. **Memory Tier Placement**: Decide data placement across memory tiers
4. **Batch Coalescing**: Combine multiple prefetch operations

## Implementation Phases

### Phase 1: Basic Runtime (Current)
- Simple offload operations
- Manual prefetch control
- Direct memory mapping

### Phase 2: Optimization Layer
- Adaptive prefetching
- Access pattern analysis
- Dynamic tier management

### Phase 3: Hardware Integration
- CXL controller integration
- DMA engine support
- Hardware prefetcher coordination

### Phase 4: Advanced Features
- Multi-node support
- Coherence management
- Fault tolerance

## Usage Example

```cpp
// Runtime initialization
auto runtime = CiraRuntime::create();
auto offload_engine = runtime->getOffloadEngine();
auto mem_manager = runtime->getMemoryManager();

// Allocate graph in remote memory
auto edge_data = mem_manager->allocateRemote(
    sizeof(Edge) * num_edges,
    MemoryTier::CXL_MEMORY
);

// Execute graph traversal
for (size_t i = 0; i < num_edges; i += cache_line_size) {
    // Prefetch next cache line
    offload_engine->loadEdge(edge_data, i, prefetch_distance);

    for (size_t j = 0; j < cache_line_size; j++) {
        auto edge = offload_engine->loadEdge(edge_data, i + j, 0);

        // Prefetch node data
        auto node_from = offload_engine->loadNode(edge, "from", 1);
        auto node_to = offload_engine->loadNode(edge, "to", 1);

        // Get physical addresses for computation
        auto paddr_from = offload_engine->getPhysicalAddr("from", node_from);
        auto paddr_to = offload_engine->getPhysicalAddr("to", node_to);

        // Execute update function
        update_node(edge, paddr_from, paddr_to);
    }

    // Evict processed cache line
    offload_engine->evictEdge(edge_data, i);
}
```