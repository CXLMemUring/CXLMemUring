#ifndef CIRA_RUNTIME_H
#define CIRA_RUNTIME_H

#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>

namespace cira {
namespace runtime {

// Forward declarations
class OffloadEngine;
class RemoteMemoryManager;
class PrefetchController;
class GraphRuntime;

// Memory tier enumeration
enum class MemoryTier {
    LOCAL_DRAM = 0,
    CXL_ATTACHED = 1,
    CXL_POOLED = 2,
    FAR_MEMORY = 3
};

// Remote memory reference
struct RemoteMemRef {
    void* base_addr;
    size_t size;
    MemoryTier tier;
    uint64_t metadata;
};

// Access profile for optimization
struct AccessProfile {
    enum Pattern { SEQUENTIAL, STRIDED, RANDOM };
    Pattern pattern;
    size_t stride;
    size_t working_set_size;
    double temporal_locality;
};

// Main runtime interface
class CiraRuntime {
public:
    static std::unique_ptr<CiraRuntime> create();

    virtual ~CiraRuntime() = default;

    virtual OffloadEngine* getOffloadEngine() = 0;
    virtual RemoteMemoryManager* getMemoryManager() = 0;
    virtual PrefetchController* getPrefetchController() = 0;
    virtual GraphRuntime* getGraphRuntime() = 0;

    // Runtime configuration
    virtual void configure(const std::string& config_file) = 0;
    virtual void setVerbosity(int level) = 0;
    virtual void enableProfiling(bool enable) = 0;
};

// Offload engine for remote memory operations
class OffloadEngine {
public:
    virtual ~OffloadEngine() = default;

    // Edge operations
    virtual void* loadEdge(RemoteMemRef* edge_ptr, size_t index,
                          size_t prefetch_distance = 0) = 0;
    virtual void evictEdge(RemoteMemRef* edge_ptr, size_t index) = 0;

    // Node operations
    virtual void* loadNode(void* edge_element, const char* field_name,
                          size_t prefetch_distance = 0) = 0;

    // Physical address operations
    virtual uintptr_t getPhysicalAddr(const char* field_name, void* node_data) = 0;

    // Batch operations for efficiency
    virtual void batchLoadEdges(RemoteMemRef* edge_ptr, size_t start_index,
                                size_t count, void** results) = 0;
};

// Remote memory management
class RemoteMemoryManager {
public:
    virtual ~RemoteMemoryManager() = default;

    // Allocation and deallocation
    virtual RemoteMemRef* allocateRemote(size_t size, MemoryTier tier) = 0;
    virtual void deallocateRemote(RemoteMemRef* ref) = 0;

    // Memory mapping
    virtual void* mapRemote(RemoteMemRef* ref) = 0;
    virtual void unmapRemote(void* addr) = 0;

    // Memory tier management
    virtual void migrate(RemoteMemRef* ref, MemoryTier new_tier) = 0;
    virtual MemoryTier getTier(RemoteMemRef* ref) = 0;

    // Statistics and monitoring
    virtual size_t getUsedMemory(MemoryTier tier) = 0;
    virtual size_t getAvailableMemory(MemoryTier tier) = 0;
};

// Prefetch controller for optimization
class PrefetchController {
public:
    virtual ~PrefetchController() = default;

    // Adaptive prefetching
    virtual void adaptivePrefetch(void* base_addr, size_t stride,
                                  size_t distance) = 0;

    // Batch prefetching
    virtual void batchPrefetch(void** addresses, size_t count) = 0;

    // Profile-guided optimization
    virtual void updateProfile(const AccessProfile& profile) = 0;
    virtual void optimizePrefetch(const AccessProfile& profile) = 0;

    // Control interface
    virtual void enablePrefetch(bool enable) = 0;
    virtual void setPrefetchDistance(size_t distance) = 0;
};

// Graph-specific runtime
class GraphRuntime {
public:
    struct Graph {
        RemoteMemRef* edges;
        RemoteMemRef* vertices;
        size_t num_edges;
        size_t num_vertices;
    };

    struct VertexProgram {
        void (*compute)(void* vertex_data, void* edge_data);
        void (*combine)(void* acc, void* value);
    };

    struct EdgeProgram {
        void (*scatter)(void* edge_data, void* src_vertex, void* dst_vertex);
        void (*gather)(void* dst_vertex, void* message);
    };

    virtual ~GraphRuntime() = default;

    // Graph initialization
    virtual void initializeGraph(Graph* graph, RemoteMemoryManager* mem_mgr) = 0;

    // Execution models
    virtual void executeVertexProgram(VertexProgram* program, Graph* graph) = 0;
    virtual void executeEdgeProgram(EdgeProgram* program, Graph* graph) = 0;

    // Optimized traversal patterns
    virtual void bfsTraversal(Graph* graph, size_t start_vertex,
                             void (*visit)(size_t vertex_id)) = 0;
    virtual void dfsTraversal(Graph* graph, size_t start_vertex,
                             void (*visit)(size_t vertex_id)) = 0;
};

// C API for FFI
extern "C" {
    void* cira_runtime_create();
    void cira_runtime_destroy(void* runtime);
    void* cira_offload_load_edge(void* engine, void* edge_ptr,
                                 size_t index, size_t prefetch_distance);
    void* cira_offload_load_node(void* engine, void* edge_element,
                                 const char* field_name, size_t prefetch_distance);
    uintptr_t cira_offload_get_paddr(void* engine, const char* field_name,
                                     void* node_data);
    void cira_offload_evict_edge(void* engine, void* edge_ptr, size_t index);
}

} // namespace runtime
} // namespace cira

#endif // CIRA_RUNTIME_H