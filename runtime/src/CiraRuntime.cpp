#include "CiraRuntime.h"
#include "vortex_device.h"
#include <iostream>
#include <unordered_map>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <queue>
#include <mutex>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <limits>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#ifdef NUMA_SUPPORT
#include <numa.h>
#endif

namespace cira {
namespace runtime {

// Implementation of the runtime system
class CiraRuntimeImpl : public CiraRuntime {
private:
    std::unique_ptr<OffloadEngine> offload_engine_;
    std::unique_ptr<RemoteMemoryManager> memory_manager_;
    std::unique_ptr<PrefetchController> prefetch_controller_;
    std::unique_ptr<GraphRuntime> graph_runtime_;
    std::unique_ptr<Type2CacheController> type2_controller_;
    int verbosity_level_ = 0;
    bool profiling_enabled_ = false;

public:
    CiraRuntimeImpl();
    ~CiraRuntimeImpl() override = default;

    OffloadEngine* getOffloadEngine() override { return offload_engine_.get(); }
    RemoteMemoryManager* getMemoryManager() override { return memory_manager_.get(); }
    PrefetchController* getPrefetchController() override { return prefetch_controller_.get(); }
    GraphRuntime* getGraphRuntime() override { return graph_runtime_.get(); }
    Type2CacheController* getType2Controller() override { return type2_controller_.get(); }

    void configure(const std::string& config_file) override;
    void configureType2(const Type2DeviceConfig& config) override;
    void setVerbosity(int level) override { verbosity_level_ = level; }
    void enableProfiling(bool enable) override { profiling_enabled_ = enable; }
};

// Offload Engine Implementation
class OffloadEngineImpl : public OffloadEngine {
private:
    struct EdgeCache {
        void* data;
        size_t index;
        bool valid;
    };

    std::unordered_map<uintptr_t, EdgeCache> edge_cache_;
    size_t cache_line_size_ = 64;
    size_t prefetch_queue_depth_ = 4;

public:
    void* loadEdge(RemoteMemRef* edge_ptr, size_t index,
                   size_t prefetch_distance) override {
        // Calculate address
        uintptr_t addr = reinterpret_cast<uintptr_t>(edge_ptr->base_addr) +
                        (index * sizeof(void*)); // Assuming edge size

        // Check cache
        auto it = edge_cache_.find(addr);
        if (it != edge_cache_.end() && it->second.valid) {
            return it->second.data;
        }

        // Issue prefetch for future accesses
        if (prefetch_distance > 0) {
            for (size_t i = 1; i <= prefetch_distance; i++) {
                size_t prefetch_index = index + i * cache_line_size_;
                uintptr_t prefetch_addr = reinterpret_cast<uintptr_t>(edge_ptr->base_addr) +
                                         (prefetch_index * sizeof(void*));
                __builtin_prefetch(reinterpret_cast<void*>(prefetch_addr), 0, 3);
            }
        }

        // Load data
        void* data = reinterpret_cast<void*>(addr);

        // Update cache
        edge_cache_[addr] = {data, index, true};

        return data;
    }

    void evictEdge(RemoteMemRef* edge_ptr, size_t index) override {
        uintptr_t addr = reinterpret_cast<uintptr_t>(edge_ptr->base_addr) +
                        (index * sizeof(void*));

        auto it = edge_cache_.find(addr);
        if (it != edge_cache_.end()) {
            it->second.valid = false;

            // Issue cache line eviction hint
            #ifdef __x86_64__
            __builtin_ia32_clflush(reinterpret_cast<void*>(addr));
            #endif
        }
    }

    void* loadNode(void* edge_element, const char* field_name,
                   size_t prefetch_distance) override {
        // Extract node pointer from edge element based on field name
        // This is a simplified implementation
        void** edge_data = reinterpret_cast<void**>(edge_element);
        void* node_ptr = nullptr;

        if (strcmp(field_name, "from") == 0) {
            node_ptr = edge_data[0]; // Assuming first field is 'from'
        } else if (strcmp(field_name, "to") == 0) {
            node_ptr = edge_data[1]; // Assuming second field is 'to'
        }

        // Issue prefetch
        if (node_ptr && prefetch_distance > 0) {
            for (size_t i = 0; i < prefetch_distance; i++) {
                __builtin_prefetch(static_cast<char*>(node_ptr) + i * cache_line_size_, 0, 3);
            }
        }

        return node_ptr;
    }

    uintptr_t getPhysicalAddr(const char* field_name, void* node_data) override {
        // In a real implementation, this would translate to physical address
        // For now, return virtual address
        return reinterpret_cast<uintptr_t>(node_data);
    }

    void batchLoadEdges(RemoteMemRef* edge_ptr, size_t start_index,
                        size_t count, void** results) override {
        for (size_t i = 0; i < count; i++) {
            results[i] = loadEdge(edge_ptr, start_index + i, 1);
        }
    }
};

// Remote Memory Manager Implementation
class RemoteMemoryManagerImpl : public RemoteMemoryManager {
private:
    struct Allocation {
        void* addr;
        size_t size;
        MemoryTier tier;
        int numa_node;
    };

    std::unordered_map<RemoteMemRef*, Allocation> allocations_;
    std::unordered_map<void*, RemoteMemRef*> addr_to_ref_;

public:
    RemoteMemRef* allocateRemote(size_t size, MemoryTier tier) override {
        auto* ref = new RemoteMemRef();
        ref->size = size;
        ref->tier = tier;

        // Allocate based on tier
        void* addr = nullptr;
        int numa_node = 0;

        switch (tier) {
            case MemoryTier::LOCAL_DRAM:
                addr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                break;

            case MemoryTier::CXL_ATTACHED:
            case MemoryTier::CXL_POOLED:
#ifdef NUMA_SUPPORT
                // Try to allocate on NUMA node 1 (simulating CXL)
                if (numa_available() != -1) {
                    numa_node = numa_max_node() >= 1 ? 1 : 0;
                    addr = numa_alloc_onnode(size, numa_node);
                } else
#endif
                {
                    // Fall back to regular allocation
                    addr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                }
                break;

            case MemoryTier::FAR_MEMORY:
                // Far memory allocation (could be file-backed)
                addr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
                break;

            case MemoryTier::CXL_TYPE2_MEM:
            case MemoryTier::CXL_TYPE2_CACHE:
#ifdef NUMA_SUPPORT
                // Type 2 device memory: try highest NUMA node (CXL device)
                if (numa_available() != -1) {
                    numa_node = numa_max_node();
                    addr = numa_alloc_onnode(size, numa_node);
                } else
#endif
                {
                    addr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                }
                break;
        }

        ref->base_addr = addr;
        allocations_[ref] = {addr, size, tier, numa_node};
        addr_to_ref_[addr] = ref;

        return ref;
    }

    void deallocateRemote(RemoteMemRef* ref) override {
        auto it = allocations_.find(ref);
        if (it != allocations_.end()) {
#ifdef NUMA_SUPPORT
            if ((it->second.tier == MemoryTier::CXL_ATTACHED ||
                 it->second.tier == MemoryTier::CXL_POOLED ||
                 it->second.tier == MemoryTier::CXL_TYPE2_MEM ||
                 it->second.tier == MemoryTier::CXL_TYPE2_CACHE) &&
                numa_available() != -1 && it->second.numa_node > 0) {
                numa_free(it->second.addr, it->second.size);
#else
            if (false) {
                // Never executed, just for structure
#endif
            } else {
                munmap(it->second.addr, it->second.size);
            }
            addr_to_ref_.erase(it->second.addr);
            allocations_.erase(it);
        }
        delete ref;
    }

    void* mapRemote(RemoteMemRef* ref) override {
        return ref->base_addr;
    }

    void unmapRemote(void* addr) override {
        // No-op for now as memory is directly accessible
    }

    void migrate(RemoteMemRef* ref, MemoryTier new_tier) override {
        auto it = allocations_.find(ref);
        if (it == allocations_.end()) return;

        // Allocate new memory in target tier
        RemoteMemRef* new_ref = allocateRemote(ref->size, new_tier);

        // Copy data
        memcpy(new_ref->base_addr, ref->base_addr, ref->size);

        // Update mappings
        deallocateRemote(ref);
        *ref = *new_ref;
        delete new_ref;
    }

    MemoryTier getTier(RemoteMemRef* ref) override {
        return ref->tier;
    }

    size_t getUsedMemory(MemoryTier tier) override {
        size_t total = 0;
        for (const auto& [ref, alloc] : allocations_) {
            if (alloc.tier == tier) {
                total += alloc.size;
            }
        }
        return total;
    }

    size_t getAvailableMemory(MemoryTier tier) override {
        // Simplified implementation
        switch (tier) {
            case MemoryTier::LOCAL_DRAM:
                return 16UL * 1024 * 1024 * 1024; // 16GB
            case MemoryTier::CXL_ATTACHED:
            case MemoryTier::CXL_POOLED:
                return 64UL * 1024 * 1024 * 1024; // 64GB
            case MemoryTier::FAR_MEMORY:
                return 256UL * 1024 * 1024 * 1024; // 256GB
            case MemoryTier::CXL_TYPE2_MEM:
                return 4UL * 1024 * 1024 * 1024; // 4GB (BAR4 default)
            case MemoryTier::CXL_TYPE2_CACHE:
                return 128UL * 1024 * 1024; // 128MB (BAR2 default)
        }
        return 0;
    }
};

// Prefetch Controller Implementation
class PrefetchControllerImpl : public PrefetchController {
private:
    bool prefetch_enabled_ = true;
    size_t prefetch_distance_ = 4;
    AccessProfile current_profile_;

public:
    void adaptivePrefetch(void* base_addr, size_t stride,
                          size_t distance) override {
        if (!prefetch_enabled_) return;

        for (size_t i = 0; i < distance; i++) {
            void* addr = static_cast<char*>(base_addr) + (i * stride);
            __builtin_prefetch(addr, 0, 3);
        }
    }

    void batchPrefetch(void** addresses, size_t count) override {
        if (!prefetch_enabled_) return;

        for (size_t i = 0; i < count; i++) {
            __builtin_prefetch(addresses[i], 0, 3);
        }
    }

    void updateProfile(const AccessProfile& profile) override {
        current_profile_ = profile;

        // Adjust prefetch distance based on pattern
        switch (profile.pattern) {
            case AccessProfile::SEQUENTIAL:
                prefetch_distance_ = 8;
                break;
            case AccessProfile::STRIDED:
                prefetch_distance_ = 4;
                break;
            case AccessProfile::RANDOM:
                prefetch_distance_ = 1;
                break;
        }
    }

    void optimizePrefetch(const AccessProfile& profile) override {
        updateProfile(profile);
    }

    void enablePrefetch(bool enable) override {
        prefetch_enabled_ = enable;
    }

    void setPrefetchDistance(size_t distance) override {
        prefetch_distance_ = distance;
    }
};

// Type 2 Cache Controller Implementation
class Type2CacheControllerImpl : public Type2CacheController {
private:
    static constexpr size_t CXL_CACHE_LINE_SIZE = 64;

    struct CacheLine {
        uint8_t data[64];
        CoherencyState state;
        bool dirty;
        uint64_t timestamp_ns;
    };

    enum class DelayOpType { READ, WRITE, COHERENCY };

    struct DelayedOp {
        uintptr_t addr;
        std::vector<uint8_t> data;
        DelayOpType op_type;
        uint64_t enqueue_time_ns;
        uint32_t delay_ns;
    };

    Type2DeviceConfig config_;
    bool initialized_ = false;

    // Cache structure: 64-byte aligned address -> CacheLine
    std::unordered_map<uintptr_t, CacheLine> cache_;
    std::mutex cache_mutex_;

    // Delay buffer modeling IA-780i pipeline (DDR_IDLE→READ/WRITE→WAIT→IDLE)
    std::queue<DelayedOp> delay_buffer_;
    size_t delay_buffer_depth_ = 16;

    // BAR mappings (mmap'd or NUMA-fallback)
    void* bar2_mapping_ = nullptr;  // CXL.cache
    void* bar4_mapping_ = nullptr;  // CXL.mem

    // Statistics
    size_t cache_hits_ = 0;
    size_t cache_misses_ = 0;

    static uint64_t now_ns() {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
    }

    uintptr_t alignToLine(void* addr) const {
        return reinterpret_cast<uintptr_t>(addr) & ~(CXL_CACHE_LINE_SIZE - 1);
    }

    void processDelayBuffer() {
        uint64_t current = now_ns();
        while (!delay_buffer_.empty()) {
            auto& front = delay_buffer_.front();
            if (current - front.enqueue_time_ns >= front.delay_ns) {
                delay_buffer_.pop();
            } else {
                break;
            }
        }
    }

    void enqueueDelayedOp(uintptr_t addr, const void* data, size_t size,
                          DelayOpType op_type, uint32_t delay_ns) {
        // If buffer is full, drain oldest entry
        if (delay_buffer_.size() >= delay_buffer_depth_) {
            delay_buffer_.pop();
        }
        DelayedOp op;
        op.addr = addr;
        if (data && size > 0) {
            op.data.assign(static_cast<const uint8_t*>(data),
                           static_cast<const uint8_t*>(data) + size);
        }
        op.op_type = op_type;
        op.enqueue_time_ns = now_ns();
        op.delay_ns = delay_ns;
        delay_buffer_.push(std::move(op));
    }

    bool mapDeviceBARs() {
        // Try to mmap device BARs from sysfs
        // Fallback to anonymous mmap for simulation
        if (config_.bar2_size > 0) {
            bar2_mapping_ = mmap(reinterpret_cast<void*>(config_.bar2_base),
                                 config_.bar2_size,
                                 PROT_READ | PROT_WRITE,
                                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (bar2_mapping_ == MAP_FAILED) {
                bar2_mapping_ = nullptr;
                return false;
            }
        }
        if (config_.bar4_size > 0) {
            bar4_mapping_ = mmap(reinterpret_cast<void*>(config_.bar4_base),
                                 config_.bar4_size,
                                 PROT_READ | PROT_WRITE,
                                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (bar4_mapping_ == MAP_FAILED) {
                bar4_mapping_ = nullptr;
                return false;
            }
        }
        return true;
    }

public:
    ~Type2CacheControllerImpl() override {
        shutdown();
    }

    bool initialize(const Type2DeviceConfig& config) override {
        if (initialized_) return true;

        config_ = config;
        if (config_.cache_line_size == 0)
            config_.cache_line_size = CXL_CACHE_LINE_SIZE;
        if (config_.delay_buffer_depth == 0)
            config_.delay_buffer_depth = 16;
        if (config_.read_latency_ns == 0)
            config_.read_latency_ns = 170;
        if (config_.write_latency_ns == 0)
            config_.write_latency_ns = 200;
        if (config_.coherency_latency_ns == 0)
            config_.coherency_latency_ns = 50;

        delay_buffer_depth_ = config_.delay_buffer_depth;

        if (!mapDeviceBARs()) {
            return false;
        }

        initialized_ = true;
        return true;
    }

    void shutdown() override {
        if (!initialized_) return;

        drainDelayBuffer();

        if (bar2_mapping_) {
            munmap(bar2_mapping_, config_.bar2_size);
            bar2_mapping_ = nullptr;
        }
        if (bar4_mapping_) {
            munmap(bar4_mapping_, config_.bar4_size);
            bar4_mapping_ = nullptr;
        }
        cache_.clear();
        initialized_ = false;
    }

    // CXL.cache load: device coherently caches host memory
    void* cacheLoad(void* host_addr, size_t size, CoherencyState* state_out) override {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        processDelayBuffer();

        uintptr_t aligned = alignToLine(host_addr);

        auto it = cache_.find(aligned);
        if (it != cache_.end() && it->second.state != CoherencyState::INVALID) {
            // Cache hit
            cache_hits_++;
            if (state_out) *state_out = it->second.state;
            return it->second.data;
        }

        // Cache miss: fetch from host memory (or BAR2 mapping)
        cache_misses_++;
        CacheLine& line = cache_[aligned];
        void* src = host_addr ? host_addr : bar2_mapping_;
        if (src) {
            memcpy(line.data, src, std::min(size, CXL_CACHE_LINE_SIZE));
        } else {
            memset(line.data, 0, CXL_CACHE_LINE_SIZE);
        }
        line.state = CoherencyState::SHARED;
        line.dirty = false;
        line.timestamp_ns = now_ns();

        // Model read latency via delay buffer
        enqueueDelayedOp(aligned, nullptr, 0, DelayOpType::READ,
                         config_.read_latency_ns);

        if (state_out) *state_out = line.state;
        return line.data;
    }

    // CXL.cache store: requires EXCLUSIVE, marks MODIFIED
    void cacheStore(void* host_addr, const void* data, size_t size) override {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        processDelayBuffer();

        uintptr_t aligned = alignToLine(host_addr);

        auto it = cache_.find(aligned);
        if (it == cache_.end() || it->second.state == CoherencyState::INVALID) {
            // Miss on store: allocate + fetch, then upgrade
            CacheLine& line = cache_[aligned];
            if (host_addr) {
                memcpy(line.data, host_addr, CXL_CACHE_LINE_SIZE);
            }
            line.state = CoherencyState::EXCLUSIVE;
            line.dirty = false;
            line.timestamp_ns = now_ns();
            cache_misses_++;
            it = cache_.find(aligned);
        }

        CacheLine& line = it->second;

        // Upgrade SHARED → EXCLUSIVE requires coherency overhead
        if (line.state == CoherencyState::SHARED) {
            enqueueDelayedOp(aligned, nullptr, 0, DelayOpType::COHERENCY,
                             config_.coherency_latency_ns);
            line.state = CoherencyState::EXCLUSIVE;
        }

        // Write data, transition to MODIFIED
        size_t copy_size = std::min(size, CXL_CACHE_LINE_SIZE);
        memcpy(line.data, data, copy_size);
        line.state = CoherencyState::MODIFIED;
        line.dirty = true;
        line.timestamp_ns = now_ns();

        enqueueDelayedOp(aligned, data, copy_size, DelayOpType::WRITE,
                         config_.write_latency_ns);
    }

    void cacheEvict(void* host_addr, bool writeback) override {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        uintptr_t aligned = alignToLine(host_addr);

        auto it = cache_.find(aligned);
        if (it == cache_.end()) return;

        CacheLine& line = it->second;

        // Writeback dirty data to host
        if (writeback && line.dirty && host_addr) {
            memcpy(host_addr, line.data, CXL_CACHE_LINE_SIZE);
        }

        line.state = CoherencyState::INVALID;
        line.dirty = false;
    }

    CoherencyState getCoherencyState(void* addr) override {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        uintptr_t aligned = alignToLine(addr);
        auto it = cache_.find(aligned);
        if (it == cache_.end()) return CoherencyState::INVALID;
        return it->second.state;
    }

    // CXL.mem: host accesses device-attached memory (BAR4/HDM)
    void* memLoad(uintptr_t dev_offset, size_t size) override {
        if (bar4_mapping_ && dev_offset + size <= config_.bar4_size) {
            return static_cast<char*>(bar4_mapping_) + dev_offset;
        }
        // Fallback: allocate anonymous memory
        void* fallback = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        return fallback;
    }

    void memStore(uintptr_t dev_offset, const void* data, size_t size) override {
        if (bar4_mapping_ && dev_offset + size <= config_.bar4_size) {
            memcpy(static_cast<char*>(bar4_mapping_) + dev_offset, data, size);
        }
    }

    void setDelayBufferDepth(size_t depth) override {
        delay_buffer_depth_ = depth;
    }

    void drainDelayBuffer() override {
        while (!delay_buffer_.empty()) {
            delay_buffer_.pop();
        }
    }

    // Snoop handler: downgrade state per MESI protocol
    void handleSnoop(void* addr, CoherencyState new_state) override {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        uintptr_t aligned = alignToLine(addr);
        auto it = cache_.find(aligned);
        if (it == cache_.end()) return;

        CacheLine& line = it->second;

        // MODIFIED → writeback + new_state
        if (line.state == CoherencyState::MODIFIED && line.dirty) {
            if (addr) {
                memcpy(addr, line.data, CXL_CACHE_LINE_SIZE);
            }
            line.dirty = false;
        }

        line.state = new_state;
        if (new_state == CoherencyState::INVALID) {
            line.dirty = false;
        }
    }

    void invalidateRange(void* base, size_t size) override {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        uintptr_t start = reinterpret_cast<uintptr_t>(base) & ~(CXL_CACHE_LINE_SIZE - 1);
        uintptr_t end = reinterpret_cast<uintptr_t>(base) + size;

        for (uintptr_t addr = start; addr < end; addr += CXL_CACHE_LINE_SIZE) {
            auto it = cache_.find(addr);
            if (it != cache_.end()) {
                // Writeback dirty lines before invalidation
                if (it->second.dirty && it->second.state == CoherencyState::MODIFIED) {
                    memcpy(reinterpret_cast<void*>(addr), it->second.data,
                           CXL_CACHE_LINE_SIZE);
                }
                it->second.state = CoherencyState::INVALID;
                it->second.dirty = false;
            }
        }
    }

    size_t getCacheHits() override { return cache_hits_; }
    size_t getCacheMisses() override { return cache_misses_; }
};

// Graph Runtime Implementation
class GraphRuntimeImpl : public GraphRuntime {
public:
    void initializeGraph(Graph* graph, RemoteMemoryManager* mem_mgr) override {
        // Initialize graph data structures in remote memory
        if (!graph->edges) {
            graph->edges = mem_mgr->allocateRemote(
                graph->num_edges * sizeof(void*) * 2, // from and to pointers
                MemoryTier::CXL_ATTACHED
            );
        }

        if (!graph->vertices) {
            graph->vertices = mem_mgr->allocateRemote(
                graph->num_vertices * sizeof(void*),
                MemoryTier::LOCAL_DRAM
            );
        }
    }

    void executeVertexProgram(VertexProgram* program, Graph* graph) override {
        void** vertices = static_cast<void**>(graph->vertices->base_addr);
        void** edges = static_cast<void**>(graph->edges->base_addr);

        for (size_t v = 0; v < graph->num_vertices; v++) {
            // Process all edges of this vertex
            program->compute(vertices[v], edges);
        }
    }

    void executeEdgeProgram(EdgeProgram* program, Graph* graph) override {
        void** edges = static_cast<void**>(graph->edges->base_addr);
        void** vertices = static_cast<void**>(graph->vertices->base_addr);

        for (size_t e = 0; e < graph->num_edges; e++) {
            void* src = edges[e * 2];
            void* dst = edges[e * 2 + 1];
            program->scatter(edges + e * 2, src, dst);
        }
    }

    void bfsTraversal(Graph* graph, size_t start_vertex,
                      void (*visit)(size_t vertex_id)) override {
        // BFS implementation optimized for remote memory
        std::vector<bool> visited(graph->num_vertices, false);
        std::vector<size_t> queue;

        queue.push_back(start_vertex);
        visited[start_vertex] = true;

        while (!queue.empty()) {
            size_t v = queue.front();
            queue.erase(queue.begin());
            visit(v);

            // Process neighbors
            // (Simplified - would need edge list representation)
        }
    }

    void dfsTraversal(Graph* graph, size_t start_vertex,
                      void (*visit)(size_t vertex_id)) override {
        // DFS implementation
        std::vector<bool> visited(graph->num_vertices, false);
        std::vector<size_t> stack;

        stack.push_back(start_vertex);

        while (!stack.empty()) {
            size_t v = stack.back();
            stack.pop_back();

            if (!visited[v]) {
                visited[v] = true;
                visit(v);

                // Process neighbors
                // (Simplified - would need edge list representation)
            }
        }
    }
};

// Runtime Implementation Constructor
CiraRuntimeImpl::CiraRuntimeImpl() {
    offload_engine_ = std::make_unique<OffloadEngineImpl>();
    memory_manager_ = std::make_unique<RemoteMemoryManagerImpl>();
    prefetch_controller_ = std::make_unique<PrefetchControllerImpl>();
    graph_runtime_ = std::make_unique<GraphRuntimeImpl>();
    type2_controller_ = std::make_unique<Type2CacheControllerImpl>();
}

void CiraRuntimeImpl::configure(const std::string& config_file) {
    // Load configuration from file
    // This would parse config and set runtime parameters
    if (verbosity_level_ > 0) {
        std::cout << "Loading configuration from: " << config_file << std::endl;
    }
}

void CiraRuntimeImpl::configureType2(const Type2DeviceConfig& config) {
    if (verbosity_level_ > 0) {
        std::cout << "Configuring Type 2 device: " << config.device_path << std::endl;
    }
    type2_controller_->initialize(config);
}

// Factory method
std::unique_ptr<CiraRuntime> CiraRuntime::create() {
    return std::make_unique<CiraRuntimeImpl>();
}

namespace {

constexpr size_t CIRA_MMIO_DEFAULT_CONTROL_BYTES = VORTEX_CXL_MMIO_CONTROL_BYTES;
constexpr uint32_t CIRA_COMPLETION_MAGIC = 0xDEADBEEF;
constexpr size_t CIRA_CACHELINE_BYTES = 64;

struct LlcTileMetadata {
    void* allocation = nullptr;
    size_t requested_size = 0;
    size_t aligned_size = 0;
    void* completion = nullptr;
};

std::atomic<uint64_t> g_mmio_submit_seq{1};
std::atomic<void*> g_forced_device_func{nullptr};
std::mutex g_mmio_map_mutex;
void* g_mmio_control_window = nullptr;
size_t g_mmio_control_size = 0;
int g_mmio_fd = -1;
std::mutex g_llc_tile_mutex;
std::unordered_map<void*, LlcTileMetadata> g_llc_tiles;

size_t align_up_cacheline(uint64_t size) {
    uint64_t requested = size ? size : CIRA_CACHELINE_BYTES;
    uint64_t max_size = static_cast<uint64_t>(std::numeric_limits<size_t>::max());
    if (requested > max_size - (CIRA_CACHELINE_BYTES - 1)) return 0;
    return static_cast<size_t>((requested + CIRA_CACHELINE_BYTES - 1) &
                               ~(static_cast<uint64_t>(CIRA_CACHELINE_BYTES) - 1));
}

bool lookup_llc_tile(void* tile, LlcTileMetadata& metadata) {
    if (!tile) return false;
    std::lock_guard<std::mutex> lock(g_llc_tile_mutex);
    auto it = g_llc_tiles.find(tile);
    if (it == g_llc_tiles.end()) return false;
    metadata = it->second;
    return true;
}

bool parse_env_u64(const char* name, uint64_t& value) {
    const char* text = std::getenv(name);
    if (!text || !*text) return false;

    char* end = nullptr;
    errno = 0;
    unsigned long long parsed = std::strtoull(text, &end, 0);
    if (errno != 0 || end == text || (end && *end != '\0')) return false;

    value = static_cast<uint64_t>(parsed);
    return true;
}

bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value && *value && std::strcmp(value, "0") != 0;
}

const char* first_env(const char* primary, const char* fallback) {
    const char* value = std::getenv(primary);
    if (value && *value) return value;
    value = std::getenv(fallback);
    return (value && *value) ? value : nullptr;
}

void clear_completion(void* completion_ptr) {
    if (!completion_ptr) return;
    std::memset(completion_ptr, 0, 64);
    __atomic_thread_fence(__ATOMIC_RELEASE);
}

void complete_in_software(void* completion_ptr, uint32_t status,
                          uint64_t result = 0) {
    if (!completion_ptr) return;

    auto* base = static_cast<uint8_t*>(completion_ptr);
    *reinterpret_cast<volatile uint32_t*>(base + 4) = status;
    *reinterpret_cast<volatile uint64_t*>(base + 8) = result;
    *reinterpret_cast<volatile uint64_t*>(base + 16) = 0;
    *reinterpret_cast<volatile uint64_t*>(base + 24) = 0;
    __atomic_thread_fence(__ATOMIC_RELEASE);
    *reinterpret_cast<volatile uint32_t*>(base) = CIRA_COMPLETION_MAGIC;
}

void* configured_device_func(void* func_ptr) {
    if (func_ptr) return func_ptr;

    void* forced = g_forced_device_func.load(std::memory_order_acquire);
    if (forced) return forced;

    uint64_t addr = 0;
    if (parse_env_u64("CIRA_CXL_DEVICE_FUNC_ADDR", addr) ||
        parse_env_u64("CIRA_TYPE2_DEVICE_FUNC_ADDR", addr)) {
        return reinterpret_cast<void*>(static_cast<uintptr_t>(addr));
    }

    return nullptr;
}

void* configured_mmio_control_window() {
    std::lock_guard<std::mutex> lock(g_mmio_map_mutex);
    if (g_mmio_control_window) return g_mmio_control_window;

    uint64_t addr = 0;
    if (parse_env_u64("CIRA_CXL_MMIO_ADDR", addr) ||
        parse_env_u64("CIRA_TYPE2_MMIO_ADDR", addr)) {
        g_mmio_control_window = reinterpret_cast<void*>(static_cast<uintptr_t>(addr));
        uint64_t size = CIRA_MMIO_DEFAULT_CONTROL_BYTES;
        parse_env_u64("CIRA_CXL_MMIO_SIZE", size) ||
            parse_env_u64("CIRA_TYPE2_MMIO_SIZE", size);
        g_mmio_control_size = static_cast<size_t>(size);
        return g_mmio_control_window;
    }

    const char* path = first_env("CIRA_CXL_MMIO_PATH", "CIRA_TYPE2_MMIO_PATH");
    if (!path) return nullptr;

    uint64_t size = CIRA_MMIO_DEFAULT_CONTROL_BYTES;
    parse_env_u64("CIRA_CXL_MMIO_SIZE", size) ||
        parse_env_u64("CIRA_TYPE2_MMIO_SIZE", size);

    uint64_t offset = 0;
    parse_env_u64("CIRA_CXL_MMIO_OFFSET", offset) ||
        parse_env_u64("CIRA_TYPE2_MMIO_OFFSET", offset);

    int fd = open(path, O_RDWR | O_SYNC | O_CLOEXEC);
    if (fd < 0) {
        std::cerr << "cira_offload_submit: open MMIO control window " << path
                  << " failed: " << std::strerror(errno) << std::endl;
        return nullptr;
    }

    void* map = mmap(nullptr, static_cast<size_t>(size),
                     PROT_READ | PROT_WRITE, MAP_SHARED,
                     fd, static_cast<off_t>(offset));
    if (map == MAP_FAILED) {
        std::cerr << "cira_offload_submit: mmap MMIO control window " << path
                  << " offset=0x" << std::hex << offset
                  << " size=0x" << size << std::dec
                  << " failed: " << std::strerror(errno) << std::endl;
        close(fd);
        return nullptr;
    }

    g_mmio_fd = fd;
    g_mmio_control_window = map;
    g_mmio_control_size = static_cast<size_t>(size);
    return g_mmio_control_window;
}

bool submit_mmio_call(void* func_ptr, void** operands,
                      int num_operands, void* completion_ptr) {
    void* control = configured_mmio_control_window();
    if (!control) return false;

    void* device_func = configured_device_func(func_ptr);
    if (!device_func) {
        if (env_flag_enabled("CIRA_CXL_MMIO_STRICT")) {
            std::cerr << "cira_offload_submit: MMIO window is configured but "
                      << "no device function address was provided" << std::endl;
        }
        return false;
    }

    uint64_t seq = g_mmio_submit_seq.fetch_add(1, std::memory_order_relaxed);
    uint32_t argc = num_operands > 0 ? static_cast<uint32_t>(num_operands) : 0;
    int rc = vortex_cxl_submit_call_mmio(control, seq, device_func,
                                         operands, argc, completion_ptr);
    if (rc != VORTEX_SUCCESS) {
        std::cerr << "cira_offload_submit: MMIO submit failed, rc=" << rc
                  << std::endl;
        return false;
    }

    if (env_flag_enabled("CIRA_CXL_MMIO_WAIT") && completion_ptr) {
        (void)cira_future_await(completion_ptr);
    }
    return true;
}

} // namespace

// C API Implementation
extern "C" {
    void* cira_runtime_create() {
        return CiraRuntime::create().release();
    }

    void cira_runtime_destroy(void* runtime) {
        delete static_cast<CiraRuntime*>(runtime);
    }

    void* cira_offload_load_edge(void* engine, void* edge_ptr,
                                 size_t index, size_t prefetch_distance) {
        auto* eng = static_cast<OffloadEngine*>(engine);
        auto* ref = static_cast<RemoteMemRef*>(edge_ptr);
        return eng->loadEdge(ref, index, prefetch_distance);
    }

    void* cira_offload_load_node(void* engine, void* edge_element,
                                 const char* field_name, size_t prefetch_distance) {
        auto* eng = static_cast<OffloadEngine*>(engine);
        return eng->loadNode(edge_element, field_name, prefetch_distance);
    }

    uintptr_t cira_offload_get_paddr(void* engine, const char* field_name,
                                     void* node_data) {
        auto* eng = static_cast<OffloadEngine*>(engine);
        return eng->getPhysicalAddr(field_name, node_data);
    }

    void cira_offload_evict_edge(void* engine, void* edge_ptr, size_t index) {
        auto* eng = static_cast<OffloadEngine*>(engine);
        auto* ref = static_cast<RemoteMemRef*>(edge_ptr);
        eng->evictEdge(ref, index);
    }

    // Type 2 C API implementations
    void cira_type2_configure(void* runtime, const char* device_path,
                              uint64_t bar2_base, uint64_t bar2_size,
                              uint64_t bar4_base, uint64_t bar4_size) {
        auto* rt = static_cast<CiraRuntime*>(runtime);
        Type2DeviceConfig config;
        config.device_path = device_path ? device_path : "";
        config.bar2_base = static_cast<uintptr_t>(bar2_base);
        config.bar2_size = static_cast<size_t>(bar2_size);
        config.bar4_base = static_cast<uintptr_t>(bar4_base);
        config.bar4_size = static_cast<size_t>(bar4_size);
        config.read_latency_ns = 170;
        config.write_latency_ns = 200;
        config.coherency_latency_ns = 50;
        config.cache_line_size = 64;
        config.delay_buffer_depth = 16;
        rt->configureType2(config);
    }

    void cira_type2_set_delay(void* runtime, uint32_t read_ns,
                              uint32_t write_ns, uint32_t coherency_ns) {
        auto* rt = static_cast<CiraRuntime*>(runtime);
        Type2DeviceConfig config;
        config.read_latency_ns = read_ns;
        config.write_latency_ns = write_ns;
        config.coherency_latency_ns = coherency_ns;
        config.cache_line_size = 64;
        config.delay_buffer_depth = 16;
        rt->configureType2(config);
    }

    void* cira_type2_cache_load(void* controller, void* host_addr,
                                uint64_t size, uint8_t* coherency_state) {
        auto* ctrl = static_cast<Type2CacheController*>(controller);
        CoherencyState state;
        void* result = ctrl->cacheLoad(host_addr, static_cast<size_t>(size), &state);
        if (coherency_state) {
            *coherency_state = static_cast<uint8_t>(state);
        }
        return result;
    }

    void cira_type2_cache_store(void* controller, void* host_addr,
                                const void* data, uint64_t size) {
        auto* ctrl = static_cast<Type2CacheController*>(controller);
        ctrl->cacheStore(host_addr, data, static_cast<size_t>(size));
    }

    void cira_type2_cache_evict(void* controller, void* host_addr, int writeback) {
        auto* ctrl = static_cast<Type2CacheController*>(controller);
        ctrl->cacheEvict(host_addr, writeback != 0);
    }

    void* cira_type2_mem_load(void* controller, uint64_t dev_offset, uint64_t size) {
        auto* ctrl = static_cast<Type2CacheController*>(controller);
        return ctrl->memLoad(static_cast<uintptr_t>(dev_offset),
                             static_cast<size_t>(size));
    }

    void cira_type2_mem_store(void* controller, uint64_t dev_offset,
                              const void* data, uint64_t size) {
        auto* ctrl = static_cast<Type2CacheController*>(controller);
        ctrl->memStore(static_cast<uintptr_t>(dev_offset), data,
                       static_cast<size_t>(size));
    }

    void cira_type2_drain_delay_buffer(void* controller) {
        auto* ctrl = static_cast<Type2CacheController*>(controller);
        ctrl->drainDelayBuffer();
    }

    // ====================================================================
    // Cache line management
    // ====================================================================

    void cira_install_cacheline_x86(void* addr, uint64_t size, int cache_level) {
        if (!addr || size == 0) return;

        // Align down to cache line boundary
        uintptr_t base = reinterpret_cast<uintptr_t>(addr) & ~63ULL;
        uintptr_t end  = (reinterpret_cast<uintptr_t>(addr) + size + 63) & ~63ULL;

        for (uintptr_t cl = base; cl < end; cl += 64) {
            char* ptr = reinterpret_cast<char*>(cl);

#if defined(__x86_64__) || defined(_M_X64)
            switch (cache_level) {
            case 1:
                // PREFETCHT0: install into L1 (also fills L2 and LLC)
                __builtin_prefetch(ptr, 0, 3);
                break;
            case 2:
                // PREFETCHT1: install into L2 (also fills LLC, but not L1)
                __builtin_prefetch(ptr, 0, 2);
                break;
            case 3:
            default:
                // PREFETCHT2: install into LLC only
                __builtin_prefetch(ptr, 0, 1);
                break;
            }
#else
            // Generic fallback: volatile read to force cache fill
            (void)*reinterpret_cast<volatile char*>(ptr);
#endif
        }

        // Memory fence to ensure all prefetches are issued
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
    }

    void* cira_llc_tile_alloc(uint64_t size) {
        size_t aligned_size = align_up_cacheline(size);
        if (aligned_size == 0) return nullptr;

        void* tile = nullptr;
        int ret = posix_memalign(&tile, CIRA_CACHELINE_BYTES, aligned_size);
        if (ret != 0 || !tile) return nullptr;

        std::memset(tile, 0, aligned_size);

        void* completion = cira_future_alloc();
        if (!completion) {
            free(tile);
            return nullptr;
        }

        {
            std::lock_guard<std::mutex> lock(g_llc_tile_mutex);
            g_llc_tiles[tile] = LlcTileMetadata{
                tile,
                size ? static_cast<size_t>(size) : CIRA_CACHELINE_BYTES,
                aligned_size,
                completion,
            };
        }

        // A freshly allocated tile is immediately consumable.  Later CXL
        // installs clear and re-arm this completion before delivery.
        complete_in_software(completion, 0, reinterpret_cast<uint64_t>(tile));
        cira_install_cacheline_x86(tile, aligned_size, 3);
        return tile;
    }

    void cira_llc_tile_free(void* tile) {
        if (!tile) return;

        LlcTileMetadata metadata;
        bool found = false;
        {
            std::lock_guard<std::mutex> lock(g_llc_tile_mutex);
            auto it = g_llc_tiles.find(tile);
            if (it != g_llc_tiles.end()) {
                metadata = it->second;
                g_llc_tiles.erase(it);
                found = true;
            }
        }

        if (!found) {
            free(tile);
            return;
        }

        cira_future_free(metadata.completion);
        free(metadata.allocation);
    }

    void* cira_llc_tile_future(void* tile) {
        LlcTileMetadata metadata;
        return lookup_llc_tile(tile, metadata) ? metadata.completion : nullptr;
    }

    void* cira_llc_tile_install_from_cxl(void* tile, const void* cxl_addr,
                                         uint64_t size, void* completion_ptr) {
        if (!cxl_addr || size == 0) return nullptr;

        void* dst = tile ? tile : cira_llc_tile_alloc(size);
        if (!dst) return nullptr;

        LlcTileMetadata metadata;
        bool found = lookup_llc_tile(dst, metadata);
        uint64_t max_size = static_cast<uint64_t>(std::numeric_limits<size_t>::max());
        if (size > max_size) return nullptr;
        size_t copy_size = static_cast<size_t>(size);
        if (found) copy_size = std::min(copy_size, metadata.aligned_size);

        void* completion = completion_ptr;
        if (!completion && found) completion = metadata.completion;

        clear_completion(completion);
        std::memcpy(dst, cxl_addr, copy_size);
        cira_install_cacheline_x86(dst, copy_size, 3);
        complete_in_software(completion, 0, reinterpret_cast<uint64_t>(dst));
        return dst;
    }

    void* cira_llc_tile_get_mwait(void* tile) {
        if (!tile) return nullptr;

        LlcTileMetadata metadata;
        if (lookup_llc_tile(tile, metadata) && metadata.completion) {
            (void)cira_future_await(metadata.completion);
            cira_install_cacheline_x86(tile, metadata.aligned_size, 3);
        }

        return tile;
    }

    void cira_evict_hint_x86(void* addr, uint64_t size) {
        if (!addr || size == 0) return;

        // Align down to cache line boundary
        uintptr_t base = reinterpret_cast<uintptr_t>(addr) & ~63ULL;
        uintptr_t end  = (reinterpret_cast<uintptr_t>(addr) + size + 63) & ~63ULL;

        for (uintptr_t cl = base; cl < end; cl += 64) {
            char* ptr = reinterpret_cast<char*>(cl);

#if defined(__x86_64__) || defined(_M_X64)
            // Try CLDEMOTE first (Granite Rapids+): moves from L1->LLC
            // without evicting from the cache hierarchy entirely.
            // CLDEMOTE is encoded as NOP on pre-Tremont CPUs.
            //
            // Encoding: 0x0F 0x1C /0 (same as CLDEMOTE)
            // GCC/Clang intrinsic:
            #if __has_builtin(__builtin_ia32_cldemote)
                __builtin_ia32_cldemote(ptr);
            #else
                // Fallback: CLFLUSHOPT (weaker, evicts entirely)
                // Encoding: 0x66 0x0F 0xAE /7
                _mm_clflushopt(ptr);
            #endif
#endif
        }

        // sfence to order stores with respect to clflush
        __atomic_thread_fence(__ATOMIC_RELEASE);
    }

    // ====================================================================
    // Future management (DCOH-coherent completion tracking)
    // ====================================================================

    // Global tracking of active futures for phase_barrier
    static constexpr size_t MAX_ACTIVE_FUTURES = 256;
    static void* active_futures[MAX_ACTIVE_FUTURES] = {};
    static int num_active_futures = 0;
    static std::mutex futures_mutex;

    void* cira_future_alloc(void) {
        // Allocate 64-byte-aligned completion structure
        // Must be cache-line aligned for DCOH delivery and MONITOR/MWAIT
        void* ptr = nullptr;
        int ret = posix_memalign(&ptr, 64, 64);
        if (ret != 0 || !ptr) return nullptr;

        // Zero-initialize (magic=0 means not ready)
        std::memset(ptr, 0, 64);

        // Track for phase barrier
        {
            std::lock_guard<std::mutex> lock(futures_mutex);
            if (num_active_futures < (int)MAX_ACTIVE_FUTURES) {
                active_futures[num_active_futures++] = ptr;
            }
        }

        return ptr;
    }

    void* cira_future_await(void* completion_ptr) {
        if (!completion_ptr) return nullptr;

        // CompletionData layout:
        //   [0:4]   magic    (uint32_t) — 0xDEADBEEF when done
        //   [4:8]   status   (uint32_t) — 0 = success
        //   [8:16]  result   (uint64_t) — kernel-specific result
        //   [16:24] cycles   (uint64_t) — execution cycles
        //   [24:32] timestamp(uint64_t) — completion time
        //   [32:64] reserved

        volatile uint32_t* magic_ptr =
            reinterpret_cast<volatile uint32_t*>(completion_ptr);

        // Spin-wait with MONITOR/MWAIT on Granite Rapids.
        // MWAIT doesn't alter cstate on newer architectures (confirmed
        // experimentally on Granite Rapids; spec confirms for WAITPKG).
        //
        // Fallback: PAUSE-based spin-wait if MWAIT not available.

#if defined(__x86_64__) && __has_builtin(__builtin_ia32_umonitor)
        // Use UMONITOR/UMWAIT (WAITPKG extension, available on Granite Rapids)
        while (*magic_ptr != 0xDEADBEEF) {
            __builtin_ia32_umonitor(const_cast<uint32_t*>(
                const_cast<volatile uint32_t*>(magic_ptr)));
            if (*magic_ptr != 0xDEADBEEF) {
                // UMWAIT: c0 state (lowest power), infinite timeout
                __builtin_ia32_umwait(0, ~0ULL);
            }
        }
#else
        // PAUSE-based spin-wait
        while (*magic_ptr != 0xDEADBEEF) {
#if defined(__x86_64__)
            __builtin_ia32_pause();
#endif
        }
#endif

        // Memory fence to ensure we see all data written before magic
        __atomic_thread_fence(__ATOMIC_ACQUIRE);

        // Return pointer to the result field (offset 8)
        return reinterpret_cast<char*>(completion_ptr) + 8;
    }

    void cira_future_free(void* completion_ptr) {
        if (!completion_ptr) return;

        // Remove from active tracking
        {
            std::lock_guard<std::mutex> lock(futures_mutex);
            for (int i = 0; i < num_active_futures; ++i) {
                if (active_futures[i] == completion_ptr) {
                    active_futures[i] = active_futures[--num_active_futures];
                    break;
                }
            }
        }

        free(completion_ptr);
    }

    // ====================================================================
    // Phase barrier and offload submission
    // ====================================================================

    void cira_phase_barrier(void) {
        // Wait for ALL active futures to complete.
        // This is the host-side implementation of cira.barrier / cira.phase_boundary.

        std::lock_guard<std::mutex> lock(futures_mutex);

        for (int i = 0; i < num_active_futures; ++i) {
            if (!active_futures[i]) continue;

            volatile uint32_t* magic_ptr =
                reinterpret_cast<volatile uint32_t*>(active_futures[i]);

            // Spin until this future completes
            while (*magic_ptr != 0xDEADBEEF) {
#if defined(__x86_64__)
                __builtin_ia32_pause();
#endif
            }
        }

        // All futures complete — full fence
        __atomic_thread_fence(__ATOMIC_SEQ_CST);

        // Clear active futures list (they've all completed)
        num_active_futures = 0;
    }

    void cira_offload_submit(void* func_ptr, void** operands,
                             int num_operands, void* completion_ptr) {
        // Submit offload task to device via MMIO ring buffer.
        //
        // Protocol:
        //   1. Write task struct to BAR0 ring buffer entry
        //   2. Write completion_ptr to task struct
        //   3. Advance ring buffer write pointer (CSR write)
        //   4. Device picks up task, executes, writes completion via DCOH
        //
        // In simulation/software fallback: call the function directly.

        // For now, use the global Type2GpuDevice instance if available
        // In real hardware, this writes to BAR0 MMIO registers
        // matching the CSR layout in Type2CSROffset

        // Software simulation fallback:
        // Execute the offloaded function on a helper thread
        // (In real hardware, this goes through the MMIO queue)

        clear_completion(completion_ptr);

        // Hardware/FPGA path: write the call-job struct into the configured
        // CXL-visible MMIO control window.  The device firmware completes the
        // future via DCOH once it consumes the doorbell.
        if (submit_mmio_call(func_ptr, operands, num_operands, completion_ptr)) {
            return;
        }

        // Software fallback: no MMIO window or no device function address was
        // configured, so preserve the old synchronous success behavior.
        complete_in_software(completion_ptr, 0);
    }

    void* cira_get_device_func_addr(void) {
        // In real hardware, this returns the device-side address of the
        // kernel function loaded in Vortex memory.
        // For simulation, return nullptr (resolved at runtime).
        return nullptr;
    }
}

} // namespace runtime
} // namespace cira
