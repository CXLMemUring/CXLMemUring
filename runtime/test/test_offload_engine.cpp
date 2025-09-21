#include "CiraRuntime.h"
#include <iostream>
#include <cassert>
#include <vector>

using namespace cira::runtime;

struct TestEdge {
    int from;
    int to;
    float weight;
};

int main() {
    std::cout << "Testing Cira Offload Engine...\n";

    // Create runtime
    auto runtime = CiraRuntime::create();
    auto* offload_engine = runtime->getOffloadEngine();
    auto* memory_manager = runtime->getMemoryManager();

    // Test 1: Allocate remote memory
    std::cout << "Test 1: Remote memory allocation\n";
    size_t num_edges = 1000;
    auto* edge_ref = memory_manager->allocateRemote(
        sizeof(TestEdge) * num_edges,
        MemoryTier::CXL_ATTACHED
    );
    assert(edge_ref != nullptr);
    assert(edge_ref->size == sizeof(TestEdge) * num_edges);
    std::cout << "  ✓ Allocated " << edge_ref->size << " bytes in CXL memory\n";

    // Initialize edge data
    TestEdge* edges = static_cast<TestEdge*>(edge_ref->base_addr);
    for (size_t i = 0; i < num_edges; i++) {
        edges[i].from = i;
        edges[i].to = (i + 1) % num_edges;
        edges[i].weight = 1.0f / (i + 1);
    }

    // Test 2: Load edge with prefetching
    std::cout << "Test 2: Edge loading with prefetch\n";
    void* edge_ptr = offload_engine->loadEdge(edge_ref, 0, 2);
    assert(edge_ptr != nullptr);
    TestEdge* loaded_edge = static_cast<TestEdge*>(edge_ptr);
    assert(loaded_edge->from == 0);
    assert(loaded_edge->to == 1);
    std::cout << "  ✓ Loaded edge[0]: from=" << loaded_edge->from
              << ", to=" << loaded_edge->to << "\n";

    // Test 3: Batch load edges
    std::cout << "Test 3: Batch edge loading\n";
    std::vector<void*> batch_results(10);
    offload_engine->batchLoadEdges(edge_ref, 10, 10, batch_results.data());
    for (size_t i = 0; i < 10; i++) {
        TestEdge* e = static_cast<TestEdge*>(batch_results[i]);
        assert(e->from == static_cast<int>(10 + i));
    }
    std::cout << "  ✓ Batch loaded 10 edges successfully\n";

    // Test 4: Load node data
    std::cout << "Test 4: Node data loading\n";
    void* node_from = offload_engine->loadNode(edge_ptr, "from", 1);
    void* node_to = offload_engine->loadNode(edge_ptr, "to", 1);
    assert(node_from != nullptr);
    assert(node_to != nullptr);
    std::cout << "  ✓ Loaded node data for 'from' and 'to' fields\n";

    // Test 5: Get physical address
    std::cout << "Test 5: Physical address retrieval\n";
    uintptr_t paddr_from = offload_engine->getPhysicalAddr("from", node_from);
    uintptr_t paddr_to = offload_engine->getPhysicalAddr("to", node_to);
    assert(paddr_from != 0);
    assert(paddr_to != 0);
    std::cout << "  ✓ Got physical addresses: from=" << std::hex << paddr_from
              << ", to=" << paddr_to << std::dec << "\n";

    // Test 6: Edge eviction
    std::cout << "Test 6: Edge cache eviction\n";
    offload_engine->evictEdge(edge_ref, 0);
    std::cout << "  ✓ Evicted edge[0] from cache\n";

    // Test 7: Memory tier migration
    std::cout << "Test 7: Memory tier migration\n";
    auto original_tier = memory_manager->getTier(edge_ref);
    assert(original_tier == MemoryTier::CXL_ATTACHED);

    memory_manager->migrate(edge_ref, MemoryTier::LOCAL_DRAM);
    auto new_tier = memory_manager->getTier(edge_ref);
    assert(new_tier == MemoryTier::LOCAL_DRAM);
    std::cout << "  ✓ Migrated from CXL to LOCAL_DRAM\n";

    // Test 8: Memory statistics
    std::cout << "Test 8: Memory statistics\n";
    size_t used_dram = memory_manager->getUsedMemory(MemoryTier::LOCAL_DRAM);
    size_t avail_dram = memory_manager->getAvailableMemory(MemoryTier::LOCAL_DRAM);
    std::cout << "  DRAM: " << used_dram / (1024*1024) << " MB used, "
              << avail_dram / (1024*1024) << " MB available\n";

    // Cleanup
    memory_manager->deallocateRemote(edge_ref);
    std::cout << "\n✅ All tests passed!\n";

    return 0;
}