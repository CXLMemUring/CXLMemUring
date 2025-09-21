#include "CiraRuntime.h"
#include <iostream>
#include <vector>
#include <cstring>

using namespace cira::runtime;

// Example graph edge structure
struct Edge {
    uint32_t from;
    uint32_t to;
    float weight;
};

// Example node structure
struct Node {
    float value;
    uint32_t degree;
};

// This is what the compiler backend would generate from MLIR
int main() {
    std::cout << "=== Cira Runtime Example (Generated Code) ===\n\n";

    // Initialize runtime (generated from module initialization)
    auto runtime = CiraRuntime::create();
    auto* offload_engine = runtime->getOffloadEngine();
    auto* memory_manager = runtime->getMemoryManager();

    runtime->setVerbosity(1);
    runtime->enableProfiling(true);

    // Allocate graph data (generated from allocation ops)
    const size_t num_edges = 1000;
    const size_t num_nodes = 100;

    std::cout << "Allocating memory for " << num_edges << " edges and "
              << num_nodes << " nodes\n";

    auto* edge_data = memory_manager->allocateRemote(
        sizeof(Edge) * num_edges,
        MemoryTier::CXL_ATTACHED
    );

    auto* node_data = memory_manager->allocateRemote(
        sizeof(Node) * num_nodes,
        MemoryTier::LOCAL_DRAM
    );

    // Initialize data
    Edge* edges = static_cast<Edge*>(edge_data->base_addr);
    Node* nodes = static_cast<Node*>(node_data->base_addr);

    for (size_t i = 0; i < num_edges; i++) {
        edges[i].from = i % num_nodes;
        edges[i].to = (i + 1) % num_nodes;
        edges[i].weight = 1.0f / (i + 1);
    }

    for (size_t i = 0; i < num_nodes; i++) {
        nodes[i].value = 1.0f;
        nodes[i].degree = 10;
    }

    std::cout << "\nProcessing graph edges with offload operations...\n";

    // Generated from:
    // scf.for %i = %c0 to %num_edges step %cache_line_size {
    const size_t cache_line_size = 8;
    const size_t prefetch_distance = 2;

    for (size_t i = 0; i < num_edges; i += cache_line_size) {
        // Generated from: cira.offload.load_edge(%edge_data, %i, %prefetch_distance)
        void* prefetch_ptr = offload_engine->loadEdge(
            edge_data, i, prefetch_distance
        );

        // scf.for %j = %c0 to %cache_line_size {
        for (size_t j = 0; j < cache_line_size && (i + j) < num_edges; j++) {
            // Generated from: %edge = cira.offload.load_edge(%edge_data, %idx, %c0)
            Edge* edge = static_cast<Edge*>(
                offload_engine->loadEdge(edge_data, i + j, 0)
            );

            // For this simple test, just use the node indices from the edge
            // In a real implementation, loadNode would return actual node pointers
            if (edge->from < num_nodes && edge->to < num_nodes) {
                Node* from_node = &nodes[edge->from];
                Node* to_node = &nodes[edge->to];

                // Generated from: %paddr_from = cira.offload.get_paddr("from", %from)
                uintptr_t paddr_from = offload_engine->getPhysicalAddr("from", from_node);

                // Generated from: %paddr_to = cira.offload.get_paddr("to", %to)
                uintptr_t paddr_to = offload_engine->getPhysicalAddr("to", to_node);

                // Simple update operation using the actual node data
                to_node->value += from_node->value * edge->weight;
            }
        }

        // Generated from: cira.offload.evict_edge(%edge_data, %i)
        offload_engine->evictEdge(edge_data, i);
    }

    std::cout << "\nProcessing complete!\n";

    // Print memory statistics
    std::cout << "\nMemory Statistics:\n";
    std::cout << "  CXL Memory Used: "
              << memory_manager->getUsedMemory(MemoryTier::CXL_ATTACHED) / 1024
              << " KB\n";
    std::cout << "  Local DRAM Used: "
              << memory_manager->getUsedMemory(MemoryTier::LOCAL_DRAM) / 1024
              << " KB\n";

    // Verify results
    float total_value = 0;
    for (size_t i = 0; i < num_nodes; i++) {
        total_value += nodes[i].value;
    }
    std::cout << "\nTotal node value after processing: " << total_value << "\n";

    // Cleanup
    memory_manager->deallocateRemote(edge_data);
    memory_manager->deallocateRemote(node_data);

    std::cout << "\n✅ Runtime test completed successfully!\n";

    return 0;
}