#include "CiraRuntime.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

using namespace cira::runtime;
using namespace std::chrono;

struct Edge {
    uint32_t from;
    uint32_t to;
    float weight;
};

struct Node {
    float value;
    uint32_t degree;
    float pagerank;
};

class GraphProcessor {
private:
    CiraRuntime* runtime;
    OffloadEngine* offload_engine;
    RemoteMemoryManager* memory_manager;
    PrefetchController* prefetch_controller;

    RemoteMemRef* edge_data;
    RemoteMemRef* node_data;
    size_t num_edges;
    size_t num_nodes;

public:
    GraphProcessor(size_t edges, size_t nodes)
        : num_edges(edges), num_nodes(nodes) {
        runtime = CiraRuntime::create().release();
        offload_engine = runtime->getOffloadEngine();
        memory_manager = runtime->getMemoryManager();
        prefetch_controller = runtime->getPrefetchController();

        // Configure runtime
        runtime->setVerbosity(1);
        runtime->enableProfiling(true);
    }

    ~GraphProcessor() {
        if (edge_data) memory_manager->deallocateRemote(edge_data);
        if (node_data) memory_manager->deallocateRemote(node_data);
        delete runtime;
    }

    void initializeGraph() {
        std::cout << "Initializing graph with " << num_edges << " edges and "
                  << num_nodes << " nodes\n";

        // Allocate edge data in CXL memory
        edge_data = memory_manager->allocateRemote(
            sizeof(Edge) * num_edges,
            MemoryTier::CXL_ATTACHED
        );

        // Allocate node data in local DRAM for faster access
        node_data = memory_manager->allocateRemote(
            sizeof(Node) * num_nodes,
            MemoryTier::LOCAL_DRAM
        );

        // Initialize edges with random connections
        Edge* edges = static_cast<Edge*>(edge_data->base_addr);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> node_dist(0, num_nodes - 1);
        std::uniform_real_distribution<> weight_dist(0.1, 1.0);

        for (size_t i = 0; i < num_edges; i++) {
            edges[i].from = node_dist(gen);
            edges[i].to = node_dist(gen);
            edges[i].weight = weight_dist(gen);
        }

        // Initialize nodes
        Node* nodes = static_cast<Node*>(node_data->base_addr);
        for (size_t i = 0; i < num_nodes; i++) {
            nodes[i].value = 1.0f / num_nodes;
            nodes[i].degree = 0;
            nodes[i].pagerank = 1.0f / num_nodes;
        }

        // Calculate node degrees
        for (size_t i = 0; i < num_edges; i++) {
            nodes[edges[i].from].degree++;
        }
    }

    void runPageRank(int iterations = 10) {
        std::cout << "Running PageRank for " << iterations << " iterations\n";

        const float damping = 0.85f;
        const size_t cache_line_size = 8;
        const size_t prefetch_distance = 2;

        Node* nodes = static_cast<Node*>(node_data->base_addr);

        // Set access profile for optimization
        AccessProfile profile;
        profile.pattern = AccessProfile::SEQUENTIAL;
        profile.stride = sizeof(Edge) * cache_line_size;
        profile.working_set_size = num_edges * sizeof(Edge);
        profile.temporal_locality = 0.8;
        prefetch_controller->updateProfile(profile);

        auto start = high_resolution_clock::now();

        for (int iter = 0; iter < iterations; iter++) {
            // Reset pagerank values
            std::vector<float> new_pagerank(num_nodes, (1.0f - damping) / num_nodes);

            // Process edges in cache-line-sized chunks
            for (size_t i = 0; i < num_edges; i += cache_line_size) {
                // Prefetch next cache line
                if (i + cache_line_size < num_edges) {
                    offload_engine->loadEdge(edge_data, i + cache_line_size, prefetch_distance);
                }

                // Process current cache line
                size_t batch_size = std::min(cache_line_size, num_edges - i);
                std::vector<void*> edge_batch(batch_size);
                offload_engine->batchLoadEdges(edge_data, i, batch_size, edge_batch.data());

                for (size_t j = 0; j < batch_size; j++) {
                    Edge* edge = static_cast<Edge*>(edge_batch[j]);

                    // Load node data with prefetching
                    Node* from_node = static_cast<Node*>(
                        offload_engine->loadNode(edge, "from", 1)
                    );
                    Node* to_node = static_cast<Node*>(
                        offload_engine->loadNode(edge, "to", 1)
                    );

                    // PageRank computation
                    if (from_node->degree > 0) {
                        float contribution = damping * from_node->pagerank / from_node->degree;
                        new_pagerank[edge->to] += contribution * edge->weight;
                    }
                }

                // Evict processed cache line
                offload_engine->evictEdge(edge_data, i);
            }

            // Update pagerank values
            for (size_t i = 0; i < num_nodes; i++) {
                nodes[i].pagerank = new_pagerank[i];
            }

            if ((iter + 1) % 2 == 0) {
                std::cout << "  Iteration " << (iter + 1) << " completed\n";
            }
        }

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);

        std::cout << "PageRank completed in " << duration.count() << " ms\n";

        // Print top nodes
        std::vector<std::pair<float, size_t>> ranked_nodes;
        for (size_t i = 0; i < num_nodes; i++) {
            ranked_nodes.push_back({nodes[i].pagerank, i});
        }
        std::sort(ranked_nodes.rbegin(), ranked_nodes.rend());

        std::cout << "\nTop 10 nodes by PageRank:\n";
        for (size_t i = 0; i < std::min(size_t(10), num_nodes); i++) {
            std::cout << "  Node " << ranked_nodes[i].second
                      << ": " << ranked_nodes[i].first << "\n";
        }
    }

    void printMemoryStats() {
        std::cout << "\nMemory Statistics:\n";
        std::cout << "  CXL Memory Used: "
                  << memory_manager->getUsedMemory(MemoryTier::CXL_ATTACHED) / (1024*1024)
                  << " MB\n";
        std::cout << "  Local DRAM Used: "
                  << memory_manager->getUsedMemory(MemoryTier::LOCAL_DRAM) / (1024*1024)
                  << " MB\n";
    }
};

int main(int argc, char** argv) {
    size_t num_edges = 100000;
    size_t num_nodes = 10000;
    int iterations = 10;

    if (argc >= 3) {
        num_edges = std::stoull(argv[1]);
        num_nodes = std::stoull(argv[2]);
    }
    if (argc >= 4) {
        iterations = std::stoi(argv[3]);
    }

    std::cout << "Cira Graph Processing Example\n";
    std::cout << "==============================\n";

    GraphProcessor processor(num_edges, num_nodes);

    processor.initializeGraph();
    processor.runPageRank(iterations);
    processor.printMemoryStats();

    return 0;
}