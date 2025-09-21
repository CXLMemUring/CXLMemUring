// Example C++ code that can be processed through ClangIR -> Cira pipeline
// Usage: clangir-opt -cir-mlir-scf-prepare -cir-to-mlir example.cir |
//        cira-opt -cir-to-cira -tosa-to-cira |
//        cira-opt -cira-to-llvm

#include <vector>
#include <cstddef>

// Graph data structures
struct Node {
    float value;
    int degree;
    int* neighbors;
};

struct Edge {
    int from;
    int to;
    float weight;
};

// Example 1: Large matrix multiplication - candidate for CXL offloading
void large_matrix_multiply(float* A, float* B, float* C, size_t n) {
    // This will be converted to cira.offload "matmul" through ClangIR
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// Example 2: Graph traversal - pointer chasing pattern
float graph_bfs_sum(Node* graph, int start, int num_nodes) {
    // This will be converted to cira.offload "pointer_chase"
    std::vector<bool> visited(num_nodes, false);
    std::vector<int> queue;
    queue.push_back(start);
    visited[start] = true;

    float total_value = 0.0f;

    while (!queue.empty()) {
        int current = queue.front();
        queue.erase(queue.begin());

        // Process current node - irregular memory access
        total_value += graph[current].value;

        // Add neighbors to queue - pointer chasing
        for (int i = 0; i < graph[current].degree; i++) {
            int neighbor = graph[current].neighbors[i];
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                queue.push_back(neighbor);
            }
        }
    }

    return total_value;
}

// Example 3: Edge processing with streaming access
void process_edges_streaming(Edge* edges, Node* nodes, size_t num_edges) {
    // Large edge array with sequential access - candidate for streaming offload
    for (size_t i = 0; i < num_edges; i++) {
        Edge& edge = edges[i];

        // Update nodes based on edge weights
        nodes[edge.from].value += edge.weight * 0.1f;
        nodes[edge.to].value += edge.weight * 0.1f;
    }
}

// Example 4: Nested loop with regular access pattern
void conv2d_like_operation(float* input, float* output, int height, int width, int channels) {
    // Regular nested access pattern - moderate offload benefit
    for (int h = 0; h < height - 2; h++) {
        for (int w = 0; w < width - 2; w++) {
            for (int c = 0; c < channels; c++) {
                float sum = 0.0f;

                // 3x3 convolution kernel
                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        int idx = ((h + kh) * width + (w + kw)) * channels + c;
                        sum += input[idx] * 0.111f; // Simple average filter
                    }
                }

                int out_idx = (h * (width - 2) + w) * channels + c;
                output[out_idx] = sum;
            }
        }
    }
}

// Example 5: Small operation that should stay local
void small_vector_add(float* a, float* b, float* c, size_t n) {
    // Small operations remain local for efficiency
    for (size_t i = 0; i < n && i < 1000; i++) {
        c[i] = a[i] + b[i];
    }
}

// Example 6: Reduction operation - streaming remote access
float large_reduction(float* data, size_t n) {
    // Large reduction with streaming pattern
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
}

// Example 7: Sparse matrix operations
void sparse_matrix_vector_multiply(int* row_ptr, int* col_idx, float* values,
                                 float* x, float* y, int num_rows) {
    // Irregular access pattern with potential for remote memory
    for (int i = 0; i < num_rows; i++) {
        float sum = 0.0f;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            sum += values[j] * x[col_idx[j]];
        }
        y[i] = sum;
    }
}

// Main function demonstrating the integration
int main() {
    const size_t LARGE_SIZE = 2048;
    const size_t SMALL_SIZE = 64;

    // Allocate test data
    std::vector<float> A(LARGE_SIZE * LARGE_SIZE, 1.0f);
    std::vector<float> B(LARGE_SIZE * LARGE_SIZE, 2.0f);
    std::vector<float> C(LARGE_SIZE * LARGE_SIZE, 0.0f);

    // Large operation - will be offloaded to CXL memory
    large_matrix_multiply(A.data(), B.data(), C.data(), LARGE_SIZE);

    // Small operation - stays local
    std::vector<float> small_a(SMALL_SIZE, 1.0f);
    std::vector<float> small_b(SMALL_SIZE, 2.0f);
    std::vector<float> small_c(SMALL_SIZE, 0.0f);
    small_vector_add(small_a.data(), small_b.data(), small_c.data(), SMALL_SIZE);

    return 0;
}

/*
Expected ClangIR to Cira transformation:

1. ClangIR Phase:
   - C++ code -> ClangIR dialect (cir.for, cir.while, cir.load, cir.store)
   - clangir-opt -cir-mlir-scf-prepare -cir-to-mlir

2. Cira Analysis Phase:
   - Detect graph processing patterns
   - Analyze memory access patterns
   - Estimate offloading benefits

3. Cira Offload Generation:
   - large_matrix_multiply -> cira.offload "matmul" with CXL memory
   - graph_bfs_sum -> cira.offload "pointer_chase" with prefetching
   - process_edges_streaming -> cira.offload "streaming" with far memory
   - small operations remain as standard MLIR operations

4. Runtime Integration:
   - Generated code uses CiraRuntime for remote memory management
   - Automatic memory tier selection (LOCAL_DRAM, CXL_ATTACHED, FAR_MEMORY)
   - Optimized access patterns with prefetching and eviction hints
*/