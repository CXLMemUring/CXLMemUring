#include <stdlib.h>

// Simple graph edge structure
typedef struct {
    int from;
    int to;
    float weight;
} Edge;

// Graph traversal function
void traverse_edges(Edge* edges, int num_edges, float* results) {
    for (int i = 0; i < num_edges; i++) {
        // Simple computation on edge data
        results[i] = edges[i].weight * 2.0f;
        
        // Simulate neighbor access pattern
        if (i + 1 < num_edges) {
            results[i] += edges[i + 1].weight * 0.5f;
        }
    }
}

// Matrix multiplication-like pattern
void matrix_compute(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

int main() {
    const int n = 100;
    Edge* edges = (Edge*)malloc(n * sizeof(Edge));
    float* results = (float*)malloc(n * sizeof(float));
    
    traverse_edges(edges, n, results);
    
    free(edges);
    free(results);
    return 0;
}
