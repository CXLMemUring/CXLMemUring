// Simple graph traversal without system headers
typedef struct {
    int from;
    int to;
    float weight;
} Edge;

void process_edges(Edge* edges, float* output, int count) {
    for (int i = 0; i < count; i++) {
        // Graph access pattern with indirection
        output[edges[i].to] = edges[i].weight * 2.0f;
        
        // Neighbor access simulation
        if (i > 0) {
            output[edges[i].from] += edges[i-1].weight;
        }
    }
}

float sum_array(float* arr, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}
