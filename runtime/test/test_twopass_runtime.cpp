// Test program for the two-pass execution runtime
// Demonstrates the profiling and timing injection workflow

#include "../vortex_verilator_sim.h"
#include "../offload_profiler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

// Simulated kernel binary (would be actual RISC-V binary in real use)
static const uint8_t dummy_kernel[] = {
    0x13, 0x00, 0x00, 0x00,  // nop
    0x73, 0x00, 0x10, 0x00   // ebreak
};

// Simulated linked list node
struct Node {
    int64_t data;
    struct Node* next;
};

// Create a linked list for testing
static struct Node* create_linked_list(size_t num_nodes) {
    struct Node* head = NULL;
    struct Node* prev = NULL;

    for (size_t i = 0; i < num_nodes; i++) {
        struct Node* node = (struct Node*)malloc(sizeof(struct Node));
        node->data = (int64_t)i * 100;
        node->next = NULL;

        if (prev) {
            prev->next = node;
        } else {
            head = node;
        }
        prev = node;
    }

    return head;
}

// Free linked list
static void free_linked_list(struct Node* head) {
    while (head) {
        struct Node* next = head->next;
        free(head);
        head = next;
    }
}

// Simulate host independent work (could run in parallel with Vortex)
static int64_t do_independent_work(size_t work_units) {
    int64_t result = 0;
    for (size_t i = 0; i < work_units; i++) {
        // Some computation that doesn't depend on offloaded data
        result += (i * 17) % 1000;
    }
    return result;
}

// Test Pass 1: Profiling
static int test_profiling_pass(const char* profile_output) {
    printf("\n=== Test Pass 1: Profiling ===\n\n");

    // Initialize two-pass context
    twopass_context_t ctx;
    int ret = twopass_init(&ctx, NULL);  // NULL = estimation mode
    if (ret != 0) {
        fprintf(stderr, "Failed to initialize two-pass context\n");
        return -1;
    }

    // Set profiling mode
    twopass_set_profiling_mode(&ctx);

    // Register offload regions
    uint32_t region_list_traverse = twopass_register_region(&ctx, "linked_list_traverse");
    uint32_t region_batch_process = twopass_register_region(&ctx, "batch_process");

    printf("Registered regions:\n");
    printf("  [%u] linked_list_traverse\n", region_list_traverse);
    printf("  [%u] batch_process\n", region_batch_process);

    // Create test data
    struct Node* list = create_linked_list(1000);

    //==========================================================================
    // Test Region 1: Linked List Traversal
    //==========================================================================
    printf("\n--- Region 1: Linked List Traversal ---\n");

    // Mark start of host independent work (at dominator point)
    twopass_host_work_start(&ctx, region_list_traverse);

    // Launch kernel (runs Verilator simulation in parallel)
    ret = twopass_launch_kernel(&ctx, region_list_traverse,
                                dummy_kernel, sizeof(dummy_kernel),
                                NULL, 0);
    if (ret != 0) {
        fprintf(stderr, "Failed to launch kernel for region %u\n", region_list_traverse);
    }

    // Do independent work while Vortex simulation runs
    int64_t independent_result = do_independent_work(10000);
    printf("  Independent work result: %ld\n", independent_result);

    // Mark end of host independent work
    twopass_host_work_end(&ctx, region_list_traverse);

    // Synchronization point - compute injection delay
    twopass_sync_point(&ctx, region_list_traverse);

    //==========================================================================
    // Test Region 2: Batch Processing
    //==========================================================================
    printf("\n--- Region 2: Batch Processing ---\n");

    twopass_host_work_start(&ctx, region_batch_process);

    ret = twopass_launch_kernel(&ctx, region_batch_process,
                                dummy_kernel, sizeof(dummy_kernel),
                                NULL, 0);

    // Less independent work for this region
    independent_result = do_independent_work(1000);
    printf("  Independent work result: %ld\n", independent_result);

    twopass_host_work_end(&ctx, region_batch_process);
    twopass_sync_point(&ctx, region_batch_process);

    //==========================================================================
    // Save profile data
    //==========================================================================
    ret = twopass_save_profile(&ctx, profile_output);
    if (ret != 0) {
        fprintf(stderr, "Failed to save profile to %s\n", profile_output);
    }

    // Generate compiler annotations
    char annotations_path[256];
    snprintf(annotations_path, sizeof(annotations_path), "%s.annotations.h", profile_output);
    twopass_annotate_dominator_tree(&ctx, annotations_path);

    // Cleanup
    free_linked_list(list);
    twopass_destroy(&ctx);

    printf("\n=== Profiling Pass Complete ===\n");
    return 0;
}

// Test Pass 2: Timing Injection
static int test_injection_pass(const char* profile_input) {
    printf("\n=== Test Pass 2: Timing Injection ===\n\n");

    // Initialize two-pass context
    twopass_context_t ctx;
    int ret = twopass_init(&ctx, NULL);
    if (ret != 0) {
        fprintf(stderr, "Failed to initialize two-pass context\n");
        return -1;
    }

    // Load profile and set injection mode
    ret = twopass_set_injection_mode(&ctx, profile_input);
    if (ret != 0) {
        fprintf(stderr, "Failed to load profile from %s\n", profile_input);
        twopass_destroy(&ctx);
        return -1;
    }

    // Create test data
    struct Node* list = create_linked_list(1000);

    //==========================================================================
    // Execute with timing injection
    //==========================================================================
    printf("\n--- Executing with timing injection ---\n");

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Region 1: Linked List Traversal
    // In real code, this would be the actual application logic
    printf("  Processing region 0 (linked_list_traverse)...\n");

    int64_t sum = 0;
    struct Node* node = list;
    while (node) {
        sum += node->data;
        node = node->next;
    }

    // Inject timing delay at sync point
    twopass_sync_point(&ctx, 0);  // region_id = 0

    // Region 2: Batch Processing
    printf("  Processing region 1 (batch_process)...\n");

    int64_t batch_result = do_independent_work(5000);

    twopass_sync_point(&ctx, 1);  // region_id = 1

    clock_gettime(CLOCK_MONOTONIC, &end);

    //==========================================================================
    // Report results
    //==========================================================================
    uint64_t total_ns = (end.tv_sec - start.tv_sec) * 1000000000ULL +
                        (end.tv_nsec - start.tv_nsec);

    printf("\n--- Results ---\n");
    printf("  List sum: %ld\n", sum);
    printf("  Batch result: %ld\n", batch_result);
    printf("  Total execution time: %.3f ms\n", total_ns / 1e6);

    // Cleanup
    free_linked_list(list);
    twopass_destroy(&ctx);

    printf("\n=== Injection Pass Complete ===\n");
    return 0;
}

// Test CXL latency model
static void test_latency_model() {
    printf("\n=== Testing CXL Latency Model ===\n\n");

    cxl_latency_model_t model;
    cxl_latency_model_init(&model);

    printf("Initial model parameters:\n");
    printf("  Base latency: %lu ns\n", model.base_latency_ns);
    printf("  Protocol overhead: %lu ns\n", model.protocol_overhead_ns);
    printf("  Queue delay: %lu ns\n", model.queue_delay_ns);
    printf("  Contention factor: %lu\n", model.contention_factor);

    // Test various access patterns
    printf("\nLatency calculations:\n");

    uint64_t lat_64 = cxl_latency_model_calculate(&model, 64, false);
    printf("  64B random access: %lu ns\n", lat_64);

    uint64_t lat_4k_seq = cxl_latency_model_calculate(&model, 4096, true);
    printf("  4KB sequential access: %lu ns\n", lat_4k_seq);

    uint64_t lat_4k_rand = cxl_latency_model_calculate(&model, 4096, false);
    printf("  4KB random access: %lu ns\n", lat_4k_rand);

    // Simulate contention
    printf("\nSimulating contention updates:\n");
    for (int i = 0; i < 5; i++) {
        // Observed higher latency than expected
        cxl_latency_model_update(&model, 250, 64);
        printf("  After high-latency observation %d: contention=%lu\n",
               i + 1, model.contention_factor);
    }
}

// Test cycle/time conversion
static void test_time_conversion() {
    printf("\n=== Testing Time Conversion ===\n\n");

    double freq_mhz = 200.0;  // Vortex at 200MHz

    printf("At %.0f MHz:\n", freq_mhz);

    uint64_t cycles[] = {1, 100, 1000, 10000, 100000, 1000000};
    for (int i = 0; i < 6; i++) {
        uint64_t ns = cycles_to_ns(cycles[i], freq_mhz);
        uint64_t back = ns_to_cycles(ns, freq_mhz);
        printf("  %lu cycles -> %lu ns -> %lu cycles\n", cycles[i], ns, back);
    }
}

int main(int argc, char** argv) {
    printf("========================================\n");
    printf("Two-Pass Execution Runtime Test\n");
    printf("========================================\n");

    const char* profile_path = "twopass_profile.json";

    if (argc > 1) {
        profile_path = argv[1];
    }

    // Test helper functions
    test_time_conversion();
    test_latency_model();

    // Run Pass 1: Profiling
    int ret = test_profiling_pass(profile_path);
    if (ret != 0) {
        fprintf(stderr, "Profiling pass failed\n");
        return 1;
    }

    // Run Pass 2: Timing Injection
    ret = test_injection_pass(profile_path);
    if (ret != 0) {
        fprintf(stderr, "Injection pass failed\n");
        return 1;
    }

    printf("\n========================================\n");
    printf("All tests completed successfully!\n");
    printf("========================================\n");

    return 0;
}
