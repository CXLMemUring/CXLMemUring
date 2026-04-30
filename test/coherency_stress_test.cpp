// coherency_stress_test.cpp — CXL cache coherency race condition tests
//
// Tests the DCOH write-back + host cache coherency model described in the
// CIRA paper. Each test exercises a potential race condition between the
// CXL Type 2 device (Vortex) and the host CPU.
//
// Andi's insight: "an eviction of a cxlmemuring-based prefetch should result
// in a CXL memory access (performance hit) rather than a segfault."
//
// These tests are inherently non-deterministic — run many iterations to
// surface race conditions. Use: ./coherency_stress_test --iterations=10000
//
// Build:
//   g++ -O2 -march=native -pthread -std=c++17 \
//       -I../runtime/include \
//       -o coherency_stress_test coherency_stress_test.cpp \
//       -lnuma -lpthread
//
// Run on CXL hardware:
//   numactl -m 2 ./coherency_stress_test --iterations=10000
//   # or with LLC pollution:
//   ./coherency_stress_test --iterations=10000 --pollute-llc

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <thread>
#include <vector>

#ifdef __x86_64__
#include <immintrin.h>
#include <x86intrin.h>
#endif

// =============================================================================
// CompletionData — matches kernel_gpu.cpp and CiraRuntime.cpp layout
// =============================================================================
struct alignas(64) CompletionData {
    volatile uint32_t magic;      // 0xDEADBEEF when done
    volatile uint32_t status;     // 0 = success
    volatile uint64_t result;     // kernel-specific result
    volatile uint64_t cycles;     // execution cycles
    volatile uint64_t timestamp;  // completion timestamp
    uint8_t reserved[32];         // pad to 64 bytes (one cache line)
};
static_assert(sizeof(CompletionData) == 64, "CompletionData must be one cache line");

#define COMPLETION_MAGIC 0xDEADBEEF
#define CACHELINE_SIZE   64

// =============================================================================
// Linked list node for chain-chase tests
// =============================================================================
struct alignas(64) ChainNode {
    ChainNode* next;              // offset 0: next pointer
    uint64_t   data;              // offset 8: payload
    uint64_t   sequence;          // offset 16: monotonic sequence number
    uint64_t   checksum;          // offset 24: simple checksum for corruption detect
    uint8_t    pad[32];           // pad to 64 bytes
};
static_assert(sizeof(ChainNode) == 64, "ChainNode must be one cache line");

// =============================================================================
// Globals
// =============================================================================
static int g_iterations = 1000;
static bool g_pollute_llc = false;
static bool g_verbose = false;
static int g_num_threads = 4;

// Statistics
struct TestStats {
    std::atomic<uint64_t> pass{0};
    std::atomic<uint64_t> fail{0};
    std::atomic<uint64_t> corruption{0};
    std::atomic<uint64_t> eviction_refetch{0};  // performance: data was re-fetched from CXL
    std::atomic<uint64_t> total_latency_ns{0};
};

// =============================================================================
// Utility functions
// =============================================================================

static inline uint64_t rdtsc_fenced() {
#ifdef __x86_64__
    unsigned int lo, hi;
    __asm__ volatile("lfence; rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
#else
    return 0;
#endif
}

// Allocate cache-line-aligned memory
static void* alloc_aligned(size_t size) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, CACHELINE_SIZE, size) != 0) return nullptr;
    memset(ptr, 0, size);
    return ptr;
}

// Simulate DCOH writeback: device writes to host-visible address.
// In real hardware, this goes through CXL.cache DCOH protocol.
// Here we simulate with a store + optional delay to model CXL latency.
static void simulate_dcoh_writeback(volatile void* addr, const void* data,
                                     size_t size, uint32_t delay_ns = 165) {
    // Model CXL write latency
    auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start <
           std::chrono::nanoseconds(delay_ns)) {
#ifdef __x86_64__
        _mm_pause();
#endif
    }

    // Atomic store of data (models DCOH cache line push)
    memcpy(const_cast<void*>(addr), data, size);

    // DCOH ensures the cache line is visible in host LLC
    // Model with a store fence
#ifdef __x86_64__
    _mm_sfence();
#else
    __atomic_thread_fence(__ATOMIC_RELEASE);
#endif
}

// Pollute LLC to force evictions of target cachelines.
// Allocates and touches enough data to overflow the LLC.
static void pollute_llc(size_t llc_size_bytes = 48 * 1024 * 1024) {
    // Allocate 2x LLC size to ensure eviction
    size_t pollution_size = llc_size_bytes * 2;
    volatile char* pollution = (volatile char*)malloc(pollution_size);
    if (!pollution) return;

    // Sequential scan to fill LLC
    for (size_t i = 0; i < pollution_size; i += CACHELINE_SIZE) {
        pollution[i] = (char)(i & 0xFF);
    }

    // Read back to ensure lines are resident
    volatile char sink = 0;
    for (size_t i = 0; i < pollution_size; i += CACHELINE_SIZE) {
        sink += pollution[i];
    }
    (void)sink;
    free(const_cast<char*>(pollution));
}

// Build a linked list of ChainNodes
static ChainNode* build_chain(size_t depth) {
    ChainNode* head = (ChainNode*)alloc_aligned(sizeof(ChainNode) * depth);
    if (!head) return nullptr;

    for (size_t i = 0; i < depth; ++i) {
        head[i].next = (i + 1 < depth) ? &head[i + 1] : nullptr;
        head[i].data = 0x1000 + i;
        head[i].sequence = i;
        head[i].checksum = head[i].data ^ head[i].sequence ^ 0xCAFEBABE;
    }
    return head;
}

static bool verify_node(const ChainNode* node) {
    return (node->checksum == (node->data ^ node->sequence ^ 0xCAFEBABE));
}

// =============================================================================
// TEST 1: DCOH Completion Race (Fence semantics — like D flip-flop)
//
// The CompletionData magic field acts as a "fence" — like a D flip-flop in
// hardware. The invariant is:
//   IF magic == 0xDEADBEEF THEN all prior writes are visible to the host.
//
// The "device" thread writes payload data, then writes magic.
// The "host" thread spins on magic, then reads payload.
// Race: can the host see magic=0xDEADBEEF but stale/zero payload?
//
// This is the fundamental store-ordering test for DCOH.
// =============================================================================
void test_dcoh_completion_race(TestStats& stats) {
    printf("\n=== TEST 1: DCOH Completion Race (D flip-flop fence) ===\n");
    printf("  Invariant: magic==0xDEADBEEF implies all prior writes visible\n");
    printf("  Iterations: %d\n\n", g_iterations);

    for (int iter = 0; iter < g_iterations; ++iter) {
        CompletionData* comp = (CompletionData*)alloc_aligned(sizeof(CompletionData));
        if (!comp) { stats.fail++; continue; }

        // Also allocate a separate data buffer (simulating the LLC tile)
        volatile uint64_t* data_buf = (volatile uint64_t*)alloc_aligned(CACHELINE_SIZE * 16);
        if (!data_buf) { free(comp); stats.fail++; continue; }

        const uint64_t EXPECTED_VALUE = 0xAAAABBBBCCCCDDDDULL + iter;

        std::atomic<bool> device_started{false};

        // "Device" thread — simulates Vortex DCOH writeback
        std::thread device_thread([&]() {
            device_started.store(true, std::memory_order_release);

            // Step 1: Write payload data to LLC tile buffer
            for (int i = 0; i < 16; ++i) {
                data_buf[i * 8] = EXPECTED_VALUE + i;  // each cache line
            }

            // Step 2: Fence — all data writes must be visible before magic
            // In real hardware, DCOH provides this ordering.
            // The magic write to the CompletionData cache line is the "fence output"
            // — like the Q output of a D flip-flop on the clock edge.
            __atomic_thread_fence(__ATOMIC_RELEASE);

            // Step 3: Write magic (DCOH pushes this cache line to host LLC)
            comp->result = EXPECTED_VALUE;
            comp->status = 0;
            __atomic_thread_fence(__ATOMIC_RELEASE);
            comp->magic = COMPLETION_MAGIC;
        });

        // Wait for device thread to start
        while (!device_started.load(std::memory_order_acquire)) {
#ifdef __x86_64__
            _mm_pause();
#endif
        }

        // "Host" thread — spin on magic, then check payload
        // This is what cira_future_await() does
        while (comp->magic != COMPLETION_MAGIC) {
#ifdef __x86_64__
            _mm_pause();
#endif
        }

        // Acquire fence — match the release fence in the device thread
        __atomic_thread_fence(__ATOMIC_ACQUIRE);

        // Check: all data written before magic must now be visible
        bool data_ok = true;
        for (int i = 0; i < 16; ++i) {
            uint64_t val = data_buf[i * 8];
            if (val != EXPECTED_VALUE + (uint64_t)i) {
                if (g_verbose) {
                    printf("  FAIL iter=%d: data_buf[%d]=%lx expected=%lx\n",
                           iter, i, (unsigned long)val,
                           (unsigned long)(EXPECTED_VALUE + i));
                }
                data_ok = false;
                break;
            }
        }

        if (comp->result != EXPECTED_VALUE) {
            if (g_verbose) {
                printf("  FAIL iter=%d: result=%lx expected=%lx\n",
                       iter, (unsigned long)comp->result,
                       (unsigned long)EXPECTED_VALUE);
            }
            data_ok = false;
        }

        if (data_ok) {
            stats.pass++;
        } else {
            stats.fail++;
            stats.corruption++;
        }

        device_thread.join();
        free(const_cast<uint64_t*>(data_buf));
        free(comp);
    }
}

// =============================================================================
// TEST 2: Cacheline Eviction During Prefetch
//
// Scenario:
//   1. "Device" installs cachelines into host LLC (via prefetch/DCOH)
//   2. "Polluter" thread fills LLC with unrelated data, evicting installed lines
//   3. "Host" reads the installed data
//
// Expected: host reads correct data (re-fetched from CXL memory if evicted)
// Bug: host reads stale/zero data (coherency failure)
//
// This is the race Vickie identified: "the latter cache line is possibly
// overwritten by someone else when writing back from device"
// =============================================================================
void test_cacheline_eviction_during_prefetch(TestStats& stats) {
    printf("\n=== TEST 2: Cacheline Eviction During Prefetch ===\n");
    printf("  Scenario: device installs -> LLC pollution -> host reads\n");
    printf("  Iterations: %d, LLC pollution: %s\n\n",
           g_iterations, g_pollute_llc ? "ON" : "OFF");

    const size_t NUM_CACHELINES = 64;  // 4KB of installed data
    const size_t BUF_SIZE = NUM_CACHELINES * CACHELINE_SIZE;

    for (int iter = 0; iter < g_iterations; ++iter) {
        // Allocate the "CXL memory" source and "LLC tile" destination
        volatile uint8_t* cxl_src = (volatile uint8_t*)alloc_aligned(BUF_SIZE);
        volatile uint8_t* llc_tile = (volatile uint8_t*)alloc_aligned(BUF_SIZE);
        CompletionData* comp = (CompletionData*)alloc_aligned(sizeof(CompletionData));
        if (!cxl_src || !llc_tile || !comp) {
            free(const_cast<uint8_t*>(cxl_src));
            free(const_cast<uint8_t*>(llc_tile));
            free(comp);
            stats.fail++;
            continue;
        }

        // Fill CXL source with known pattern
        for (size_t i = 0; i < BUF_SIZE; ++i) {
            cxl_src[i] = (uint8_t)((i + iter) & 0xFF);
        }

        std::atomic<int> phase{0};

        // "Device" thread — copy from CXL source to LLC tile (DCOH writeback)
        std::thread device_thread([&]() {
            // Simulate device reading from CXL memory and writing to host-visible
            // LLC tile via DCOH
            for (size_t cl = 0; cl < NUM_CACHELINES; ++cl) {
                size_t off = cl * CACHELINE_SIZE;
                // Copy one cache line at a time (models DCOH per-cacheline push)
                memcpy(const_cast<uint8_t*>(&llc_tile[off]),
                       const_cast<uint8_t*>(&cxl_src[off]),
                       CACHELINE_SIZE);
            }

            __atomic_thread_fence(__ATOMIC_RELEASE);
            comp->status = 0;
            comp->magic = COMPLETION_MAGIC;
            phase.store(1, std::memory_order_release);
        });

        // Wait for device to finish
        while (phase.load(std::memory_order_acquire) < 1) {
#ifdef __x86_64__
            _mm_pause();
#endif
        }

        // Optionally pollute LLC to force eviction of installed cachelines
        if (g_pollute_llc) {
            pollute_llc();
        }

        // "Host" reads the LLC tile — if cachelines were evicted,
        // the coherency protocol should re-fetch from CXL memory
        auto t0 = std::chrono::steady_clock::now();

        bool data_ok = true;
        for (size_t i = 0; i < BUF_SIZE; ++i) {
            uint8_t expected = (uint8_t)((i + iter) & 0xFF);
            uint8_t actual = llc_tile[i];
            if (actual != expected) {
                if (g_verbose) {
                    printf("  FAIL iter=%d byte=%zu: got=0x%02x expected=0x%02x\n",
                           iter, i, actual, expected);
                }
                data_ok = false;
                stats.corruption++;
                break;
            }
        }

        auto t1 = std::chrono::steady_clock::now();
        uint64_t read_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        stats.total_latency_ns += read_ns;

        // If read took significantly longer with LLC pollution, the data was
        // likely re-fetched from CXL (expected behavior, not a bug)
        if (g_pollute_llc && read_ns > 5000) {  // > 5us suggests re-fetch
            stats.eviction_refetch++;
        }

        if (data_ok) {
            stats.pass++;
        } else {
            stats.fail++;
        }

        device_thread.join();
        free(const_cast<uint8_t*>(cxl_src));
        free(const_cast<uint8_t*>(llc_tile));
        free(comp);
    }
}

// =============================================================================
// TEST 3: Phase Boundary Ordering
//
// The cira.phase_boundary op acts as a full fence between computation phases.
// Invariant: all writes from phase N are visible before any read in phase N+1.
//
// Test: two device threads, each writing to their own buffer, with a phase
// barrier between them. The host reads both buffers after the barrier.
//
// This tests the "D flip-flop" analogy — the phase boundary is the clock edge.
// =============================================================================
void test_phase_boundary_ordering(TestStats& stats) {
    printf("\n=== TEST 3: Phase Boundary Ordering (Fence) ===\n");
    printf("  Invariant: phase_barrier guarantees all prior phase writes visible\n");
    printf("  Iterations: %d\n\n", g_iterations);

    for (int iter = 0; iter < g_iterations; ++iter) {
        // Two phases, each with their own data and completion
        volatile uint64_t* phase1_data = (volatile uint64_t*)alloc_aligned(CACHELINE_SIZE * 8);
        volatile uint64_t* phase2_data = (volatile uint64_t*)alloc_aligned(CACHELINE_SIZE * 8);
        CompletionData* comp1 = (CompletionData*)alloc_aligned(sizeof(CompletionData));
        CompletionData* comp2 = (CompletionData*)alloc_aligned(sizeof(CompletionData));

        if (!phase1_data || !phase2_data || !comp1 || !comp2) {
            free(const_cast<uint64_t*>(phase1_data));
            free(const_cast<uint64_t*>(phase2_data));
            free(comp1); free(comp2);
            stats.fail++;
            continue;
        }

        const uint64_t PHASE1_MARKER = 0x1111111100000000ULL + iter;
        const uint64_t PHASE2_MARKER = 0x2222222200000000ULL + iter;

        std::atomic<int> barrier{0};

        // Phase 1 device thread
        std::thread phase1_thread([&]() {
            for (int i = 0; i < 8; ++i) {
                phase1_data[i * 8] = PHASE1_MARKER + i;
            }
            __atomic_thread_fence(__ATOMIC_RELEASE);
            comp1->status = 0;
            comp1->magic = COMPLETION_MAGIC;
            barrier.fetch_add(1, std::memory_order_release);
        });

        // Phase 2 device thread — must NOT start until phase 1 completes
        std::thread phase2_thread([&]() {
            // Wait for phase barrier (simulates cira_phase_barrier)
            while (barrier.load(std::memory_order_acquire) < 1) {
#ifdef __x86_64__
                _mm_pause();
#endif
            }

            // Phase 2 may read phase 1 data (pipeline dependency)
            uint64_t phase1_val = phase1_data[0];
            (void)phase1_val;  // use it to prevent optimization

            for (int i = 0; i < 8; ++i) {
                phase2_data[i * 8] = PHASE2_MARKER + i;
            }
            __atomic_thread_fence(__ATOMIC_RELEASE);
            comp2->status = 0;
            comp2->magic = COMPLETION_MAGIC;
            barrier.fetch_add(1, std::memory_order_release);
        });

        // Host: wait for both phases
        while (barrier.load(std::memory_order_acquire) < 2) {
#ifdef __x86_64__
            _mm_pause();
#endif
        }
        __atomic_thread_fence(__ATOMIC_ACQUIRE);

        // Verify: all phase 1 data visible
        bool ok = true;
        for (int i = 0; i < 8; ++i) {
            if (phase1_data[i * 8] != PHASE1_MARKER + (uint64_t)i) {
                if (g_verbose)
                    printf("  FAIL iter=%d: phase1[%d]=%lx expected=%lx\n",
                           iter, i, (unsigned long)phase1_data[i * 8],
                           (unsigned long)(PHASE1_MARKER + i));
                ok = false;
                break;
            }
        }
        // Verify: all phase 2 data visible
        for (int i = 0; i < 8 && ok; ++i) {
            if (phase2_data[i * 8] != PHASE2_MARKER + (uint64_t)i) {
                if (g_verbose)
                    printf("  FAIL iter=%d: phase2[%d]=%lx expected=%lx\n",
                           iter, i, (unsigned long)phase2_data[i * 8],
                           (unsigned long)(PHASE2_MARKER + i));
                ok = false;
                break;
            }
        }
        // Verify: completion ordering
        if (comp1->magic != COMPLETION_MAGIC || comp2->magic != COMPLETION_MAGIC) {
            ok = false;
        }

        if (ok) stats.pass++;
        else { stats.fail++; stats.corruption++; }

        phase1_thread.join();
        phase2_thread.join();
        free(const_cast<uint64_t*>(phase1_data));
        free(const_cast<uint64_t*>(phase2_data));
        free(comp1); free(comp2);
    }
}

// =============================================================================
// TEST 4: Chain Chase + Concurrent Modification
//
// The "lost data" scenario Vickie identified:
//   1. Device chases a linked list, installing each node into host LLC
//   2. Host concurrently modifies nodes ahead in the chain
//   3. Question: does the device see the host's modification, or does the
//      host see the device's stale copy?
//
// In CXL coherency:
//   - Device reads node via CXL.cache -> gets ownership (E state)
//   - Device writes to host-visible buf via DCOH -> pushes to host LLC
//   - Host modifies the *original* node -> triggers snoop to device
//   - Device should see the modification on next read
//
// If coherency is correct: no data corruption, just ordering effects.
// If buggy: host reads stale data from a node it just modified.
// =============================================================================
void test_chain_chase_concurrent_modification(TestStats& stats) {
    printf("\n=== TEST 4: Chain Chase + Concurrent Host Modification ===\n");
    printf("  Scenario: device chases chain while host modifies nodes ahead\n");
    printf("  Iterations: %d, Chain depth: 16\n\n", g_iterations);

    const size_t CHAIN_DEPTH = 16;

    for (int iter = 0; iter < g_iterations; ++iter) {
        ChainNode* chain = build_chain(CHAIN_DEPTH);
        if (!chain) { stats.fail++; continue; }

        // Host-visible buffer for device's DCOH writeback
        volatile uint64_t* host_buf = (volatile uint64_t*)alloc_aligned(
            CHAIN_DEPTH * CACHELINE_SIZE);
        CompletionData* comp = (CompletionData*)alloc_aligned(sizeof(CompletionData));
        if (!host_buf || !comp) {
            free(chain); free(const_cast<uint64_t*>(host_buf)); free(comp);
            stats.fail++;
            continue;
        }

        std::atomic<int> device_position{0};
        std::atomic<bool> device_done{false};

        // "Device" thread — chase the chain, copy data to host buf
        std::thread device_thread([&]() {
            ChainNode* node = chain;
            for (size_t i = 0; i < CHAIN_DEPTH && node; ++i) {
                // Read node data
                uint64_t data = node->data;
                uint64_t seq = node->sequence;

                // Write to host-visible buffer (DCOH)
                host_buf[i * 8] = data;
                host_buf[i * 8 + 1] = seq;
                __atomic_thread_fence(__ATOMIC_RELEASE);

                device_position.store((int)i + 1, std::memory_order_release);

                // Chase next
                node = node->next;
            }
            comp->status = 0;
            __atomic_thread_fence(__ATOMIC_RELEASE);
            comp->magic = COMPLETION_MAGIC;
            device_done.store(true, std::memory_order_release);
        });

        // "Host" thread — concurrently modify nodes ahead of device
        std::thread host_modifier([&]() {
            // Wait for device to start chasing
            while (device_position.load(std::memory_order_acquire) < 2) {
#ifdef __x86_64__
                _mm_pause();
#endif
            }

            // Modify nodes ahead of the device's current position
            for (size_t i = CHAIN_DEPTH / 2; i < CHAIN_DEPTH; ++i) {
                chain[i].data = 0xF000 + i + iter;
                chain[i].checksum = chain[i].data ^ chain[i].sequence ^ 0xCAFEBABE;
                __atomic_thread_fence(__ATOMIC_RELEASE);
            }
        });

        device_thread.join();
        host_modifier.join();

        // Verify: host buffer should contain EITHER the original data
        // OR the modified data (both are valid under CXL coherency).
        // What's NOT valid: corrupted data (torn reads) or zero.
        bool ok = true;
        for (size_t i = 0; i < CHAIN_DEPTH; ++i) {
            uint64_t buf_data = host_buf[i * 8];
            uint64_t buf_seq = host_buf[i * 8 + 1];

            uint64_t original_data = 0x1000 + i;
            uint64_t modified_data = 0xF000 + i + iter;

            // Data must be one of: original OR modified (not zero, not garbage)
            if (buf_data != original_data && buf_data != modified_data) {
                // For early nodes (before modification), must be original
                if (i < CHAIN_DEPTH / 2) {
                    if (buf_data != original_data) {
                        if (g_verbose)
                            printf("  FAIL iter=%d node=%zu: data=%lx "
                                   "(expected original=%lx)\n",
                                   iter, i, (unsigned long)buf_data,
                                   (unsigned long)original_data);
                        ok = false;
                        break;
                    }
                } else {
                    // For late nodes, either value is acceptable
                    // (depends on race between device read and host write)
                    // But ZERO or garbage is never acceptable
                    if (buf_data == 0) {
                        if (g_verbose)
                            printf("  FAIL iter=%d node=%zu: data=0x0 "
                                   "(lost data!)\n", iter, i);
                        ok = false;
                        stats.corruption++;
                        break;
                    }
                }
            }

            // Sequence number should always be original (host doesn't modify it)
            if (buf_seq != i) {
                if (g_verbose)
                    printf("  FAIL iter=%d node=%zu: seq=%lu expected=%zu\n",
                           iter, i, (unsigned long)buf_seq, i);
                ok = false;
                break;
            }
        }

        if (ok) stats.pass++;
        else stats.fail++;

        // Verify chain integrity
        for (size_t i = 0; i < CHAIN_DEPTH; ++i) {
            if (!verify_node(&chain[i])) {
                if (g_verbose)
                    printf("  WARN iter=%d: node %zu checksum mismatch "
                           "(expected if host modified)\n", iter, i);
            }
        }

        free(chain);
        free(const_cast<uint64_t*>(host_buf));
        free(comp);
    }
}

// =============================================================================
// TEST 5: Multi-Writer Cacheline Contention (DCOH Directory Hoisting)
//
// "cacheline directory may hoist the state" — Vickie
//
// Multiple threads write to different fields within the SAME cache line.
// Under CXL, this causes DCOH directory bouncing. The test verifies that
// all writes eventually become visible and no data is lost.
//
// This is a classic false-sharing test adapted for CXL DCOH.
// =============================================================================
void test_dcoh_directory_contention(TestStats& stats) {
    printf("\n=== TEST 5: DCOH Directory Contention (False Sharing) ===\n");
    printf("  Scenario: %d threads write to fields in same cache line\n",
           g_num_threads);
    printf("  Iterations: %d\n\n", g_iterations);

    // One cache line, each thread owns a uint16_t slot
    struct alignas(64) SharedLine {
        volatile uint16_t slots[32];  // 32 x 2 bytes = 64 bytes = 1 cache line
    };

    for (int iter = 0; iter < g_iterations; ++iter) {
        SharedLine* line = (SharedLine*)alloc_aligned(sizeof(SharedLine));
        if (!line) { stats.fail++; continue; }

        const uint16_t WRITES_PER_THREAD = 1000;
        std::atomic<int> ready{0};

        std::vector<std::thread> threads;
        for (int t = 0; t < g_num_threads; ++t) {
            threads.emplace_back([&, t]() {
                ready.fetch_add(1, std::memory_order_release);
                // Wait for all threads
                while (ready.load(std::memory_order_acquire) < g_num_threads) {
#ifdef __x86_64__
                    _mm_pause();
#endif
                }

                // Each thread increments its own slot (false sharing)
                for (uint16_t w = 0; w < WRITES_PER_THREAD; ++w) {
                    line->slots[t] = w + 1;
                    // Small delay to increase contention window
                    for (volatile int j = 0; j < 10; ++j) {}
                }
            });
        }

        for (auto& t : threads) t.join();

        // Verify: each slot should have its final value
        bool ok = true;
        for (int t = 0; t < g_num_threads; ++t) {
            if (line->slots[t] != WRITES_PER_THREAD) {
                if (g_verbose)
                    printf("  FAIL iter=%d slot=%d: val=%u expected=%u\n",
                           iter, t, line->slots[t], WRITES_PER_THREAD);
                ok = false;
                break;
            }
        }

        if (ok) stats.pass++;
        else { stats.fail++; stats.corruption++; }

        free(line);
    }
}

// =============================================================================
// TEST 6: Eviction + Re-Install Race (The "Address Pinning" Concern)
//
// Vickie: "if the place is not original one, we lose the data"
// Andi: "cxlmemuring doesn't change physical addresses"
//
// Test: device installs data at address A. Host evicts A. Device re-installs
// at the same address A. Is the re-installed data correct?
//
// Under CXL: physical address doesn't change. Eviction puts the line back
// to CXL memory in Modified/Clean state. Re-install just re-fetches.
// =============================================================================
void test_evict_reinstall_race(TestStats& stats) {
    printf("\n=== TEST 6: Evict + Re-Install Race (Address Pinning) ===\n");
    printf("  Scenario: install -> evict -> reinstall -> verify\n");
    printf("  Iterations: %d\n\n", g_iterations);

    const size_t NUM_LINES = 32;
    const size_t BUF_SIZE = NUM_LINES * CACHELINE_SIZE;

    for (int iter = 0; iter < g_iterations; ++iter) {
        volatile uint8_t* buf = (volatile uint8_t*)alloc_aligned(BUF_SIZE);
        if (!buf) { stats.fail++; continue; }

        // Phase 1: "Device" installs data
        for (size_t i = 0; i < BUF_SIZE; ++i) {
            buf[i] = (uint8_t)((i * 7 + iter) & 0xFF);
        }
        __atomic_thread_fence(__ATOMIC_RELEASE);

        // Phase 2: "Host" evicts all installed lines
        for (size_t cl = 0; cl < NUM_LINES; ++cl) {
            void* addr = const_cast<uint8_t*>(&buf[cl * CACHELINE_SIZE]);
#ifdef __x86_64__
            _mm_clflushopt(addr);
#endif
        }
#ifdef __x86_64__
        _mm_sfence();
#endif

        // Phase 3: "Device" re-installs with different data
        for (size_t i = 0; i < BUF_SIZE; ++i) {
            buf[i] = (uint8_t)((i * 13 + iter + 1) & 0xFF);
        }
        __atomic_thread_fence(__ATOMIC_RELEASE);

        // Phase 4: Host reads — should see the re-installed data
        // (If addresses are reused correctly, this works.
        //  If "we lose the VA to PA mapping", this breaks.)
        bool ok = true;
        for (size_t i = 0; i < BUF_SIZE; ++i) {
            uint8_t expected = (uint8_t)((i * 13 + iter + 1) & 0xFF);
            uint8_t actual = buf[i];
            if (actual != expected) {
                if (g_verbose)
                    printf("  FAIL iter=%d byte=%zu: got=0x%02x expected=0x%02x\n",
                           iter, i, actual, expected);
                ok = false;
                break;
            }
        }

        if (ok) stats.pass++;
        else { stats.fail++; stats.corruption++; }

        free(const_cast<uint8_t*>(buf));
    }
}

// =============================================================================
// TEST 7: PREFETCHT0 Install + Racing CLFLUSHOPT
//
// Stress test: one thread installs cachelines with PREFETCHT0, another
// thread races to flush them with CLFLUSHOPT. After both complete,
// verify data integrity.
//
// This models: device prefetch installing vs host eviction hint racing.
// =============================================================================
void test_prefetch_vs_clflush_race(TestStats& stats) {
    printf("\n=== TEST 7: PREFETCHT0 vs CLFLUSHOPT Race ===\n");
    printf("  Scenario: concurrent prefetch-install and cache-flush\n");
    printf("  Iterations: %d\n\n", g_iterations);

    const size_t NUM_LINES = 64;
    const size_t BUF_SIZE = NUM_LINES * CACHELINE_SIZE;

    for (int iter = 0; iter < g_iterations; ++iter) {
        volatile uint8_t* buf = (volatile uint8_t*)alloc_aligned(BUF_SIZE);
        if (!buf) { stats.fail++; continue; }

        // Write known data
        for (size_t i = 0; i < BUF_SIZE; ++i) {
            buf[i] = (uint8_t)((i + iter * 3) & 0xFF);
        }
        __atomic_thread_fence(__ATOMIC_RELEASE);

        // Flush all lines first
        for (size_t cl = 0; cl < NUM_LINES; ++cl) {
#ifdef __x86_64__
            _mm_clflushopt(const_cast<uint8_t*>(&buf[cl * CACHELINE_SIZE]));
#endif
        }
#ifdef __x86_64__
        _mm_sfence();
#endif

        std::atomic<bool> go{false};

        // Thread 1: prefetch-install all lines into L1
        std::thread prefetcher([&]() {
            while (!go.load(std::memory_order_acquire)) {
#ifdef __x86_64__
                _mm_pause();
#endif
            }
            for (size_t cl = 0; cl < NUM_LINES; ++cl) {
                __builtin_prefetch(const_cast<uint8_t*>(
                    &buf[cl * CACHELINE_SIZE]), 0, 3);  // PREFETCHT0
            }
        });

        // Thread 2: flush lines concurrently
        std::thread flusher([&]() {
            while (!go.load(std::memory_order_acquire)) {
#ifdef __x86_64__
                _mm_pause();
#endif
            }
            for (size_t cl = 0; cl < NUM_LINES; ++cl) {
#ifdef __x86_64__
                _mm_clflushopt(const_cast<uint8_t*>(
                    &buf[cl * CACHELINE_SIZE]));
#endif
            }
        });

        go.store(true, std::memory_order_release);

        prefetcher.join();
        flusher.join();

#ifdef __x86_64__
        _mm_mfence();
#endif

        // Verify: data should still be correct regardless of who won
        bool ok = true;
        for (size_t i = 0; i < BUF_SIZE; ++i) {
            uint8_t expected = (uint8_t)((i + iter * 3) & 0xFF);
            uint8_t actual = buf[i];
            if (actual != expected) {
                if (g_verbose)
                    printf("  FAIL iter=%d byte=%zu: got=0x%02x expected=0x%02x\n",
                           iter, i, actual, expected);
                ok = false;
                break;
            }
        }

        if (ok) stats.pass++;
        else { stats.fail++; stats.corruption++; }

        free(const_cast<uint8_t*>(buf));
    }
}

// =============================================================================
// Main
// =============================================================================
void print_stats(const char* name, const TestStats& s) {
    uint64_t total = s.pass.load() + s.fail.load();
    printf("  %s: %lu/%lu passed", name, (unsigned long)s.pass.load(),
           (unsigned long)total);
    if (s.fail.load() > 0) {
        printf(" *** %lu FAILURES ***", (unsigned long)s.fail.load());
    }
    if (s.corruption.load() > 0) {
        printf(" (%lu corruptions)", (unsigned long)s.corruption.load());
    }
    if (s.eviction_refetch.load() > 0) {
        printf(" (%lu eviction-refetches)", (unsigned long)s.eviction_refetch.load());
    }
    if (total > 0 && s.total_latency_ns.load() > 0) {
        printf(" [avg read: %lu ns]",
               (unsigned long)(s.total_latency_ns.load() / total));
    }
    printf("\n");
}

int main(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], "--iterations=", 13) == 0) {
            g_iterations = atoi(argv[i] + 13);
        } else if (strcmp(argv[i], "--pollute-llc") == 0) {
            g_pollute_llc = true;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            g_verbose = true;
        } else if (strncmp(argv[i], "--threads=", 10) == 0) {
            g_num_threads = atoi(argv[i] + 10);
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --iterations=N     Number of test iterations [1000]\n");
            printf("  --pollute-llc      Enable LLC pollution between install/read\n");
            printf("  --threads=N        Threads for contention tests [4]\n");
            printf("  --verbose          Print individual failures\n");
            return 0;
        }
    }

    printf("================================================================\n");
    printf("CXL Cache Coherency Stress Test\n");
    printf("================================================================\n");
    printf("Iterations: %d\n", g_iterations);
    printf("LLC pollution: %s\n", g_pollute_llc ? "ON" : "OFF");
    printf("Threads: %d\n", g_num_threads);
    printf("\nThese tests exercise DCOH writeback + host cache coherency races.\n");
    printf("Run many iterations (--iterations=10000) to surface rare conditions.\n");
    printf("Run with --pollute-llc to test eviction recovery path.\n");
    printf("Run with numactl -m 2 on CXL hardware for real CXL latency.\n");
    printf("================================================================\n");

    TestStats stats[7];

    test_dcoh_completion_race(stats[0]);
    test_cacheline_eviction_during_prefetch(stats[1]);
    test_phase_boundary_ordering(stats[2]);
    test_chain_chase_concurrent_modification(stats[3]);
    test_dcoh_directory_contention(stats[4]);
    test_evict_reinstall_race(stats[5]);
    test_prefetch_vs_clflush_race(stats[6]);

    printf("\n================================================================\n");
    printf("RESULTS\n");
    printf("================================================================\n");
    print_stats("T1 DCOH completion race     ", stats[0]);
    print_stats("T2 Eviction during prefetch ", stats[1]);
    print_stats("T3 Phase boundary ordering  ", stats[2]);
    print_stats("T4 Chain chase + host modify", stats[3]);
    print_stats("T5 DCOH directory contention", stats[4]);
    print_stats("T6 Evict + reinstall race   ", stats[5]);
    print_stats("T7 PREFETCHT0 vs CLFLUSHOPT ", stats[6]);

    // Overall
    uint64_t total_pass = 0, total_fail = 0;
    for (int i = 0; i < 7; ++i) {
        total_pass += stats[i].pass.load();
        total_fail += stats[i].fail.load();
    }

    printf("\n");
    if (total_fail == 0) {
        printf("ALL %lu TESTS PASSED\n", (unsigned long)(total_pass + total_fail));
        printf("(No coherency bugs detected in %d iterations)\n", g_iterations);
    } else {
        printf("*** %lu / %lu TESTS FAILED ***\n",
               (unsigned long)total_fail,
               (unsigned long)(total_pass + total_fail));
        printf("Coherency bugs detected! Run with --verbose for details.\n");
    }
    printf("================================================================\n");

    return total_fail > 0 ? 1 : 0;
}
