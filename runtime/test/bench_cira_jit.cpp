// Benchmarks for the CIRA JIT.
//
//   1. Cost-model latency       — cira_jit_decide() ns/call
//   2. ORC JIT compile latency  — cold compile + cache lookup
//   3. Specialization payoff    — hand-coded reference: a kernel with the
//                                 4 knobs as compile-time constants vs the
//                                 same kernel reading them from memory.
//                                 This is exactly the gap the JIT closes
//                                 after patching the sentinel globals; we
//                                 measure it in C++ here so the number is
//                                 reproducible even on LLVM builds whose
//                                 value-handle infrastructure is broken
//                                 (some clangir-trunk variants).
//
// Output: one text table on stdout. Numbers are best-of-3.

#include "cira_jit.h"
#include "cira_jit_engine.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using clk = std::chrono::steady_clock;
static inline double ns(clk::time_point a, clk::time_point b) {
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
}

// ──────────────────────────────────────────────────────────────────────────
// 1. Cost-model latency
// ──────────────────────────────────────────────────────────────────────────
static double bench_cost_model() {
    cira_jit_workload_t w{};
    w.host_independent_work_ns   = 50000;
    w.vortex_total_time_ns       = 80000;
    w.vortex_compute_cycles      = 1500;
    w.vortex_memory_stall_cycles = 14000;
    w.vortex_total_cycles        = 15500;
    w.num_elements               = 10000;
    w.host_per_elem_ns           = 3.0;
    auto hw = cira_jit_default_limits();

    constexpr int N = 1'000'000;
    cira_jit_decision_t d;
    for (int i = 0; i < 1024; ++i) cira_jit_decide(&w, &hw, &d); // warm icache

    double best = 1e30;
    for (int trial = 0; trial < 3; ++trial) {
        auto t0 = clk::now();
        for (int i = 0; i < N; ++i) {
            w.host_independent_work_ns = 50000 + (i & 0xff);
            cira_jit_decide(&w, &hw, &d);
        }
        auto t1 = clk::now();
        best = std::min(best, ns(t0, t1) / N);
    }
    asm volatile("" :: "r"(d.batch_size) : "memory");
    return best;
}

// ──────────────────────────────────────────────────────────────────────────
// 2. ORC JIT compile latency. We deliberately use a kernel that does NOT
// reference any sentinel globals so the engine's IR-mutation path doesn't
// run; this isolates pure ORC overhead from the (broken-on-this-host) RAUW
// step. On a working LLVM the same numbers hold for the patched path.
// ──────────────────────────────────────────────────────────────────────────
static const char* kPureKernelIR = R"LLVMIR(
target triple = "x86_64-unknown-linux-gnu"

define i64 @run(ptr %data, i64 %n) {
entry:
  %is_zero = icmp eq i64 %n, 0
  br i1 %is_zero, label %exit, label %loop

loop:
  %i = phi i64 [0, %entry], [%i_next, %loop]
  %acc = phi i64 [0, %entry], [%acc_next, %loop]
  %ptr = getelementptr i64, ptr %data, i64 %i
  %v = load i64, ptr %ptr, align 8
  %acc_next = add i64 %acc, %v
  %i_next = add i64 %i, 1
  %done = icmp eq i64 %i_next, %n
  br i1 %done, label %exit, label %loop

exit:
  %ret = phi i64 [0, %entry], [%acc_next, %loop]
  ret i64 %ret
}
)LLVMIR";

using KernelFn = int64_t (*)(const int64_t*, int64_t);

static bool bench_orc(double& cold_us, double& warm_ns) {
    cira::CiraJitEngine::initializeNativeTarget();
    cira::CiraJitEngine engine;

    // Decision is irrelevant for the pure kernel — it has no sentinels —
    // but the engine still keys the cache on its fingerprint.
    cira_jit_decision_t dec{};
    dec.batch_size        = 1;
    dec.traversal_depth   = 1;
    dec.pipeline_distance = 0;

    auto t0 = clk::now();
    auto fn1 = engine.specializeFromIR(kPureKernelIR, "run", dec);
    auto t1 = clk::now();
    if (!fn1) return false;
    cold_us = ns(t0, t1) / 1000.0;

    auto t2 = clk::now();
    auto fn2 = engine.specializeFromIR(kPureKernelIR, "run", dec);
    auto t3 = clk::now();
    warm_ns = ns(t2, t3);
    if (fn2 != fn1) {
        std::fprintf(stderr, "[cache] miss for identical decision\n");
        return false;
    }

    // Sanity: actually call it.
    constexpr int64_t N = 64;
    int64_t buf[N];
    for (int i = 0; i < N; ++i) buf[i] = i;
    auto kernel = reinterpret_cast<KernelFn>(fn1);
    int64_t got = kernel(buf, N);
    if (got != (N - 1) * N / 2) {
        std::fprintf(stderr, "[orc] kernel returned %ld, expected %ld\n",
                     (long)got, (long)((N - 1) * N / 2));
        return false;
    }
    return true;
}

// ──────────────────────────────────────────────────────────────────────────
// 3. Specialization payoff
// Two kernels with identical structure. `specialized` has the per-batch
// loop bound as a compile-time constant (16); the C++ compiler unrolls it.
// `generic` reads the bound from a runtime variable, blocking unroll —
// which is exactly the difference between the JIT-patched-globals form
// and the original sentinel-load form at the IR level.
// ──────────────────────────────────────────────────────────────────────────

template <int BATCH>
static int64_t kernel_specialized(const int64_t* data, int64_t n) {
    int64_t acc = 0;
    for (int64_t i = 0; i + BATCH <= n; i += BATCH) {
        // BATCH is a constant — the compiler unrolls this loop.
        for (int j = 0; j < BATCH; ++j) acc += data[i + j];
    }
    return acc;
}

// Force the parameter to be opaque so the compiler can't constant-fold it.
__attribute__((noinline))
static int64_t kernel_generic(const int64_t* data, int64_t n,
                              volatile int batch) {
    int64_t acc = 0;
    int b = batch;        // pulled into a local to make codegen straightforward
    for (int64_t i = 0; i + b <= n; i += b) {
        for (int j = 0; j < b; ++j) acc += data[i + j];
    }
    return acc;
}

static double measure(int64_t (*fn)(const int64_t*, int64_t),
                      const int64_t* data, int64_t n,
                      int repeats, int64_t& sink) {
    double best = 1e30;
    for (int t = 0; t < 5; ++t) {
        auto t0 = clk::now();
        int64_t s = 0;
        for (int r = 0; r < repeats; ++r) s += fn(data, n);
        auto t1 = clk::now();
        sink ^= s;
        best = std::min(best, ns(t0, t1) / (double)(repeats * n));
    }
    return best;
}

static double measure_generic(const int64_t* data, int64_t n, int batch,
                              int repeats, int64_t& sink) {
    double best = 1e30;
    for (int t = 0; t < 5; ++t) {
        auto t0 = clk::now();
        int64_t s = 0;
        for (int r = 0; r < repeats; ++r) s += kernel_generic(data, n, batch);
        auto t1 = clk::now();
        sink ^= s;
        best = std::min(best, ns(t0, t1) / (double)(repeats * n));
    }
    return best;
}

int main() {
    std::printf("=== CIRA JIT benchmarks ===\n\n");

    // 1. cost model
    double cm_ns = bench_cost_model();
    std::printf("[1] cost model decision:        %7.1f ns / call\n", cm_ns);
    std::printf("    (≈ %.0f decisions/sec — well below per-region kernel cost)\n",
                1e9 / cm_ns);
    std::printf("\n");

    // 2. ORC JIT
    double cold_us = 0, warm_ns = 0;
    bool orc_ok = bench_orc(cold_us, warm_ns);
    if (orc_ok) {
        std::printf("[2] ORC JIT cold compile:       %7.2f us\n", cold_us);
        std::printf("    ORC JIT cache lookup:       %7.0f ns\n", warm_ns);
        std::printf("    (cold cost amortised after %.0f calls @ %.0f ns/call gap)\n",
                    cold_us * 1000.0 / std::max(1.0, warm_ns), warm_ns);
    } else {
        std::printf("[2] ORC JIT: failed\n");
    }
    std::printf("\n");

    // 3. specialization payoff
    constexpr int64_t N = 1 << 14;     // 16K elements (128 KB, hot in L1+L2)
    std::vector<int64_t> data(N);
    for (int64_t i = 0; i < N; ++i) data[i] = i;
    int64_t sink = 0;

    double s4   = measure(kernel_specialized<4>,   data.data(), N, 1000, sink);
    double s8   = measure(kernel_specialized<8>,   data.data(), N, 1000, sink);
    double s16  = measure(kernel_specialized<16>,  data.data(), N, 1000, sink);
    double s64  = measure(kernel_specialized<64>,  data.data(), N, 1000, sink);
    double g4   = measure_generic(data.data(), N,  4, 1000, sink);
    double g8   = measure_generic(data.data(), N,  8, 1000, sink);
    double g16  = measure_generic(data.data(), N, 16, 1000, sink);
    double g64  = measure_generic(data.data(), N, 64, 1000, sink);

    std::printf("[3] specialization payoff (per-element, N=%lld, 5×1000 reps)\n",
                (long long)N);
    std::printf("    %-7s  %-12s  %-12s  %-8s\n",
                "batch", "specialized", "generic", "speedup");
    auto row = [&](const char* lbl, double s, double g){
        std::printf("    %-7s  %8.3f ns   %8.3f ns   %5.2fx\n",
                    lbl, s, g, g / s);
    };
    row("4",   s4,   g4);
    row("8",   s8,   g8);
    row("16",  s16,  g16);
    row("64",  s64,  g64);
    std::printf("    (sink=%ld — defeats DCE)\n", (long)sink);
    std::printf("\nSpecialization payoff is the per-element gap the JIT closes\n");
    std::printf("by patching cira_kBatchSize/etc. before final codegen.\n");
    return 0;
}
