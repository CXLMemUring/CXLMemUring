// Unit tests for the CIRA JIT cost-model (cira_jit.h).
//
// We exercise four canonical workload shapes and assert the JIT picks
// reasonable values for each knob. The tests do NOT depend on LLVM ORC —
// they validate the decision logic in isolation.

#include "cira_jit.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

// Scenario builder — only fields the cost-model reads need to be set.
cira_jit_workload_t make_workload(uint64_t t_host_ns,
                                  uint64_t t_vortex_ns,
                                  uint64_t compute_cyc,
                                  uint64_t stall_cyc,
                                  uint64_t n_elems,
                                  double host_per_elem_ns = 0.0) {
    cira_jit_workload_t w;
    std::memset(&w, 0, sizeof(w));
    w.host_independent_work_ns   = t_host_ns;
    w.vortex_total_time_ns       = t_vortex_ns;
    w.vortex_compute_cycles      = compute_cyc;
    w.vortex_memory_stall_cycles = stall_cyc;
    w.vortex_total_cycles        = compute_cyc + stall_cyc;
    w.num_elements               = n_elems;
    w.host_per_elem_ns           = host_per_elem_ns;
    return w;
}

void check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "[FAIL] %s\n", msg);
        std::exit(1);
    }
}

void dump(const char* label, const cira_jit_decision_t& d) {
    char buf[200];
    cira_jit_format(&d, buf, sizeof(buf));
    std::printf("  %-30s %s\n", label, buf);
}

} // namespace

// --- Scenario 1: tiny region ----------------------------------------------
// Whole region runs in <2 sync overheads. JIT must keep it on the host.
static void test_tiny_region() {
    auto w  = make_workload(/*t_host*/   200,   /*t_vortex*/ 200,
                            /*compute*/  10,    /*stall*/    10,
                            /*n_elems*/  16);
    auto hw = cira_jit_default_limits();
    cira_jit_decision_t d;
    cira_jit_decide(&w, &hw, &d);
    dump("tiny_region", d);
    check(!d.should_offload,            "tiny region must stay on host");
    check(d.host_device_split == 1.0f,  "tiny region split must be 1.0");
    check(d.reason_bits & CIRA_JIT_REASON_SMALL_REGION, "expect SMALL_REGION reason");
}

// --- Scenario 2: memory-bound, latency hidden ------------------------------
// Heavy device-side memory stalls but T_host >= T_vortex — the device
// finishes first, JIT should set a deep prefetch pipe + offload fully.
static void test_memory_bound_hidden() {
    auto w  = make_workload(/*t_host*/   100000,  /*t_vortex*/ 50000,
                            /*compute*/  2000,    /*stall*/    8000,
                            /*n_elems*/  10000);
    auto hw = cira_jit_default_limits();
    cira_jit_decision_t d;
    cira_jit_decide(&w, &hw, &d);
    dump("mem_bound_hidden", d);
    check(d.should_offload,                          "must offload");
    check(d.latency_hidden,                          "latency must be hidden");
    check(d.reason_bits & CIRA_JIT_REASON_LATENCY_HIDDEN, "expect HIDDEN reason");
    check(d.host_device_split == 0.0f,               "device-only when hidden");
    check(d.pipeline_distance >= 8,                  "stall ⇒ deep prefetch pipe");
    check(d.traversal_depth   >= 4,                  "high stall ⇒ deep traversal");
    check(d.batch_size >= hw.min_batch_size,         "batch must respect min");
    check(d.batch_size <= hw.max_batch_size,         "batch must respect max");
}

// --- Scenario 3: memory-bound, latency exposed -----------------------------
// Device is the long pole — JIT should rebalance some work back to host
// and the split should be > 0.
static void test_memory_bound_exposed() {
    auto w  = make_workload(/*t_host*/   30000,   /*t_vortex*/ 80000,
                            /*compute*/  1500,    /*stall*/    14000,
                            /*n_elems*/  10000,   /*host_per_elem*/ 3.0);
    auto hw = cira_jit_default_limits();
    cira_jit_decision_t d;
    cira_jit_decide(&w, &hw, &d);
    dump("mem_bound_exposed", d);
    check(d.should_offload,                              "still profitable to offload");
    check(!d.latency_hidden,                             "latency NOT hidden");
    check(d.host_device_split > 0.0f,                    "should rebalance to host");
    check(d.host_device_split < 1.0f,                    "but not entirely");
    check(d.reason_bits & CIRA_JIT_REASON_SPLIT_REBALANCE, "expect SPLIT_REBALANCE");
}

// --- Scenario 4: compute-bound (low stall ratio) ---------------------------
// Stall fraction <10% and device slower than host — JIT should bail.
static void test_compute_bound() {
    auto w  = make_workload(/*t_host*/   10000,  /*t_vortex*/ 50000,
                            /*compute*/  9500,   /*stall*/    500,
                            /*n_elems*/  1000);
    auto hw = cira_jit_default_limits();
    cira_jit_decision_t d;
    cira_jit_decide(&w, &hw, &d);
    dump("compute_bound", d);
    check(!d.should_offload,                              "compute-bound: stay on host");
    check(d.reason_bits & CIRA_JIT_REASON_COMPUTE_BOUND,  "expect COMPUTE_BOUND");
}

// --- Scenario 5: clamp behaviour ------------------------------------------
// Force values past the ceilings and verify the clamp fires.
static void test_clamp() {
    cira_jit_decision_t d;
    std::memset(&d, 0, sizeof(d));
    d.batch_size        = 99999;
    d.traversal_depth   = 99999;
    d.pipeline_distance = 99999;
    d.host_device_split = 5.0f;
    auto hw = cira_jit_default_limits();
    cira_jit_clamp_to_hw(&d, &hw);
    dump("clamp", d);
    check(d.batch_size        == hw.max_batch_size,        "batch clamped");
    check(d.traversal_depth   == hw.max_traversal_depth,   "depth clamped");
    check(d.pipeline_distance == hw.max_pipeline_distance, "pipe clamped");
    check(d.host_device_split == 1.0f,                     "split clamped");
    check(d.reason_bits & CIRA_JIT_REASON_HW_CLAMPED,      "expect HW_CLAMPED");
}

int main() {
    std::puts("=== cira_jit cost-model tests ===");
    test_tiny_region();
    test_memory_bound_hidden();
    test_memory_bound_exposed();
    test_compute_bound();
    test_clamp();
    std::puts("All cira_jit tests passed.");
    return 0;
}
