// RUN: cira-opt %s -cira-two-pass-timing-injection=profile=%p/mcf_profile.json -cira-profile-guided-prefetch | FileCheck %s
//
// MCF Pricing Kernel Two-Pass Execution Test
// Demonstrates compiler-based timing analysis and injection for MCF pricing hotspot
//
// Two-Pass Methodology:
// Pass 1 (Profiling): Run Vortex simulation, collect T_host and T_vortex
// Pass 2 (Injection): Inject usleep delays at sync points when T_vortex > T_host
//
// From the paper:
//   Gain = Σ(L_CXL - L_LLC) - (C_sync + C_vortex_busy)
//   Where: L_CXL=165ns, L_LLC=15ns, C_sync=50ns

module attributes {
  twopass.clock_freq_mhz = 200.0,
  twopass.cxl_latency_ns = 165,
  twopass.llc_latency_ns = 15,
  twopass.sync_overhead_ns = 50
} {

// MCF arc data structure layout:
//   struct arc_t {
//     int64_t cost;       // offset 0
//     int32_t ident;      // offset 8  (BASIC=0, AT_LOWER=1, AT_UPPER=2)
//     node_t* tail;       // offset 12
//     node_t* head;       // offset 20
//   }

// MCF basket structure:
//   struct basket_t {
//     arc_t* a;           // offset 0
//     int64_t cost;       // offset 8 (reduced cost)
//     uint64_t abs_cost;  // offset 16
//   }

// Test 1: MCF Pricing Kernel - the main hotspot
// CHECK-LABEL: func.func @mcf_pricing_kernel
// CHECK: twopass.region_id = 0
// CHECK: twopass.vortex_cycles
func.func @mcf_pricing_kernel(
    %arcs: memref<?x32xi8>,           // Array of arc_t structures
    %num_arcs: index,
    %basket: memref<?x24xi8>,         // Output basket array
    %basket_size: memref<1xi64>       // Atomic counter for candidates
) attributes {
  // Two-pass annotations (will be filled by profiling pass)
  twopass.region_id = 0 : i32,
  twopass.region_name = "mcf_pricing_kernel"
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index      // offset to ident
  %c12 = arith.constant 12 : index    // offset to tail
  %c20 = arith.constant 20 : index    // offset to head
  %zero_i64 = arith.constant 0 : i64
  %at_lower = arith.constant 1 : i32
  %at_upper = arith.constant 2 : i32

  // CHECK: cira.offload
  // CHECK-SAME: {twopass.injection_delay_ns = {{[0-9]+}}
  cira.offload {
    // Stream prefetch for arc data (strided access pattern)
    // CHECK: cira.prefetch_stream
    // CHECK-SAME: stride = 32
    %arc_stream = cira.prefetch_stream %arcs, stride=32 : memref<?x32xi8>

    scf.parallel (%i) = (%c0) to (%num_arcs) step (%c1) {
      // Load arc->cost (offset 0)
      %cost_ptr = memref.subview %arcs[%i, %c0][1, 8][1, 1] : memref<?x32xi8> to memref<8xi8>
      %cost = cira.load_async %cost_ptr : memref<8xi8> -> !cira.future<i64>

      // Load arc->ident (offset 8)
      %ident_ptr = memref.subview %arcs[%i, %c8][1, 4][1, 1] : memref<?x32xi8> to memref<4xi8>
      %ident_fut = cira.load_async %ident_ptr : memref<4xi8> -> !cira.future<i32>

      // Load arc->tail->potential (indirect: offset 12, then potential at offset 0)
      %tail_ptr = memref.subview %arcs[%i, %c12][1, 8][1, 1] : memref<?x32xi8> to memref<8xi8>
      %tail_fut = cira.load_async %tail_ptr : memref<8xi8> -> !cira.future<i64>

      // Load arc->head->potential (indirect: offset 20, then potential at offset 0)
      %head_ptr = memref.subview %arcs[%i, %c20][1, 8][1, 1] : memref<?x32xi8> to memref<8xi8>
      %head_fut = cira.load_async %head_ptr : memref<8xi8> -> !cira.future<i64>

      // Await all futures (computation overlaps with memory)
      %cost_val = cira.future_await %cost : !cira.future<i64> -> i64
      %ident_val = cira.future_await %ident_fut : !cira.future<i32> -> i32
      %tail_pot = cira.future_await %tail_fut : !cira.future<i64> -> i64
      %head_pot = cira.future_await %head_fut : !cira.future<i64> -> i64

      // Compute reduced cost: red_cost = cost - tail->potential + head->potential
      %diff1 = arith.subi %cost_val, %tail_pot : i64
      %red_cost = arith.addi %diff1, %head_pot : i64

      // Check dual infeasibility:
      // (red_cost < 0 && ident == AT_LOWER) || (red_cost > 0 && ident == AT_UPPER)
      %neg = arith.cmpi slt, %red_cost, %zero_i64 : i64
      %pos = arith.cmpi sgt, %red_cost, %zero_i64 : i64
      %is_lower = arith.cmpi eq, %ident_val, %at_lower : i32
      %is_upper = arith.cmpi eq, %ident_val, %at_upper : i32

      %cond1 = arith.andi %neg, %is_lower : i1
      %cond2 = arith.andi %pos, %is_upper : i1
      %is_candidate = arith.ori %cond1, %cond2 : i1

      // If candidate, add to basket (atomic increment)
      scf.if %is_candidate {
        // Atomically increment basket_size
        %old_size = memref.atomic_rmw addi %basket_size[%c0], %c1
            : (memref<1xi64>, index) -> i64
        // Store candidate info in basket[old_size]
        // (simplified - actual impl would store arc pointer and costs)
      }

      scf.reduce
    }

    cira.yield
  }

  // CHECK: twopass.sync_point
  // This is where timing injection happens in pass 2
  cira.barrier {twopass.sync_point = 0 : i32}

  return
}

// Test 2: MCF Price Out Implicit Arcs
// Another hot region that benefits from offloading
// CHECK-LABEL: func.func @mcf_price_out_impl
func.func @mcf_price_out_impl(
    %org_arcs: memref<?x32xi8>,
    %num_org_arcs: index,
    %impl_arcs: memref<?x32xi8>,
    %num_impl_arcs: memref<1xi64>
) attributes {
  twopass.region_id = 1 : i32,
  twopass.region_name = "mcf_price_out_impl"
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: cira.offload
  cira.offload {
    // Prefetch original arcs
    %org_stream = cira.prefetch_stream %org_arcs, stride=32 : memref<?x32xi8>

    scf.for %i = %c0 to %num_org_arcs step %c1 {
      // Process each original arc to generate implicit arcs
      // (simplified - actual impl is more complex)
    }
    cira.yield
  }

  cira.barrier {twopass.sync_point = 1 : i32}
  return
}

// Test 3: Full simplex iteration with multiple offload regions
// Demonstrates dominator tree integration for prefetch placement
// CHECK-LABEL: func.func @mcf_simplex_iteration
func.func @mcf_simplex_iteration(
    %network: memref<?xi8>,      // Network structure
    %arcs: memref<?x32xi8>,
    %num_arcs: index
) -> i1 attributes {
  twopass.function_profile = true
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %true = arith.constant true

  // Allocate basket for candidates
  %basket = memref.alloc(%num_arcs) : memref<?x24xi8>
  %basket_size = memref.alloc() : memref<1xi64>

  // Phase 1: Pricing (parallel, offloadable)
  // CHECK: twopass.optimal_prefetch_depth
  // CHECK: twopass.should_hoist_h2d = true
  call @mcf_pricing_kernel(%arcs, %num_arcs, %basket, %basket_size)
      : (memref<?x32xi8>, index, memref<?x24xi8>, memref<1xi64>) -> ()

  // Phase 2: Price implicit arcs (parallel, offloadable)
  %impl_arcs = memref.alloc(%num_arcs) : memref<?x32xi8>
  %num_impl = memref.alloc() : memref<1xi64>
  call @mcf_price_out_impl(%arcs, %num_arcs, %impl_arcs, %num_impl)
      : (memref<?x32xi8>, index, memref<?x32xi8>, memref<1xi64>) -> ()

  // Phase 3: Pivot selection (sequential, CPU-bound)
  // Not offloaded - tree traversal with dependencies

  // Phase 4: Flow update (mostly sequential)
  // Not offloaded

  memref.dealloc %basket : memref<?x24xi8>
  memref.dealloc %basket_size : memref<1xi64>
  memref.dealloc %impl_arcs : memref<?x32xi8>
  memref.dealloc %num_impl : memref<1xi64>

  return %true : i1
}

// Test 4: Cost model verification
// Demonstrates the gain calculation from the paper
// CHECK-LABEL: func.func @verify_cost_model
func.func @verify_cost_model(%chain_depth: index) -> i1 attributes {
  // From paper Section 4.4:
  // Gain = chain_depth × (L_CXL - L_LLC) - (C_sync + C_vortex_busy)
  //
  // With L_CXL=165ns, L_LLC=15ns, C_sync=50ns:
  // Gain = chain_depth × 150 - 50
  //
  // Profitable when chain_depth >= 1 (actually 4 minimum for overhead)
  twopass.cost_model_params = {
    l_cxl_ns = 165,
    l_llc_ns = 15,
    c_sync_ns = 50,
    min_chain_depth = 4
  }
} {
  %c4 = arith.constant 4 : index

  // CHECK: twopass.should_offload
  %should_offload = arith.cmpi sge, %chain_depth, %c4 : index

  return %should_offload : i1
}

}
