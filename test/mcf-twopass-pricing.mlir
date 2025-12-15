// RUN: cira %s --cira-twopass-timing="profile=%p/mcf_profile.json" --cira-profile-prefetch | FileCheck %s
//
// MCF Pricing Kernel Two-Pass Execution Test
// Demonstrates compiler-based timing analysis and injection for MCF pricing hotspot
//
// Two-Pass Methodology:
// Pass 1 (Profiling): Run Vortex simulation, collect T_host and T_vortex
// Pass 2 (Injection): Inject usleep delays at sync points when T_vortex > T_host
//
// From the paper:
//   Gain = sum(L_CXL - L_LLC) - (C_sync + C_vortex_busy)
//   Where: L_CXL=165ns, L_LLC=15ns, C_sync=50ns

module attributes {
  twopass.clock_freq_mhz = 200.0 : f64,
  twopass.cxl_latency_ns = 165 : i64,
  twopass.llc_latency_ns = 15 : i64,
  twopass.sync_overhead_ns = 50 : i64
} {

// Test 1: MCF Pricing Kernel - the main hotspot
// CHECK-LABEL: func.func @mcf_pricing_kernel
func.func @mcf_pricing_kernel(
    %arcs: memref<?x32xi8>,
    %num_arcs: index
) attributes {
  twopass.region_id = 0 : i32,
  twopass.region_name = "mcf_pricing_kernel"
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero_i64 = arith.constant 0 : i64

  // Offload region for Vortex prefetching
  // CHECK: cira.offload
  cira.offload attributes {region_name = "mcf_pricing_kernel"} {
    // Simplified computation loop
    scf.for %i = %c0 to %num_arcs step %c1 {
      // Compute reduced cost placeholder
      %cost_val = arith.constant 100 : i64
      %tail_pot = arith.constant 50 : i64
      %head_pot = arith.constant 30 : i64

      // red_cost = cost - tail->potential + head->potential
      %diff1 = arith.subi %cost_val, %tail_pot : i64
      %red_cost = arith.addi %diff1, %head_pot : i64

      // Check if candidate (simplified)
      %is_candidate = arith.cmpi slt, %red_cost, %zero_i64 : i64
    }
    cira.yield
  }

  // Sync point - timing injection happens here in pass 2
  cira.phase_boundary "sync_0"

  return
}

// Test 2: MCF Price Out Implicit Arcs
// CHECK-LABEL: func.func @mcf_price_out_impl
func.func @mcf_price_out_impl(
    %org_arcs: memref<?x32xi8>,
    %num_org_arcs: index
) attributes {
  twopass.region_id = 1 : i32,
  twopass.region_name = "mcf_price_out_impl"
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: cira.offload
  cira.offload attributes {region_name = "mcf_price_out_impl"} {
    scf.for %i = %c0 to %num_org_arcs step %c1 {
      // Process each original arc (simplified)
      %val = arith.constant 42 : i64
    }
    cira.yield
  }

  cira.phase_boundary "sync_1"
  return
}

// Test 3: Full simplex iteration
// CHECK-LABEL: func.func @mcf_simplex_iteration
func.func @mcf_simplex_iteration(
    %arcs: memref<?x32xi8>,
    %num_arcs: index
) -> i1 attributes {
  twopass.function_profile = true
} {
  %true = arith.constant true

  // Phase 1: Pricing (parallel, offloadable)
  func.call @mcf_pricing_kernel(%arcs, %num_arcs)
      : (memref<?x32xi8>, index) -> ()

  // Phase 2: Price implicit arcs (parallel, offloadable)
  func.call @mcf_price_out_impl(%arcs, %num_arcs)
      : (memref<?x32xi8>, index) -> ()

  return %true : i1
}

// Test 4: Cost model verification
// CHECK-LABEL: func.func @verify_cost_model
func.func @verify_cost_model(%chain_depth: index) -> i1 {
  // Gain = chain_depth * (L_CXL - L_LLC) - (C_sync + C_vortex_busy)
  // With L_CXL=165ns, L_LLC=15ns, C_sync=50ns:
  // Gain = chain_depth * 150 - 50
  // Profitable when chain_depth >= 4 (minimum for overhead)
  %c4 = arith.constant 4 : index

  %should_offload = arith.cmpi sge, %chain_depth, %c4 : index

  return %should_offload : i1
}

}
