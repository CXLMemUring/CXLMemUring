// RUN: cira %s --convert-cira-to-llvm-x86 | FileCheck %s --check-prefix=X86
// RUN: cira %s --vortex-kernel-gen 2>&1 | FileCheck %s --check-prefix=KERNELGEN
//
// End-to-end test: CPU-to-GPU offloading with cacheline installation
//
// Exercises the full CIRA pipeline from the paper:
//   1. Host allocates LLC tile + future (cira.alloc_cxl, cira.future_create)
//   2. Device-side offload: install_cacheline + prefetch_chain
//   3. Host awaits via MONITOR/MWAIT (cira.future_await)
//   4. Host consumes from LLC, releases buffer, phase boundary
//
// Maps to paper Section 3.3, Listings 1-2

module attributes {
  twopass.clock_freq_mhz = 200.0 : f64,
  twopass.cxl_latency_ns = 165 : i64,
  twopass.llc_latency_ns = 15 : i64,
  twopass.sync_overhead_ns = 50 : i64
} {

// ============================================================================
// Test 1: Pointer-chasing prefetch with cacheline install (Paper Listing 1)
// ============================================================================

// X86-LABEL: func.func @linked_list_traverse
// X86: llvm.call @cira_future_alloc
// X86: llvm.call @cira_install_cacheline_x86
// X86: llvm.call @cira_future_await
// X86: llvm.call @cira_phase_barrier
func.func @linked_list_traverse(
    %start_node: !cira.handle<i64>,
    %num_nodes: index
) attributes {
  twopass.region_id = 0 : i32,
  twopass.region_name = "linked_list_traverse"
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index

  // Host: allocate LLC tile buffer
  %buf = cira.alloc_cxl %c16 : !cira.handle<i64>

  // Host: create future (64-byte aligned CompletionData for DCOH)
  %f = cira.future_create : !cira.future<i64>

  // Device-side offload: chase chain, install cachelines, prefetch
  // KERNELGEN: Generating prefetch chain kernel
  cira.offload attributes {
    cira.access_pattern = "pointer_chase",
    cira.chain_depth_estimate = 16 : i64,
    region_name = "linked_list_prefetch"
  } {
    // Device: take ownership of node cacheline via CXL.cache
    cira.install_cacheline %start_node, %c16 : !cira.handle<i64>, index

    // Device: create stream and chase 16 nodes ahead
    %stream = cira.stream_create_indirect %start_node, offset=8
        : !cira.handle<i64> -> !cira.stream<i64>
    cira.prefetch_chain %stream, depth=16 : !cira.stream<i64>

    cira.yield
  }

  // Host: computation loop - overlaps with Vortex prefetching
  scf.for %i = %c0 to %num_nodes step %c1 {
    // Host: mwait on completion cacheline until device signals done
    %val = cira.future_await %f : !cira.future<i64> -> i64

    // Consume from LLC tile (data already installed by device DCOH)
    %loaded = cira.load_async %buf : !cira.handle<i64> -> !cira.future<i64>
  }

  // Drain all outstanding offload tasks
  cira.barrier

  // Release LLC tile buffer
  cira.release %buf : !cira.handle<i64>

  // Phase boundary before next computation phase
  cira.phase_boundary "traverse_done"

  return
}

// ============================================================================
// Test 2: Hash join probe with pipelined chain walk (Paper Listing 2)
// ============================================================================

// X86-LABEL: func.func @hash_join_probe
// X86: llvm.call @cira_future_alloc
// X86: llvm.call @cira_install_cacheline_x86
// X86: llvm.call @cira_future_await
// X86: llvm.call @cira_evict_hint_x86
func.func @hash_join_probe(
    %buckets: !cira.handle<i64>,
    %num_probes: index
) attributes {
  twopass.region_id = 1 : i32,
  twopass.region_name = "hash_join_probe"
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index

  // Allocate LLC tile for hash table nodes
  %ht_buf = cira.alloc_cxl %c8 : !cira.handle<i64>

  // Create future for chain node delivery
  %chain_f = cira.future_create : !cira.future<i64>

  scf.for %i = %c0 to %num_probes step %c1 {
    // Device: walk bucket chain, install each node
    cira.offload attributes {
      cira.access_pattern = "hash_chain_walk",
      region_name = "hash_join_chain_walk"
    } {
      // Install bucket head cacheline
      cira.install_cacheline %buckets, %c8 : !cira.handle<i64>, index

      // Chase bucket->next chain
      cira.prefetch_indirect %buckets, offset=16, depth=8
          : !cira.handle<i64>

      cira.yield
    }

    // Host: wait for chain nodes in LLC
    %node_data = cira.future_await %chain_f : !cira.future<i64> -> i64
  }

  cira.barrier

  // Evict processed hash table data to prevent LLC pollution
  cira.evict_hint %buckets, %c8 : !cira.handle<i64>

  // Release buffer
  cira.release %ht_buf : !cira.handle<i64>

  cira.phase_boundary "probe_batch_done"

  return
}

// ============================================================================
// Test 3: Speculative prefetch with eviction hint
// ============================================================================

// X86-LABEL: func.func @speculative_prefetch
// X86: llvm.call @cira_evict_hint_x86
func.func @speculative_prefetch(
    %data: !cira.handle<f32>,
    %size: index
) attributes {
  twopass.region_id = 2 : i32,
  twopass.region_name = "speculative_prefetch"
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index

  // Stream prefetch for sequential data
  cira.prefetch_stream %data, %size, stride=%c64 : !cira.handle<f32>, index

  scf.for %i = %c0 to %size step %c1 {
    // Speculative prefetch: device executes if idle
    %result = cira.speculate confidence = 0.8 : !cira.future<f32> {
      %fut = cira.load_async %data : !cira.handle<f32> -> !cira.future<f32>
      cira.yield %fut : !cira.future<f32>
    }

    // After consuming, hint eviction for LLC pollution control
    cira.evict_hint %data : !cira.handle<f32>
  }

  cira.phase_boundary "speculative_done"
  return
}

// ============================================================================
// Test 4: Bulk cacheline install for CSR neighbor arrays
// ============================================================================

// X86-LABEL: func.func @bulk_cacheline_install
// X86: llvm.call @cira_install_cacheline_x86
// X86: llvm.call @cira_evict_hint_x86
func.func @bulk_cacheline_install(
    %csr_edges: !cira.handle<i32>,
    %num_neighbors: index
) attributes {
  twopass.region_id = 3 : i32,
  twopass.region_name = "bulk_cacheline_install"
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // Install into LLC level 3 (just LLC, not L1/L2)
  cira.install_cacheline %csr_edges, %num_neighbors level = 3
      : !cira.handle<i32>, index

  // Process - all hits in LLC now
  scf.for %j = %c0 to %num_neighbors step %c1 {
    %edge = cira.load_async %csr_edges[%j]
        : !cira.handle<i32> -> !cira.future<i32>
    %neighbor = cira.future_await %edge : !cira.future<i32> -> i32
  }

  // Evict processed data to free LLC space for other tenants
  cira.evict_hint %csr_edges, %num_neighbors : !cira.handle<i32>

  cira.phase_boundary "neighbors_processed"
  return
}

// ============================================================================
// Test 5: Multi-phase pipeline (MCF pricing + price-out)
// ============================================================================

// X86-LABEL: func.func @multi_phase_pipeline
func.func @multi_phase_pipeline(
    %arcs: !cira.handle<i64>,
    %num_arcs: index
) attributes {
  twopass.function_profile = true
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index

  // ---- Phase 1: Pricing kernel (offload prefetch) ----
  %pricing_buf = cira.alloc_cxl %c16 : !cira.handle<i64>
  %pricing_f = cira.future_create : !cira.future<i64>

  // KERNELGEN: Generating prefetch chain kernel
  cira.offload attributes {
    cira.access_pattern = "arc_chain_walk",
    region_name = "pricing_prefetch"
  } {
    %arc_stream = cira.stream_create_indirect %arcs, offset=24
        : !cira.handle<i64> -> !cira.stream<i64>
    cira.prefetch_chain %arc_stream, depth=16 : !cira.stream<i64>
    cira.install_cacheline %arcs, %c16 : !cira.handle<i64>, index
    cira.yield
  }

  // Host: process pricing
  scf.for %i = %c0 to %num_arcs step %c1 {
    %val = cira.future_await %pricing_f : !cira.future<i64> -> i64
  }

  // Release pricing buffer
  cira.release %pricing_buf : !cira.handle<i64>

  // Phase boundary: pricing -> price-out
  cira.phase_boundary "pricing_done"

  // ---- Phase 2: Price-out kernel ----
  %priceout_buf = cira.alloc_cxl %c16 : !cira.handle<i64>
  %priceout_f = cira.future_create : !cira.future<i64>

  cira.offload attributes {
    cira.access_pattern = "implicit_arc_eval",
    region_name = "priceout_prefetch"
  } {
    cira.install_cacheline %arcs, %c16 : !cira.handle<i64>, index
    cira.prefetch_stream %arcs, %c16 : !cira.handle<i64>, index
    cira.yield
  }

  scf.for %j = %c0 to %num_arcs step %c1 {
    %val2 = cira.future_await %priceout_f : !cira.future<i64> -> i64
  }

  // Release price-out buffer
  cira.release %priceout_buf : !cira.handle<i64>

  cira.phase_boundary "priceout_done"

  return
}

}
