// RUN: cira %s --allow-unregistered-dialect \
// RUN:   --cira-twopass-timing="profile=%p/misc_profile.json" | FileCheck %s
//
// Spatter gather/scatter, UME mesh gradient, Partitioned Hash Join
// CIRA dialect lowering from real benchmark source code
//
// Spatter (CudaBackend.cu):  sparse[pattern[j] + delta*i]
// UME (gradient.cc):         point_grad[c_to_p[c]] += csurf[c] * zone_field[c_to_z[c]]
// PHJoin (phjoin.c/jobs.c):  partitioned[psum[hash(tuple)]] = tuple  →  search(table, key)

module attributes {
  twopass.clock_freq_mhz = 200.0 : f64,
  twopass.cxl_latency_ns = 165 : i64,
  twopass.llc_latency_ns = 15 : i64,
  twopass.sync_overhead_ns = 50 : i64
} {

// ════════════════════════════════════════════════════════════════
// SPATTER — Gather: dense[i] = sparse[pattern[i] + delta*count]
// ════════════════════════════════════════════════════════════════
// From CudaBackend.cu:8-24
// Data types: double (8 bytes), pattern: size_t (8 bytes)
// Indirection: pattern[j] used to index into sparse array
//
// CHECK-LABEL: func.func @spatter_gather
func.func @spatter_gather(
    %pattern: memref<?xi64>,            // index pattern array
    %sparse: memref<?xf64>,             // source data (CXL-resident, irregular access)
    %dense: memref<?xf64>,              // destination (sequential write)
    %pattern_length: index,
    %delta: i64,                        // stride between iterations
    %count: index                       // number of gather repetitions
) attributes {
  twopass.region_id = 0 : i32,
  twopass.region_name = "spatter_gather"
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: cira.offload
  cira.offload attributes {
    region_name = "spatter_gather",
    cira.access_pattern = "indexed_gather",
    cira.element_width = 8
  } {
    scf.for %i = %c0 to %count step %c1 {
      %i_i64 = arith.index_cast %i : index to i64
      %offset = arith.muli %delta, %i_i64 : i64

      scf.for %j = %c0 to %pattern_length step %c1 {
        // Load index from pattern: pattern[j]
        %pat_idx = memref.load %pattern[%j] : memref<?xi64>
        // Compute final address: pattern[j] + delta * i
        %addr = arith.addi %pat_idx, %offset : i64
        %addr_idx = arith.index_cast %addr : i64 to index

        // Indirect gather: sparse[pattern[j] + delta*i]
        %val = memref.load %sparse[%addr_idx] : memref<?xf64>

        // Sequential write to dense
        %dense_idx = arith.addi %j, %c0 : index  // simplified
        memref.store %val, %dense[%dense_idx] : memref<?xf64>
      }
    }
    cira.yield
  }

  cira.phase_boundary "gather_done"
  return
}

// ════════════════════════════════════════════════════════════════
// SPATTER — Scatter: sparse[pattern[i] + delta*count] = dense[i]
// ════════════════════════════════════════════════════════════════
// From CudaBackend.cu:26-36
//
// CHECK-LABEL: func.func @spatter_scatter
func.func @spatter_scatter(
    %pattern: memref<?xi64>,
    %sparse: memref<?xf64>,             // destination (CXL-resident, irregular write)
    %dense: memref<?xf64>,              // source (sequential read)
    %pattern_length: index,
    %delta: i64,
    %count: index
) attributes {
  twopass.region_id = 1 : i32,
  twopass.region_name = "spatter_scatter"
} {
  %c0_s = arith.constant 0 : index
  %c1_s = arith.constant 1 : index

  // CHECK: cira.offload
  cira.offload attributes {
    region_name = "spatter_scatter",
    cira.access_pattern = "indexed_scatter",
    cira.element_width = 8
  } {
    scf.for %i = %c0_s to %count step %c1_s {
      %i_i64 = arith.index_cast %i : index to i64
      %offset = arith.muli %delta, %i_i64 : i64

      scf.for %j = %c0_s to %pattern_length step %c1_s {
        // Read source value sequentially
        %val = memref.load %dense[%j] : memref<?xf64>

        // Compute scatter target: pattern[j] + delta*i
        %pat_idx = memref.load %pattern[%j] : memref<?xi64>
        %addr = arith.addi %pat_idx, %offset : i64
        %addr_idx = arith.index_cast %addr : i64 to index

        // Indirect scatter write
        memref.store %val, %sparse[%addr_idx] : memref<?xf64>
      }
    }
    cira.yield
  }

  cira.phase_boundary "scatter_done"
  return
}

// ════════════════════════════════════════════════════════════════
// SPATTER — Multi-Gather: dense[i] = sparse[pattern[pattern_gather[i]]]
// ════════════════════════════════════════════════════════════════
// From CudaBackend.cu:85-103
// Two-level indirection: pattern_gather[j] → pattern[...] → sparse[...]
//
// CHECK-LABEL: func.func @spatter_multi_gather
func.func @spatter_multi_gather(
    %pattern: memref<?xi64>,            // first-level index array
    %pattern_gather: memref<?xi64>,     // second-level index array
    %sparse: memref<?xf64>,             // source (CXL-resident)
    %dense: memref<?xf64>,              // destination
    %pattern_length: index,
    %delta: i64,
    %count: index
) attributes {
  twopass.region_id = 2 : i32,
  twopass.region_name = "spatter_multi_gather"
} {
  %c0_mg = arith.constant 0 : index
  %c1_mg = arith.constant 1 : index

  // CHECK: cira.offload
  cira.offload attributes {
    region_name = "spatter_multi_gather",
    cira.access_pattern = "nested_indexed_gather",
    cira.chain_depth_estimate = 2
  } {
    scf.for %i = %c0_mg to %count step %c1_mg {
      %i_i64 = arith.index_cast %i : index to i64
      %offset = arith.muli %delta, %i_i64 : i64

      scf.for %j = %c0_mg to %pattern_length step %c1_mg {
        // Level 1: pattern_gather[j] → index into pattern
        %pg_idx = memref.load %pattern_gather[%j] : memref<?xi64>
        %pg_idx_idx = arith.index_cast %pg_idx : i64 to index

        // Level 2: pattern[pattern_gather[j]] → index into sparse
        %pat_idx = memref.load %pattern[%pg_idx_idx] : memref<?xi64>
        %addr = arith.addi %pat_idx, %offset : i64
        %addr_idx = arith.index_cast %addr : i64 to index

        // Double-indirect gather
        %val = memref.load %sparse[%addr_idx] : memref<?xf64>
        memref.store %val, %dense[%j] : memref<?xf64>
      }
    }
    cira.yield
  }

  cira.phase_boundary "multi_gather_done"
  return
}

// ════════════════════════════════════════════════════════════════
// UME — Zone-to-Point Gradient (gradient.cc:40-47)
// ════════════════════════════════════════════════════════════════
// point_gradient[c_to_p[c]] += csurf[c] * zone_field[c_to_z[c]]
// point_volume[c_to_p[c]]  += corner_volume[c]
//
// Double indirection via connectivity maps:
//   c_to_z_map[c] → zone index z
//   c_to_p_map[c] → point index p
//
// CHECK-LABEL: func.func @ume_gradzatp
func.func @ume_gradzatp(
    %corner_type: memref<?xi32>,        // corner type mask (>=1 = interior)
    %c_to_z_map: memref<?xi32>,         // corner→zone connectivity
    %c_to_p_map: memref<?xi32>,         // corner→point connectivity
    %corner_volume: memref<?xf64>,      // per-corner volume
    %csurf_x: memref<?xf64>,           // corner surface area (x component)
    %csurf_y: memref<?xf64>,           // corner surface area (y component)
    %csurf_z: memref<?xf64>,           // corner surface area (z component)
    %zone_field: memref<?xf64>,         // zone scalar field (CXL-resident)
    %point_grad_x: memref<?xf64>,      // output gradient (x)
    %point_grad_y: memref<?xf64>,      // output gradient (y)
    %point_grad_z: memref<?xf64>,      // output gradient (z)
    %point_volume: memref<?xf64>,      // output per-point volume
    %num_corners: index
) attributes {
  twopass.region_id = 3 : i32,
  twopass.region_name = "ume_gradzatp"
} {
  %c0_ume = arith.constant 0 : index
  %c1_ume = arith.constant 1 : index
  %one_i32 = arith.constant 1 : i32

  // CHECK: cira.offload
  cira.offload attributes {
    region_name = "ume_gradzatp",
    cira.access_pattern = "connectivity_gather_scatter",
    cira.chain_depth_estimate = 2
  } {
    scf.for %c = %c0_ume to %num_corners step %c1_ume {
      // Skip non-interior corners
      %ctype = memref.load %corner_type[%c] : memref<?xi32>
      %is_interior = arith.cmpi sge, %ctype, %one_i32 : i32
      scf.if %is_interior {
        // Indirect lookups via connectivity maps
        %z_i32 = memref.load %c_to_z_map[%c] : memref<?xi32>
        %z = arith.index_cast %z_i32 : i32 to index
        %p_i32 = memref.load %c_to_p_map[%c] : memref<?xi32>
        %p = arith.index_cast %p_i32 : i32 to index

        // Load corner surface area vector (csurf[c])
        %cs_x = memref.load %csurf_x[%c] : memref<?xf64>
        %cs_y = memref.load %csurf_y[%c] : memref<?xf64>
        %cs_z = memref.load %csurf_z[%c] : memref<?xf64>

        // Load zone field value: zone_field[c_to_z[c]]
        %zf = memref.load %zone_field[%z] : memref<?xf64>

        // Accumulate: point_grad[c_to_p[c]] += csurf[c] * zone_field[z]
        %prod_x = arith.mulf %cs_x, %zf : f64
        %prod_y = arith.mulf %cs_y, %zf : f64
        %prod_z = arith.mulf %cs_z, %zf : f64

        %old_gx = memref.load %point_grad_x[%p] : memref<?xf64>
        %old_gy = memref.load %point_grad_y[%p] : memref<?xf64>
        %old_gz = memref.load %point_grad_z[%p] : memref<?xf64>
        memref.store %prod_x, %point_grad_x[%p] : memref<?xf64>
        memref.store %prod_y, %point_grad_y[%p] : memref<?xf64>
        memref.store %prod_z, %point_grad_z[%p] : memref<?xf64>

        // point_volume[c_to_p[c]] += corner_volume[c]
        %cv = memref.load %corner_volume[%c] : memref<?xf64>
        %old_pv = memref.load %point_volume[%p] : memref<?xf64>
        %new_pv = arith.addf %old_pv, %cv : f64
        memref.store %new_pv, %point_volume[%p] : memref<?xf64>
      }
    }
    cira.yield
  }

  cira.phase_boundary "ume_gradzatp_done"
  return
}

// ════════════════════════════════════════════════════════════════
// Hash Join — Radix Partition Phase (phjoin.c:98-101)
// ════════════════════════════════════════════════════════════════
// partitioned[start + psum[hash(tuple.payload)]] = tuple
//
// CHECK-LABEL: func.func @hashjoin_radix_partition
func.func @hashjoin_radix_partition(
    %tuples_key: memref<?xi32>,         // input tuple keys
    %tuples_payload: memref<?xi32>,     // input tuple payloads
    %partitioned_key: memref<?xi32>,    // output partitioned keys
    %partitioned_pay: memref<?xi32>,    // output partitioned payloads
    %psum: memref<?xi32>,               // prefix sum array (write target)
    %num_tuples: index,
    %hash_mask: i32,
    %start_offset: i32
) attributes {
  twopass.region_id = 4 : i32,
  twopass.region_name = "hashjoin_radix_partition"
} {
  %c0_hj = arith.constant 0 : index
  %c1_hj = arith.constant 1 : index
  %one_i32 = arith.constant 1 : i32

  // CHECK: cira.offload
  cira.offload attributes {
    region_name = "hashjoin_radix_partition",
    cira.access_pattern = "hash_scatter",
    cira.element_width = 8
  } {
    scf.for %i = %c0_hj to %num_tuples step %c1_hj {
      // Hash the payload to get partition index
      %payload = memref.load %tuples_payload[%i] : memref<?xi32>
      %hash_val = arith.andi %payload, %hash_mask : i32
      %hash_idx = arith.index_cast %hash_val : i32 to index

      // psum[hash_val]++ — load, increment, store
      %cur_psum = memref.load %psum[%hash_idx] : memref<?xi32>
      %write_pos = arith.addi %start_offset, %cur_psum : i32
      %write_idx = arith.index_cast %write_pos : i32 to index
      %next_psum = arith.addi %cur_psum, %one_i32 : i32
      memref.store %next_psum, %psum[%hash_idx] : memref<?xi32>

      // Scatter: partitioned[start + psum[hash]] = tuple
      %key = memref.load %tuples_key[%i] : memref<?xi32>
      memref.store %key, %partitioned_key[%write_idx] : memref<?xi32>
      memref.store %payload, %partitioned_pay[%write_idx] : memref<?xi32>
    }
    cira.yield
  }

  cira.phase_boundary "partition_done"
  return
}

// ════════════════════════════════════════════════════════════════
// Hash Join — Hopscotch Probe Phase (hopscotch.c search())
// ════════════════════════════════════════════════════════════════
// for i in [hash(val)..hash(val)+NEIGHBOURHOOD]:
//   if bucket[i%cap].payload == val: match
//
// CHECK-LABEL: func.func @hashjoin_hopscotch_probe
func.func @hashjoin_hopscotch_probe(
    %bucket_keys: memref<?xi32>,        // hash table bucket keys
    %bucket_payloads: memref<?xi32>,    // hash table bucket payloads
    %probe_keys: memref<?xi32>,         // probe-side keys
    %result_keys: memref<?xi32>,        // matched output keys
    %result_payloads: memref<?xi32>,    // matched output payloads
    %num_probe: index,
    %capacity: i32,                     // hash table capacity
    %neighbourhood: i32                 // hopscotch neighbourhood size (48)
) attributes {
  twopass.region_id = 5 : i32,
  twopass.region_name = "hashjoin_hopscotch_probe"
} {
  %c0_hp = arith.constant 0 : index
  %c1_hp = arith.constant 1 : index

  // CHECK: cira.offload
  cira.offload attributes {
    region_name = "hashjoin_hopscotch_probe",
    cira.access_pattern = "hopscotch_neighbourhood_scan",
    cira.neighbourhood_size = 48,
    cira.element_width = 8
  } {
    scf.for %pi = %c0_hp to %num_probe step %c1_hp {
      %probe_val = memref.load %probe_keys[%pi] : memref<?xi32>

      // Hash → initial bucket index
      %hash_raw = arith.remsi %probe_val, %capacity : i32
      %limit = arith.addi %hash_raw, %neighbourhood : i32

      // Scan neighbourhood [hash .. hash+48)
      %hash_idx = arith.index_cast %hash_raw : i32 to index
      %limit_idx = arith.index_cast %limit : i32 to index

      scf.for %bi = %hash_idx to %limit_idx step %c1_hp {
        // Wrap around: bi % capacity
        %bi_i32 = arith.index_cast %bi : index to i32
        %wrapped = arith.remsi %bi_i32, %capacity : i32
        %widx = arith.index_cast %wrapped : i32 to index

        // Check bucket payload against probe value
        %bkt_pay = memref.load %bucket_payloads[%widx] : memref<?xi32>
        %match = arith.cmpi eq, %bkt_pay, %probe_val : i32
        scf.if %match {
          %bkt_key = memref.load %bucket_keys[%widx] : memref<?xi32>
          memref.store %bkt_key, %result_keys[%pi] : memref<?xi32>
          memref.store %probe_val, %result_payloads[%pi] : memref<?xi32>
        }
      }
    }
    cira.yield
  }

  cira.phase_boundary "probe_done"
  return
}

// ════════════════════════════════════════════════════════════════
// Full Hash Join: partition → build → probe (multi-phase)
// ════════════════════════════════════════════════════════════════
// CHECK-LABEL: func.func @hashjoin_full_pipeline
func.func @hashjoin_full_pipeline(
    %tuples_key: memref<?xi32>, %tuples_payload: memref<?xi32>,
    %partitioned_key: memref<?xi32>, %partitioned_pay: memref<?xi32>,
    %psum: memref<?xi32>,
    %bucket_keys: memref<?xi32>, %bucket_payloads: memref<?xi32>,
    %probe_keys: memref<?xi32>,
    %result_keys: memref<?xi32>, %result_payloads: memref<?xi32>,
    %num_tuples: index, %num_probe: index,
    %hash_mask: i32, %start_offset: i32,
    %capacity: i32, %neighbourhood: i32
) attributes {
  twopass.function_profile = true
} {
  // Phase 1: Radix partition
  func.call @hashjoin_radix_partition(
      %tuples_key, %tuples_payload, %partitioned_key, %partitioned_pay,
      %psum, %num_tuples, %hash_mask, %start_offset)
    : (memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>,
       memref<?xi32>, index, i32, i32) -> ()

  // Phase 2: Hopscotch probe
  func.call @hashjoin_hopscotch_probe(
      %bucket_keys, %bucket_payloads, %probe_keys,
      %result_keys, %result_payloads,
      %num_probe, %capacity, %neighbourhood)
    : (memref<?xi32>, memref<?xi32>, memref<?xi32>,
       memref<?xi32>, memref<?xi32>, index, i32, i32) -> ()

  return
}

}
