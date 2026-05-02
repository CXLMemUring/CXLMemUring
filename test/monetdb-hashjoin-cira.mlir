// RUN: cira %s --cira-twopass-timing="profile=%p/monetdb_profile.json" --cira-profile-prefetch | FileCheck %s
//
// MonetDB Hash Join Probe — CIRA dialect lowering from real gdk_join.c
//
// Original C (gdk_join.c:3183):
//   for (rb = HASHget(hsh, HASHprobe(hsh, v));
//        rb != BUN_NONE;
//        rb = HASHgetlink(hsh, rb)) {
//       if (*(cmp)(v, BUNtail(ri, rb)) == 0)
//           HASHLOOPBODY();
//   }
//
// Pointer-chasing chain:
//   Hash.Bckt[bucket_idx] → first_bun
//   Hash.Link[bun] → next_bun  (repeated — the critical chain)
//   BATiter.base[bun << shift] → column value
//
// CIRA transformation:
//   Vortex prefetches Link[bun] chain ahead of host consumption
//   Host does comparison + result accumulation concurrently
//
// Hardware: Intel Agilex 7 CXL FPGA, Vortex RISC-V @ 200MHz
// Cost model: Gain = depth * (L_CXL - L_LLC) - C_sync
//           = 16 * (165 - 15) - 50 = 2350 ns per offload

module attributes {
  twopass.clock_freq_mhz = 200.0 : f64,
  twopass.cxl_latency_ns = 165 : i64,
  twopass.llc_latency_ns = 15 : i64,
  twopass.sync_overhead_ns = 50 : i64
} {

// ── Hash Join Probe: the primary pointer-chasing hotspot ─────────
//
// MonetDB struct layout (from gdk.h):
//   Hash:    { type:i32, width:i8, ..., Bckt:ptr(off=48), Link:ptr(off=56) }
//   BATiter: { bat:ptr, heap:ptr, base:ptr(off=16), ..., width:i16, shift:i8 }
//
// CHECK-LABEL: func.func @monetdb_hashjoin_probe
func.func @monetdb_hashjoin_probe(
    %link_base: memref<?xi32>,          // Hash.Link array (CXL-resident)
    %bckt_base: memref<?xi32>,          // Hash.Bckt array (CXL-resident)
    %data_base: memref<?xi64>,          // BATiter.base    (CXL-resident)
    %probe_keys: memref<?xi64>,         // probe-side keys (local DRAM)
    %result_oids: memref<?xi64>,        // output OIDs
    %num_probe: index,
    %hash_mask: i32
) attributes {
  twopass.region_id = 0 : i32,
  twopass.region_name = "monetdb_hashjoin_probe"
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %BUN_NONE = arith.constant -1 : i32
  %result_count = arith.constant 0 : index

  // ── Offload region: Vortex prefetches hash chain links ───────
  // The Vortex core chases Link[bun] → next_bun ahead of host,
  // installing each cache line into host LLC via DCOH writeback.
  // CHECK: cira.offload
  cira.offload attributes {
    region_name = "monetdb_hashjoin_probe",
    cira.access_pattern = "hash_chain_walk",
    cira.chain_field_offset = 56,       // offset of Link ptr in Hash struct
    cira.element_width = 4              // sizeof(BUN4type) = 4 bytes
  } {
    // For each probe key, walk the hash chain
    scf.for %probe_idx = %c0 to %num_probe step %c1 {
      // Step 1: Hash the probe key → bucket index
      %key = memref.load %probe_keys[%probe_idx] : memref<?xi64>
      %key_i32 = arith.trunci %key : i64 to i32
      %bucket_idx = arith.andi %key_i32, %hash_mask : i32
      %bucket_idx_idx = arith.index_cast %bucket_idx : i32 to index

      // Step 2: Load first BUN from Bckt[bucket_idx]
      //   This is Bckt + bucket_idx * 4  (width=4 for BUN4)
      %first_bun = memref.load %bckt_base[%bucket_idx_idx] : memref<?xi32>

      // Step 3: Walk hash chain: Link[bun] → next_bun
      //   This is the critical pointer-chasing loop that Vortex prefetches
      %walk_result = scf.while (%current_bun = %first_bun)
          : (i32) -> i32 {
        // Check chain termination
        %not_done = arith.cmpi ne, %current_bun, %BUN_NONE : i32
        scf.condition(%not_done) %current_bun : i32
      } do {
      ^chain_body(%bun: i32):
        %bun_idx = arith.index_cast %bun : i32 to index

        // Load column value at BATiter.base[bun]
        //   data_base + bun * sizeof(i64)
        %data_val = memref.load %data_base[%bun_idx] : memref<?xi64>

        // Compare probe key with stored value
        %match = arith.cmpi eq, %key, %data_val : i64

        // If match, record the OID in result
        scf.if %match {
          %bun_i64 = arith.extsi %bun : i32 to i64
          memref.store %bun_i64, %result_oids[%probe_idx] : memref<?xi64>
        }

        // Chase chain: next_bun = Link[current_bun]
        //   This is the load that Vortex prefetches ahead
        %next_bun = memref.load %link_base[%bun_idx] : memref<?xi32>
        scf.yield %next_bun : i32
      }
    }
    cira.yield
  }

  // ── Sync point: hash join probe complete ───────────────────
  cira.phase_boundary "hashjoin_probe_done"

  return
}

// ── Sequential Selection Scan ────────────────────────────────────
// From gdk_select.c: scanloop macro
// Pattern: dst[cnt] = o  where  v = src[o - hseq]  passes predicate
// This is a streaming access — Vortex uses prefetch_stream, not chain
//
// CHECK-LABEL: func.func @monetdb_selection_scan
func.func @monetdb_selection_scan(
    %src_base: memref<?xi64>,           // column data (CXL-resident)
    %candidate_oids: memref<?xi64>,     // candidate OIDs to check
    %result_oids: memref<?xi64>,        // passing OIDs
    %num_candidates: index,
    %predicate_val: i64,                // selection predicate constant
    %hseq: i64                          // head sequence base
) attributes {
  twopass.region_id = 1 : i32,
  twopass.region_name = "monetdb_selection_scan"
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cnt = arith.constant 0 : index

  // CHECK: cira.offload
  cira.offload attributes {
    region_name = "monetdb_selection_scan",
    cira.access_pattern = "sequential_scan",
    cira.stride = 8                     // sizeof(i64) stride
  } {
    scf.for %i = %c0 to %num_candidates step %c1 {
      // Load candidate OID
      %oid = memref.load %candidate_oids[%i] : memref<?xi64>

      // Compute array index: o - hseq
      %offset = arith.subi %oid, %hseq : i64
      %idx = arith.index_cast %offset : i64 to index

      // Load column value: src[o - hseq]
      %val = memref.load %src_base[%idx] : memref<?xi64>

      // Apply selection predicate
      %passes = arith.cmpi sgt, %val, %predicate_val : i64
      scf.if %passes {
        memref.store %oid, %result_oids[%i] : memref<?xi64>
      }
    }
    cira.yield
  }

  cira.phase_boundary "selection_scan_done"
  return
}

// ── BAT Group Aggregation (BATgroupavg3) ─────────────────────────
// Aggregation-heavy queries (TPC-H Q1) spend ~50% time here.
// Pattern: for each group, accumulate values from column
// Indirect: group_ids[i] → group_idx, then accum[group_idx] += val[i]
//
// CHECK-LABEL: func.func @monetdb_bat_group_avg
func.func @monetdb_bat_group_avg(
    %values: memref<?xf64>,             // source column (CXL-resident)
    %group_ids: memref<?xi32>,          // group assignment per row
    %accum: memref<?xf64>,             // per-group accumulator
    %counts: memref<?xi32>,            // per-group count
    %num_rows: index
) attributes {
  twopass.region_id = 2 : i32,
  twopass.region_name = "monetdb_bat_group_avg"
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %one_i32 = arith.constant 1 : i32

  // CHECK: cira.offload
  cira.offload attributes {
    region_name = "monetdb_bat_group_avg",
    cira.access_pattern = "indirect_scatter",
    cira.chain_depth_estimate = 1       // single-level indirection
  } {
    scf.for %i = %c0 to %num_rows step %c1 {
      // Load value and group ID
      %val = memref.load %values[%i] : memref<?xf64>
      %gid = memref.load %group_ids[%i] : memref<?xi32>
      %gid_idx = arith.index_cast %gid : i32 to index

      // Indirect scatter: accum[group_ids[i]] += values[i]
      //   The group_ids[i] → gid lookup is the indirection
      //   Vortex can prefetch accum[group_ids[i+depth]] ahead
      %old_sum = memref.load %accum[%gid_idx] : memref<?xf64>
      %new_sum = arith.addf %old_sum, %val : f64
      memref.store %new_sum, %accum[%gid_idx] : memref<?xf64>

      // Increment count
      %old_cnt = memref.load %counts[%gid_idx] : memref<?xi32>
      %new_cnt = arith.addi %old_cnt, %one_i32 : i32
      memref.store %new_cnt, %counts[%gid_idx] : memref<?xi32>
    }
    cira.yield
  }

  cira.phase_boundary "aggregation_done"
  return
}

// ── Full TPC-C new_order transaction ─────────────────────────────
// Multi-phase: stock lookup (hash probe) → order insert → payment
// Each phase has distinct access pattern requiring different prefetch
//
// CHECK-LABEL: func.func @monetdb_tpcc_new_order
func.func @monetdb_tpcc_new_order(
    %link_base: memref<?xi32>,
    %bckt_base: memref<?xi32>,
    %stock_data: memref<?xi64>,
    %order_data: memref<?xi64>,
    %probe_keys: memref<?xi64>,
    %result_oids: memref<?xi64>,
    %num_items: index,
    %hash_mask: i32,
    %hseq: i64,
    %predicate_val: i64
) -> i1 attributes {
  twopass.function_profile = true
} {
  %true = arith.constant true

  // Phase 1: Stock level lookup via hash join (pointer-chasing)
  func.call @monetdb_hashjoin_probe(
      %link_base, %bckt_base, %stock_data,
      %probe_keys, %result_oids, %num_items, %hash_mask)
    : (memref<?xi32>, memref<?xi32>, memref<?xi64>,
       memref<?xi64>, memref<?xi64>, index, i32) -> ()

  // Phase 2: Selection scan for availability check (streaming)
  func.call @monetdb_selection_scan(
      %stock_data, %result_oids, %order_data,
      %num_items, %predicate_val, %hseq)
    : (memref<?xi64>, memref<?xi64>, memref<?xi64>,
       index, i64, i64) -> ()

  return %true : i1
}

// ── Cost model verification for MonetDB ──────────────────────────
// Hash chain depth in MonetDB depends on load factor:
//   avg_chain_length = num_tuples / num_buckets
// For TPC-C stock table (100K rows, 64K buckets): avg ~1.5
// For TPC-H lineitem (6M rows, 1M buckets): avg ~6
// Profitable when chain_depth >= ceil(50 / 150) = 1
//
// CHECK-LABEL: func.func @monetdb_cost_check
func.func @monetdb_cost_check(%chain_depth: index) -> i1 {
  %c1 = arith.constant 1 : index
  %should_offload = arith.cmpi sge, %chain_depth, %c1 : index
  return %should_offload : i1
}

}
