// RUN: cira %s --allow-unregistered-dialect \
// RUN:   --cira-twopass-timing="profile=%p/gapbs_profile.json" \
// RUN:   --cira-profile-prefetch | FileCheck %s
//
// GAPBS Benchmark Suite — CIRA dialect lowering from real source
//
// CSR graph storage (graph.h):
//   out_index_[u]   → pointer to first outgoing edge of u
//   out_neighbors_[] → flat edge array (NodeID = i32)
//   in_index_[u]    → pointer to first incoming edge
//   in_neighbors_[] → flat incoming edge array
//
// Access pattern:  for v in g.out_neigh(u):
//   begin = out_index[u], end = out_index[u+1]
//   v = out_neighbors[begin .. end]
//
// Pointer-chasing: vertex u → out_index[u] → out_neighbors[k] → data[v]

module attributes {
  twopass.clock_freq_mhz = 200.0 : f64,
  twopass.cxl_latency_ns = 165 : i64,
  twopass.llc_latency_ns = 15 : i64,
  twopass.sync_overhead_ns = 50 : i64
} {

// ════════════════════════════════════════════════════════════════
// PageRank — Pull-Based Score Update (pr.cc:45-54)
// ════════════════════════════════════════════════════════════════
//
//  for u in [0, num_nodes):
//    incoming = 0
//    for v in g.in_neigh(u):
//      incoming += outgoing_contrib[v]    ← scattered read
//    scores[u] = base_score + kDamp * incoming
//    outgoing_contrib[u] = scores[u] / out_degree[u]
//
// CHECK-LABEL: func.func @gapbs_pagerank_pull
func.func @gapbs_pagerank_pull(
    %in_index: memref<?xi32>,           // CSR in_index array
    %in_neighbors: memref<?xi32>,       // CSR in_neighbors array
    %scores: memref<?xf32>,             // per-vertex scores
    %outgoing_contrib: memref<?xf32>,   // per-vertex contribution
    %out_degree: memref<?xi32>,         // per-vertex out-degree
    %num_nodes: index
) attributes {
  twopass.region_id = 0 : i32,
  twopass.region_name = "gapbs_pagerank_pull"
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %base_score = arith.constant 0.00588235 : f32   // (1-0.85)/N for N~256
  %kDamp = arith.constant 0.85 : f32
  %zero_f = arith.constant 0.0 : f32

  // CHECK: cira.offload
  cira.offload attributes {
    region_name = "gapbs_pagerank_pull",
    cira.access_pattern = "csr_in_neighbor_gather",
    cira.index_array_offset = 0,
    cira.edge_array_offset = 0
  } {
    scf.for %u = %c0 to %num_nodes step %c1 {
      // Load CSR row boundaries: in_index[u], in_index[u+1]
      %u_plus1 = arith.addi %u, %c1 : index
      %row_start_i32 = memref.load %in_index[%u] : memref<?xi32>
      %row_end_i32 = memref.load %in_index[%u_plus1] : memref<?xi32>
      %row_start = arith.index_cast %row_start_i32 : i32 to index
      %row_end = arith.index_cast %row_end_i32 : i32 to index

      // Accumulate incoming contributions from in-neighbors
      %incoming = scf.for %k = %row_start to %row_end step %c1
          iter_args(%acc = %zero_f) -> f32 {
        // v = in_neighbors[k]  — neighbor vertex ID
        %v_i32 = memref.load %in_neighbors[%k] : memref<?xi32>
        %v = arith.index_cast %v_i32 : i32 to index
        // outgoing_contrib[v] — scattered read (the hot access)
        %contrib = memref.load %outgoing_contrib[%v] : memref<?xf32>
        %new_acc = arith.addf %acc, %contrib : f32
        scf.yield %new_acc : f32
      }

      // scores[u] = base_score + kDamp * incoming
      %damped = arith.mulf %kDamp, %incoming : f32
      %new_score = arith.addf %base_score, %damped : f32
      memref.store %new_score, %scores[%u] : memref<?xf32>

      // outgoing_contrib[u] = scores[u] / out_degree[u]
      %deg_i32 = memref.load %out_degree[%u] : memref<?xi32>
      %deg_f = arith.sitofp %deg_i32 : i32 to f32
      %new_contrib = arith.divf %new_score, %deg_f : f32
      memref.store %new_contrib, %outgoing_contrib[%u] : memref<?xf32>
    }
    cira.yield
  }

  cira.phase_boundary "pr_iteration_done"
  return
}

// ════════════════════════════════════════════════════════════════
// BFS — Top-Down Step (bfs.cc:74-84)
// ════════════════════════════════════════════════════════════════
//
//  for u in frontier:
//    for v in g.out_neigh(u):
//      if parent[v] < 0:
//        CAS(parent[v], curr, u)  → mark visited
//        next_frontier.push(v)
//
// CHECK-LABEL: func.func @gapbs_bfs_td_step
func.func @gapbs_bfs_td_step(
    %out_index: memref<?xi32>,
    %out_neighbors: memref<?xi32>,
    %parent: memref<?xi32>,             // -out_degree if unvisited, parent if visited
    %frontier: memref<?xi32>,           // current frontier vertices
    %next_frontier: memref<?xi32>,      // output next frontier
    %frontier_size: index
) attributes {
  twopass.region_id = 1 : i32,
  twopass.region_name = "gapbs_bfs_td_step"
} {
  %c0_td = arith.constant 0 : index
  %c1_td = arith.constant 1 : index
  %neg1 = arith.constant -1 : i32
  %zero_td = arith.constant 0 : i32

  // CHECK: cira.offload
  cira.offload attributes {
    region_name = "gapbs_bfs_td_step",
    cira.access_pattern = "csr_frontier_expand"
  } {
    scf.for %fi = %c0_td to %frontier_size step %c1_td {
      // u = frontier[fi]
      %u_i32 = memref.load %frontier[%fi] : memref<?xi32>
      %u = arith.index_cast %u_i32 : i32 to index

      // CSR row: out_index[u] .. out_index[u+1]
      %u_plus1 = arith.addi %u, %c1_td : index
      %row_start_i32 = memref.load %out_index[%u] : memref<?xi32>
      %row_end_i32 = memref.load %out_index[%u_plus1] : memref<?xi32>
      %row_start = arith.index_cast %row_start_i32 : i32 to index
      %row_end = arith.index_cast %row_end_i32 : i32 to index

      // Expand: for v in out_neigh(u)
      scf.for %k = %row_start to %row_end step %c1_td {
        %v_i32 = memref.load %out_neighbors[%k] : memref<?xi32>
        %v = arith.index_cast %v_i32 : i32 to index

        // if parent[v] < 0 → unvisited
        %pv = memref.load %parent[%v] : memref<?xi32>
        %unvisited = arith.cmpi slt, %pv, %zero_td : i32
        scf.if %unvisited {
          memref.store %u_i32, %parent[%v] : memref<?xi32>
          memref.store %v_i32, %next_frontier[%fi] : memref<?xi32>
        }
      }
    }
    cira.yield
  }

  cira.phase_boundary "bfs_td_done"
  return
}

// ════════════════════════════════════════════════════════════════
// BFS — Bottom-Up Step (bfs.cc:51-60)
// ════════════════════════════════════════════════════════════════
//
//  for u in [0, num_nodes):
//    if parent[u] < 0:
//      for v in g.in_neigh(u):
//        if front_bitmap[v]:
//          parent[u] = v; break
//
// CHECK-LABEL: func.func @gapbs_bfs_bu_step
func.func @gapbs_bfs_bu_step(
    %in_index: memref<?xi32>,
    %in_neighbors: memref<?xi32>,
    %parent: memref<?xi32>,
    %front_bitmap: memref<?xi32>,       // bitmap: 1 bit per vertex
    %num_nodes: index
) attributes {
  twopass.region_id = 2 : i32,
  twopass.region_name = "gapbs_bfs_bu_step"
} {
  %c0_bu = arith.constant 0 : index
  %c1_bu = arith.constant 1 : index
  %zero_i32 = arith.constant 0 : i32

  // CHECK: cira.offload
  cira.offload attributes {
    region_name = "gapbs_bfs_bu_step",
    cira.access_pattern = "full_vertex_scan_with_in_neighbor_probe"
  } {
    scf.for %u = %c0_bu to %num_nodes step %c1_bu {
      %pv = memref.load %parent[%u] : memref<?xi32>
      %unvisited = arith.cmpi slt, %pv, %zero_i32 : i32
      scf.if %unvisited {
        %u_plus1 = arith.addi %u, %c1_bu : index
        %row_start_i32 = memref.load %in_index[%u] : memref<?xi32>
        %row_end_i32 = memref.load %in_index[%u_plus1] : memref<?xi32>
        %row_start = arith.index_cast %row_start_i32 : i32 to index
        %row_end = arith.index_cast %row_end_i32 : i32 to index

        scf.for %k = %row_start to %row_end step %c1_bu {
          %v_i32 = memref.load %in_neighbors[%k] : memref<?xi32>
          %v = arith.index_cast %v_i32 : i32 to index

          // Check front_bitmap[v]
          %bitmask = memref.load %front_bitmap[%v] : memref<?xi32>
          %in_front = arith.cmpi ne, %bitmask, %zero_i32 : i32
          scf.if %in_front {
            memref.store %v_i32, %parent[%u] : memref<?xi32>
          }
        }
      }
    }
    cira.yield
  }

  cira.phase_boundary "bfs_bu_done"
  return
}

// ════════════════════════════════════════════════════════════════
// BFS — Direction-Optimizing Wrapper (bfs.cc:143-159)
// ════════════════════════════════════════════════════════════════
//
// Calls TD step when frontier is small, switches to BU when large
// alpha=15, beta=18 thresholds drive phase switching
//
// CHECK-LABEL: func.func @gapbs_bfs_direction_optimizing
func.func @gapbs_bfs_direction_optimizing(
    %out_index: memref<?xi32>, %out_neighbors: memref<?xi32>,
    %in_index: memref<?xi32>, %in_neighbors: memref<?xi32>,
    %parent: memref<?xi32>,
    %frontier: memref<?xi32>, %next_frontier: memref<?xi32>,
    %front_bitmap: memref<?xi32>,
    %frontier_size: index, %num_nodes: index
) attributes {
  twopass.function_profile = true
} {
  // Phase 1: Top-down step
  func.call @gapbs_bfs_td_step(
      %out_index, %out_neighbors, %parent,
      %frontier, %next_frontier, %frontier_size)
    : (memref<?xi32>, memref<?xi32>, memref<?xi32>,
       memref<?xi32>, memref<?xi32>, index) -> ()

  // Phase switch: TD → BU (when frontier > edges/alpha)
  // Phase 2: Bottom-up step
  func.call @gapbs_bfs_bu_step(
      %in_index, %in_neighbors, %parent,
      %front_bitmap, %num_nodes)
    : (memref<?xi32>, memref<?xi32>, memref<?xi32>,
       memref<?xi32>, index) -> ()

  return
}

// ════════════════════════════════════════════════════════════════
// SSSP — Delta-Stepping Relaxation (sssp.cc:69-84)
// ════════════════════════════════════════════════════════════════
//
//  for u in frontier:
//    for wn in g.out_neigh(u):  // wn = {v, weight}
//      new_dist = dist[u] + wn.w
//      if new_dist < dist[wn.v]:
//        CAS(dist[wn.v], old, new_dist)
//
// CHECK-LABEL: func.func @gapbs_sssp_relax
func.func @gapbs_sssp_relax(
    %out_index: memref<?xi32>,
    %out_neighbors: memref<?xi32>,      // vertex IDs
    %edge_weights: memref<?xi32>,       // edge weights (parallel to neighbors)
    %dist: memref<?xi32>,               // per-vertex distance
    %frontier: memref<?xi32>,
    %frontier_size: index
) attributes {
  twopass.region_id = 3 : i32,
  twopass.region_name = "gapbs_sssp_relax"
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %inf = arith.constant 2147483647 : i32

  // CHECK: cira.offload
  cira.offload attributes {
    region_name = "gapbs_sssp_relax",
    cira.access_pattern = "csr_frontier_relax_weighted"
  } {
    scf.for %fi = %c0 to %frontier_size step %c1 {
      %u_i32 = memref.load %frontier[%fi] : memref<?xi32>
      %u = arith.index_cast %u_i32 : i32 to index
      %dist_u = memref.load %dist[%u] : memref<?xi32>

      // CSR row boundaries
      %u_plus1 = arith.addi %u, %c1 : index
      %rs_i32 = memref.load %out_index[%u] : memref<?xi32>
      %re_i32 = memref.load %out_index[%u_plus1] : memref<?xi32>
      %rs = arith.index_cast %rs_i32 : i32 to index
      %re = arith.index_cast %re_i32 : i32 to index

      scf.for %k = %rs to %re step %c1 {
        %v_i32 = memref.load %out_neighbors[%k] : memref<?xi32>
        %v = arith.index_cast %v_i32 : i32 to index
        %w = memref.load %edge_weights[%k] : memref<?xi32>

        // new_dist = dist[u] + weight
        %new_dist = arith.addi %dist_u, %w : i32

        // Relaxation: if new_dist < dist[v], update
        %old_dist = memref.load %dist[%v] : memref<?xi32>
        %shorter = arith.cmpi slt, %new_dist, %old_dist : i32
        scf.if %shorter {
          memref.store %new_dist, %dist[%v] : memref<?xi32>
        }
      }
    }
    cira.yield
  }

  cira.phase_boundary "sssp_relax_done"
  return
}

// ════════════════════════════════════════════════════════════════
// BC — Forward BFS + Backward Accumulation (bc.cc:67-135)
// ════════════════════════════════════════════════════════════════
//
// Phase 1 (Forward): BFS from source, compute path_counts[v]
//   for u in frontier:
//     for v in g.out_neigh(u):
//       if depths[v] == depth: path_counts[v] += path_counts[u]
//
// Phase 2 (Backward): reverse BFS, accumulate deltas
//   for u at depth d (reverse order):
//     for v in g.out_neigh(u):
//       delta_u += (path_counts[u]/path_counts[v]) * (1+deltas[v])
//     scores[u] += delta_u
//
// CHECK-LABEL: func.func @gapbs_bc_forward
func.func @gapbs_bc_forward(
    %out_index: memref<?xi32>,
    %out_neighbors: memref<?xi32>,
    %depths: memref<?xi32>,
    %path_counts: memref<?xf32>,
    %frontier: memref<?xi32>,
    %frontier_size: index,
    %current_depth: i32
) attributes {
  twopass.region_id = 4 : i32,
  twopass.region_name = "gapbs_bc_forward"
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: cira.offload
  cira.offload attributes {
    region_name = "gapbs_bc_forward",
    cira.access_pattern = "csr_bfs_sigma_propagation"
  } {
    scf.for %fi = %c0 to %frontier_size step %c1 {
      %u_i32 = memref.load %frontier[%fi] : memref<?xi32>
      %u = arith.index_cast %u_i32 : i32 to index
      %pc_u = memref.load %path_counts[%u] : memref<?xf32>

      %u_plus1 = arith.addi %u, %c1 : index
      %rs_i32 = memref.load %out_index[%u] : memref<?xi32>
      %re_i32 = memref.load %out_index[%u_plus1] : memref<?xi32>
      %rs = arith.index_cast %rs_i32 : i32 to index
      %re = arith.index_cast %re_i32 : i32 to index

      scf.for %k = %rs to %re step %c1 {
        %v_i32 = memref.load %out_neighbors[%k] : memref<?xi32>
        %v = arith.index_cast %v_i32 : i32 to index

        // if depths[v] == current_depth: path_counts[v] += path_counts[u]
        %dv = memref.load %depths[%v] : memref<?xi32>
        %at_depth = arith.cmpi eq, %dv, %current_depth : i32
        scf.if %at_depth {
          %old_pc = memref.load %path_counts[%v] : memref<?xf32>
          %new_pc = arith.addf %old_pc, %pc_u : f32
          memref.store %new_pc, %path_counts[%v] : memref<?xf32>
        }
      }
    }
    cira.yield
  }

  cira.phase_boundary "bc_forward_done"
  return
}

// CHECK-LABEL: func.func @gapbs_bc_backward
func.func @gapbs_bc_backward(
    %out_index: memref<?xi32>,
    %out_neighbors: memref<?xi32>,
    %path_counts: memref<?xf32>,
    %deltas: memref<?xf32>,
    %scores: memref<?xf32>,
    %depth_frontier: memref<?xi32>,     // vertices at current depth
    %depth_size: index
) attributes {
  twopass.region_id = 5 : i32,
  twopass.region_name = "gapbs_bc_backward"
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %one_f = arith.constant 1.0 : f32
  %zero_f = arith.constant 0.0 : f32

  // CHECK: cira.offload
  cira.offload attributes {
    region_name = "gapbs_bc_backward",
    cira.access_pattern = "csr_reverse_bfs_accumulate"
  } {
    scf.for %di = %c0 to %depth_size step %c1 {
      %u_i32 = memref.load %depth_frontier[%di] : memref<?xi32>
      %u = arith.index_cast %u_i32 : i32 to index
      %pc_u = memref.load %path_counts[%u] : memref<?xf32>

      %u_plus1 = arith.addi %u, %c1 : index
      %rs_i32 = memref.load %out_index[%u] : memref<?xi32>
      %re_i32 = memref.load %out_index[%u_plus1] : memref<?xi32>
      %rs = arith.index_cast %rs_i32 : i32 to index
      %re = arith.index_cast %re_i32 : i32 to index

      // delta_u = sum over successors: (pc_u/pc_v) * (1 + deltas[v])
      %delta_u = scf.for %k = %rs to %re step %c1
          iter_args(%acc = %zero_f) -> f32 {
        %v_i32 = memref.load %out_neighbors[%k] : memref<?xi32>
        %v = arith.index_cast %v_i32 : i32 to index
        %pc_v = memref.load %path_counts[%v] : memref<?xf32>
        %delta_v = memref.load %deltas[%v] : memref<?xf32>

        %ratio = arith.divf %pc_u, %pc_v : f32
        %one_plus_dv = arith.addf %one_f, %delta_v : f32
        %contrib = arith.mulf %ratio, %one_plus_dv : f32
        %new_acc = arith.addf %acc, %contrib : f32
        scf.yield %new_acc : f32
      }

      memref.store %delta_u, %deltas[%u] : memref<?xf32>
      %old_score = memref.load %scores[%u] : memref<?xf32>
      %new_score = arith.addf %old_score, %delta_u : f32
      memref.store %new_score, %scores[%u] : memref<?xf32>
    }
    cira.yield
  }

  cira.phase_boundary "bc_backward_done"
  return
}

// ════════════════════════════════════════════════════════════════
// CC — Connected Components Link + Compress (cc.cc:41-65)
// ════════════════════════════════════════════════════════════════
//
// Path compression: while comp[n] != comp[comp[n]]: comp[n] = comp[comp[n]]
// This is the deepest pointer-chasing pattern in GAPBS.
//
// CHECK-LABEL: func.func @gapbs_cc_compress
func.func @gapbs_cc_compress(
    %comp: memref<?xi32>,
    %num_nodes: index
) attributes {
  twopass.region_id = 6 : i32,
  twopass.region_name = "gapbs_cc_compress"
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: cira.offload
  cira.offload attributes {
    region_name = "gapbs_cc_compress",
    cira.access_pattern = "union_find_path_compression",
    cira.chain_depth_estimate = 8
  } {
    scf.for %n = %c0 to %num_nodes step %c1 {
      // while comp[n] != comp[comp[n]]: comp[n] = comp[comp[n]]
      %init_cn = memref.load %comp[%n] : memref<?xi32>
      %result = scf.while (%cn = %init_cn) : (i32) -> i32 {
        %cn_idx = arith.index_cast %cn : i32 to index
        %ccn = memref.load %comp[%cn_idx] : memref<?xi32>  // comp[comp[n]]
        %not_root = arith.cmpi ne, %cn, %ccn : i32
        scf.condition(%not_root) %ccn : i32
      } do {
      ^bb0(%ccn: i32):
        // comp[n] = comp[comp[n]]
        memref.store %ccn, %comp[%n] : memref<?xi32>
        scf.yield %ccn : i32
      }
    }
    cira.yield
  }

  cira.phase_boundary "cc_compress_done"
  return
}

// ════════════════════════════════════════════════════════════════
// Full Brandes BC wrapper (forward + backward phases)
// ════════════════════════════════════════════════════════════════
// CHECK-LABEL: func.func @gapbs_bc_brandes
func.func @gapbs_bc_brandes(
    %out_index: memref<?xi32>, %out_neighbors: memref<?xi32>,
    %depths: memref<?xi32>, %path_counts: memref<?xf32>,
    %deltas: memref<?xf32>, %scores: memref<?xf32>,
    %frontier: memref<?xi32>,
    %frontier_size: index, %current_depth: i32,
    %depth_frontier: memref<?xi32>, %depth_size: index
) attributes {
  twopass.function_profile = true
} {
  // Phase 1: Forward BFS (sigma propagation)
  func.call @gapbs_bc_forward(
      %out_index, %out_neighbors, %depths, %path_counts,
      %frontier, %frontier_size, %current_depth)
    : (memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xf32>,
       memref<?xi32>, index, i32) -> ()

  // Phase 2: Backward accumulation (delta propagation)
  func.call @gapbs_bc_backward(
      %out_index, %out_neighbors, %path_counts, %deltas, %scores,
      %depth_frontier, %depth_size)
    : (memref<?xi32>, memref<?xi32>, memref<?xf32>, memref<?xf32>,
       memref<?xf32>, memref<?xi32>, index) -> ()

  return
}

}
