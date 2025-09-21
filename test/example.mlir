// Example MLIR code using the Cira dialect for graph processing

func.func @traverse_graph_opt(%arg0: !cira.remotable<struct<edge>>) {
  %c0 = arith.constant 0 : index
  %num_edges = arith.constant 1000 : index
  %elements_per_line = arith.constant 8 : index
  %n_ahead = arith.constant 2 : index
  %n_ahead_node = arith.constant 1 : index

  scf.for %i = %c0 to %num_edges step %elements_per_line {
    // Offload prefetching of a cache line from far memory
    cira.offload.load_edge(%arg0, %i, %n_ahead) : (!cira.remotable<struct<edge>>, index, index) -> ()

    scf.for %j = %c0 to %elements_per_line step %c1 {
      // Load the edge element via the offload abstraction
      %edge_element = cira.offload.load_edge(%arg0, %i, %j) : (!cira.remotable<struct<edge>>, index, index) -> !cira.edge

      // Offload the prefetching of the corresponding node elements
      %node_from = cira.offload.load_node(%edge_element, "from", %n_ahead_node) : (!cira.edge, index) -> !cira.node
      %node_to = cira.offload.load_node(%edge_element, "to", %n_ahead_node) : (!cira.edge, index) -> !cira.node

      // Wait and obtain the physical addresses of node data
      %paddr_from = cira.offload.get_paddr("from", %node_from) : (!cira.node) -> !llvm.ptr
      %paddr_to = cira.offload.get_paddr("to", %node_to) : (!cira.node) -> !llvm.ptr

      // Call an update function that works on the node data
      cira.call @update_node(%edge_element, %paddr_from, %paddr_to) : (!cira.edge, !llvm.ptr, !llvm.ptr) -> ()
    }

    // Provide an eviction hint for the processed edge cache line
    cira.offload.evict_edge(%arg0, %i) : (!cira.remotable<struct<edge>>, index) -> ()
  }
  return
}

// Update function that operates on node data
func.func private @update_node(%edge: !cira.edge, %from: !llvm.ptr, %to: !llvm.ptr) -> ()