// RUN: polygeist-opt %s --convert-cira-to-llvm | FileCheck %s

// This test demonstrates the usage of the cira dialect for graph traversal
module {
  // Define the edge structure type
  !edge_type = !llvm.struct<"edge", (i32, i32)>
  !remotable_edge = !cira.remotable<!edge_type>
  
  // Graph traversal function using cira dialect
  func.func @trvs_graph_opt(%arg0: !remotable_edge) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %num_edges = arith.constant 1000 : index
    %elements_per_line = arith.constant 16 : index
    %n_ahead = arith.constant 8 : index
    %n_ahead_node = arith.constant 4 : index
    
    // Outer loop over cache lines
    scf.for %i = %c0 to %num_edges step %elements_per_line {
      // Offload prefetching of a cache line from far memory
      cira.offload.load_edge(%arg0, %i, %n_ahead) : !edge_type
      
      // Inner loop over elements in cache line
      scf.for %j = %c0 to %elements_per_line step %c1 {
        // Calculate actual index
        %idx = arith.addi %i, %j : index
        
        // Load the edge element via the offload abstraction
        %edge_element = cira.offload.load_edge(%arg0, %idx) : !edge_type
        
        // Offload the prefetching of the corresponding node elements
        %node_from = cira.offload.load_node(%edge_element, "from", %n_ahead_node) : i32
        %node_to = cira.offload.load_node(%edge_element, "to", %n_ahead_node) : i32
        
        // Wait and obtain the physical addresses of node data
        %paddr_from = cira.offload.get_paddr("from", %node_from) : !llvm.ptr
        %paddr_to = cira.offload.get_paddr("to", %node_to) : !llvm.ptr
        
        // Call an update function that works on the node data
        cira.call @update_node(%edge_element, %paddr_from, %paddr_to) : (!edge_type, !llvm.ptr, !llvm.ptr) -> ()
      }
      
      // Provide an eviction hint for the processed edge cache line
      cira.offload.evict_edge(%arg0, %i)
    }
    
    return
  }
  
  // Dummy update function declaration
  func.func @update_node(%edge: !edge_type, %from_ptr: !llvm.ptr, %to_ptr: !llvm.ptr) {
    return
  }
}

// CHECK-LABEL: @trvs_graph_opt
// CHECK: llvm.mlir.constant
// CHECK: llvm.getelementptr
// CHECK: llvm.load
// CHECK: llvm.extractvalue
// CHECK: llvm.call @update_node