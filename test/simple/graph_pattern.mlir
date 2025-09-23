module {
  func.func @graph_traversal(%edges: memref<1000x3xf32>, %nodes: memref<1000xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c1000 = arith.constant 1000 : index
    %c8 = arith.constant 8 : index
    
    // Outer loop with cache-line-sized step (potential graph traversal pattern)
    scf.for %i = %c0 to %c1000 step %c8 {
      // Inner loop for edge processing
      scf.for %j = %c0 to %c8 step %c1 {
        %idx = arith.addi %i, %j : index
        
        // Load edge data (from, to, weight pattern)
        %from_val = memref.load %edges[%idx, %c0] : memref<1000x3xf32>
        %to_val = memref.load %edges[%idx, %c1] : memref<1000x3xf32>
        %weight = memref.load %edges[%idx, %c2] : memref<1000x3xf32>
        
        // Process edge weight
        %doubled = arith.mulf %weight, %weight : f32
        
        // Store to nodes (graph access pattern)
        %from_idx = arith.fptosi %from_val : f32 to index
        %to_idx = arith.fptosi %to_val : f32 to index
        
        memref.store %doubled, %nodes[%from_idx] : memref<1000xf32>
        memref.store %weight, %nodes[%to_idx] : memref<1000xf32>
      }
    }
    
    return
  }
}
