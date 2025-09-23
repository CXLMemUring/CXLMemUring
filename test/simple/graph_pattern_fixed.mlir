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
        
        // Store to nodes (using direct indices for simplicity)
        memref.store %doubled, %nodes[%idx] : memref<1000xf32>
      }
    }
    
    return
  }
  
  func.func @matrix_multiply(%A: memref<100x100xf32>, %B: memref<100x100xf32>, %C: memref<100x100xf32>) {
    %c0 = arith.constant 0 : index
    %c100 = arith.constant 100 : index
    %c1 = arith.constant 1 : index
    %f0 = arith.constant 0.0 : f32
    
    scf.for %i = %c0 to %c100 step %c1 {
      scf.for %j = %c0 to %c100 step %c1 {
        %sum = scf.for %k = %c0 to %c100 step %c1 iter_args(%acc = %f0) -> f32 {
          %a_val = memref.load %A[%i, %k] : memref<100x100xf32>
          %b_val = memref.load %B[%k, %j] : memref<100x100xf32>
          %prod = arith.mulf %a_val, %b_val : f32
          %new_acc = arith.addf %acc, %prod : f32
          scf.yield %new_acc : f32
        }
        memref.store %sum, %C[%i, %j] : memref<100x100xf32>
      }
    }
    
    return
  }
}
