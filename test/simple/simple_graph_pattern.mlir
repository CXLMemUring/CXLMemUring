module {
  func.func @graph_traversal(%edges: memref<1000x3xf32>, %nodes: memref<1000xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c1000 = arith.constant 1000 : index
    %c8 = arith.constant 8 : index
    
    // Simple loop without results
    scf.for %i = %c0 to %c1000 step %c8 {
      %weight = memref.load %edges[%i, %c2] : memref<1000x3xf32>
      %doubled = arith.mulf %weight, %weight : f32
      memref.store %doubled, %nodes[%i] : memref<1000xf32>
    }
    
    return
  }
}
