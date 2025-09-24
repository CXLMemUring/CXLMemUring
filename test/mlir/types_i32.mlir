module {
  func.func @i32_graph(%edges: memref<10000x3xi32>, %nodes: memref<10000xi32>) {
    %c0 = arith.constant 0 : index
    %c10000 = arith.constant 10000 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %i = %c0 to %c10000 step %c1 {
      %w_i32 = memref.load %edges[%i, %c2] : memref<10000x3xi32>
      %from_idx_i32 = memref.load %edges[%i, %c0] : memref<10000x3xi32>
      %to_idx_i32 = memref.load %edges[%i, %c1] : memref<10000x3xi32>
      %from_idx = arith.index_cast %from_idx_i32 : i32 to index
      %to_idx = arith.index_cast %to_idx_i32 : i32 to index
      %from_val = memref.load %nodes[%from_idx] : memref<10000xi32>
      %prod = arith.muli %from_val, %w_i32 : i32
      memref.store %prod, %nodes[%to_idx] : memref<10000xi32>
    }
    return
  }
}

