module {
  func.func @f64_graph(%edges: memref<10000x3xf64>, %nodes: memref<10000xf64>) {
    %c0 = arith.constant 0 : index
    %c10000 = arith.constant 10000 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %i = %c0 to %c10000 step %c1 {
      %w = memref.load %edges[%i, %c2] : memref<10000x3xf64>
      %from_idx_f = memref.load %edges[%i, %c0] : memref<10000x3xf64>
      %to_idx_f   = memref.load %edges[%i, %c1] : memref<10000x3xf64>
      %from_i32 = arith.fptosi %from_idx_f : f64 to i32
      %to_i32   = arith.fptosi %to_idx_f   : f64 to i32
      %from_idx = arith.index_cast %from_i32 : i32 to index
      %to_idx   = arith.index_cast %to_i32   : i32 to index
      %from_val = memref.load %nodes[%from_idx] : memref<10000xf64>
      %prod = arith.mulf %from_val, %w : f64
      memref.store %prod, %nodes[%to_idx] : memref<10000xf64>
    }
    return
  }
}

