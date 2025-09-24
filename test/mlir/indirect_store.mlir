module {
  func.func @indirect_store(%idxs: memref<10000x2xi32>, %dst: memref<10000xf32>, %src: memref<10000xf32>) {
    %c0 = arith.constant 0 : index
    %c10000 = arith.constant 10000 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c10000 step %c1 {
      %base_i32 = memref.load %idxs[%i, %c0] : memref<10000x2xi32>
      %off_i32  = memref.load %idxs[%i, %c1] : memref<10000x2xi32>
      %sum_i32 = arith.addi %base_i32, %off_i32 : i32
      %idx = arith.index_cast %sum_i32 : i32 to index
      %v = memref.load %src[%i] : memref<10000xf32>
      memref.store %v, %dst[%idx] : memref<10000xf32>
    }
    return
  }
}

