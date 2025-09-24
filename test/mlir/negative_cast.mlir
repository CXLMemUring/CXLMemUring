module {
  func.func @bad_cast(%A: memref<10xf32>) {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c10 step %c1 {
      %v = memref.load %A[%i] : memref<10xf32>
      // Intentionally malformed: fptosi to index (should go via i32)
      %bad = arith.fptosi %v : f32 to index
      %idx = arith.index_cast %bad : index to index
      %zero = arith.constant 0.0 : f32
      memref.store %zero, %A[%idx] : memref<10xf32>
    }
    return
  }
}

