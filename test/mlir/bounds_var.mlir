module {
  func.func @bounds_var(%A: memref<10000xf32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    scf.for %i = %c0 to %n step %c2 {
      %v = memref.load %A[%i] : memref<10000xf32>
      %one = arith.constant 1.0 : f32
      %nv = arith.addf %v, %one : f32
      memref.store %nv, %A[%i] : memref<10000xf32>
    }
    return
  }
}

