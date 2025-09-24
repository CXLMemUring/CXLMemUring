module {
  func.func @reduction(%A: memref<4096xf32>, %out: memref<1xf32>) {
    %c0 = arith.constant 0 : index
    %c4096 = arith.constant 4096 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32
    %res = scf.for %i = %c0 to %c4096 step %c1 iter_args(%acc = %zero) -> (f32) {
      %v = memref.load %A[%i] : memref<4096xf32>
      %nacc = arith.addf %acc, %v : f32
      scf.yield %nacc : f32
    }
    %c0i = arith.constant 0 : index
    memref.store %res, %out[%c0i] : memref<1xf32>
    return
  }
}

