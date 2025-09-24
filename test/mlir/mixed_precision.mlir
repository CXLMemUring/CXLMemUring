module {
  func.func @mixed_precision(%A: memref<512xf32>, %B: memref<512xf64>, %C: memref<512xf64>) {
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c512 step %c1 {
      %a = memref.load %A[%i] : memref<512xf32>
      %a64 = arith.extf %a : f32 to f64
      %b = memref.load %B[%i] : memref<512xf64>
      %s = arith.addf %a64, %b : f64
      memref.store %s, %C[%i] : memref<512xf64>
    }
    return
  }
}

