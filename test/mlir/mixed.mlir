module {
  func.func @mixed(%A: memref<1024xf32>, %B: memref<1024xf32>, %C: memref<1024xf32>) {
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c1024 step %c1 {
      %a = memref.load %A[%i] : memref<1024xf32>
      %b = memref.load %B[%i] : memref<1024xf32>
      %s = arith.addf %a, %b : f32
      memref.store %s, %C[%i] : memref<1024xf32>
      %d = arith.subf %s, %a : f32
      memref.store %d, %A[%i] : memref<1024xf32>
    }
    return
  }
}

