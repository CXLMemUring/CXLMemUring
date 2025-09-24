module {
  func.func @aliasing(%A: memref<1024xf32>) {
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c1024 step %c1 {
      %x = memref.load %A[%i] : memref<1024xf32>
      %y = arith.addf %x, %x : f32
      memref.store %y, %A[%i] : memref<1024xf32>
    }
    return
  }
}

