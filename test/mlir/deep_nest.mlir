module {
  func.func @deep(%A: memref<64x32x16xf32>) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c64 step %c2 {
      scf.for %j = %c0 to %c32 step %c3 {
        scf.for %k = %c0 to %c16 step %c4 {
          %v = memref.load %A[%i, %j, %k] : memref<64x32x16xf32>
          %one = arith.constant 1.0 : f32
          %nv = arith.addf %v, %one : f32
          memref.store %nv, %A[%i, %j, %k] : memref<64x32x16xf32>
        }
      }
    }
    return
  }
}

