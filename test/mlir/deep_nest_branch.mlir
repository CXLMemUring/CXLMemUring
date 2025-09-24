module {
  func.func @deep_branch(%A: memref<64x32x16xf32>, %B: memref<64x32x16xf32>) {
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
          %a = memref.load %A[%i, %j, %k] : memref<64x32x16xf32>
          %b = memref.load %B[%i, %j, %k] : memref<64x32x16xf32>
          %cmp = arith.cmpf ogt, %a, %b : f32
          scf.if %cmp {
            %s = arith.addf %a, %b : f32
            memref.store %s, %A[%i, %j, %k] : memref<64x32x16xf32>
          } else {
            %d = arith.subf %b, %a : f32
            memref.store %d, %B[%i, %j, %k] : memref<64x32x16xf32>
          }
        }
      }
    }
    return
  }
}

