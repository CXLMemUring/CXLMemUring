module {
  func.func @nested_shapes(%M: memref<128x256xf32>, %N: memref<128x256xf32>, %R: memref<128x256xf32>) {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    scf.for %i = %c0 to %c128 step %c4 {
      scf.for %j = %c0 to %c256 step %c3 {
        %a = memref.load %M[%i, %j] : memref<128x256xf32>
        %b = memref.load %N[%i, %j] : memref<128x256xf32>
        %s = arith.addf %a, %b : f32
        memref.store %s, %R[%i, %j] : memref<128x256xf32>
      }
    }
    return
  }
}

