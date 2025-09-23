module {
  func.func @test_offload(%arg0: memref<100xf32>, %arg1: memref<100xf32>) -> memref<100xf32> {
    %c0 = arith.constant 0 : index
    %c100 = arith.constant 100 : index
    %c1 = arith.constant 1 : index
    
    scf.for %i = %c0 to %c100 step %c1 {
      %val0 = memref.load %arg0[%i] : memref<100xf32>
      %val1 = memref.load %arg1[%i] : memref<100xf32>
      %sum = arith.addf %val0, %val1 : f32
      memref.store %sum, %arg0[%i] : memref<100xf32>
    }
    
    return %arg0 : memref<100xf32>
  }
}
