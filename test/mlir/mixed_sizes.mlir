module {
  func.func @mixed_sizes(%Ai8: memref<2048xi8>, %Bi64: memref<2048xi64>, %Cf32: memref<2048xf32>) {
    %c0 = arith.constant 0 : index
    %c2048 = arith.constant 2048 : index
    %c2 = arith.constant 2 : index
    scf.for %i = %c0 to %c2048 step %c2 {
      %b = memref.load %Bi64[%i] : memref<2048xi64>
      %b32 = arith.trunci %b : i64 to i8
      %a = memref.load %Ai8[%i] : memref<2048xi8>
      %sum = arith.addi %a, %b32 : i8
      %sumz = arith.sitofp %sum : i8 to f32
      memref.store %sumz, %Cf32[%i] : memref<2048xf32>
    }
    return
  }
}

