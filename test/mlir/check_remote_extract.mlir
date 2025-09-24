// RUN: cira %s --rmem-search-remote | FileCheck %s
module {
  func.func @simple(%A: memref<16xf32>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c16 step %c1 {
      %v = memref.load %A[%i] : memref<16xf32>
      memref.store %v, %A[%i] : memref<16xf32>
    }
    return
  }
}

// CHECK: func.call @remote_access_0
// CHECK: func.func @remote_access_0(
// CHECK: %{{.*}}: index, %{{.*}}: memref<16xf32>
