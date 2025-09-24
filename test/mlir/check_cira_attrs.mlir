// RUN: cira %s --convert-target-to-remote | FileCheck %s
module {
  func.func @attrs(%A: memref<128x3xf32>) {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %i = %c0 to %c128 step %c1 {
      %v = memref.load %A[%i, %c2] : memref<128x3xf32>
      memref.store %v, %A[%i, %c2] : memref<128x3xf32>
    }
    return
  }
}

// Conservative check of the IR content for this simple case.
// CHECK: memref.load {{.*}} : memref<128x3xf32>
