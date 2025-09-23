module {
  func.func @graph_traversal(%arg0: memref<1000x3xf32>, %arg1: memref<1000xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c1000 = arith.constant 1000 : index
    %c8 = arith.constant 8 : index
    scf.for %arg2 = %c0 to %c1000 step %c8 {
      func.call @remote_access_0(%arg2, %arg0, %c2, %arg1, %c0, %c1000, %c8) : (index, memref<1000x3xf32>, index, memref<1000xf32>, index, index, index) -> ()
    }
    return
  }
  func.func @remote_access_0(%arg0: index, %arg1: memref<1000x3xf32>, %arg2: index, %arg3: memref<1000xf32>, %arg4: index, %arg5: index, %arg6: index) {
    %0 = memref.load %arg1[%arg0, %arg2] : memref<1000x3xf32>
    %1 = arith.mulf %0, %0 : f32
    memref.store %1, %arg3[%arg0] : memref<1000xf32>
    return
  }
}

