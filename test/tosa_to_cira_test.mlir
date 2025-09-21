// RUN: cira %s -tosa-to-cira | FileCheck %s

module {

// Test TOSA MatMul conversion to Cira offload
func.func @test_large_matmul(%arg0: tensor<1x1024x1024xf32>, %arg1: tensor<1x1024x1024xf32>) -> tensor<1x1024x1024xf32> {
  // Large matrix multiplication (2B+ FLOPs) should be offloaded
  %0 = "tosa.matmul"(%arg0, %arg1) : (tensor<1x1024x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x1024x1024xf32>
  return %0 : tensor<1x1024x1024xf32>
}
// CHECK-LABEL: func.func @test_large_matmul
// CHECK: cira.offload "matmul"
// CHECK-NOT: tosa.matmul

// Test small MatMul that should NOT be offloaded
func.func @test_small_matmul(%arg0: tensor<1x32x32xf32>, %arg1: tensor<1x32x32xf32>) -> tensor<1x32x32xf32> {
  // Small matrix multiplication should remain local
  %0 = "tosa.matmul"(%arg0, %arg1) : (tensor<1x32x32xf32>, tensor<1x32x32xf32>) -> tensor<1x32x32xf32>
  return %0 : tensor<1x32x32xf32>
}
// CHECK-LABEL: func.func @test_small_matmul
// CHECK: tosa.matmul
// CHECK-NOT: cira.offload

// Test large reduction conversion to Cira offload
func.func @test_large_reduction(%arg0: tensor<1000000xf32>) -> tensor<1xf32> {
  // Large reduction (>100K elements) should be offloaded
  %0 = "tosa.reduce_sum"(%arg0) {axis = 0 : i64} : (tensor<1000000xf32>) -> tensor<1xf32>
  return %0 : tensor<1xf32>
}
// CHECK-LABEL: func.func @test_large_reduction
// CHECK: cira.offload "reduction"
// CHECK-NOT: tosa.reduce_sum

// Test element-wise operations that should NOT be offloaded
func.func @test_elementwise_ops(%arg0: tensor<1024x1024xf32>, %arg1: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
  // Element-wise operations have low computational intensity
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<1024x1024xf32>, tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %1 = "tosa.mul"(%0, %arg1) {shift = 0 : i32} : (tensor<1024x1024xf32>, tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  return %1 : tensor<1024x1024xf32>
}
// CHECK-LABEL: func.func @test_elementwise_ops
// CHECK: tosa.add
// CHECK: tosa.mul
// CHECK-NOT: cira.offload

}