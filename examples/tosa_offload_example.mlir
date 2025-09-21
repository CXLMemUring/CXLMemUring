// Example TOSA program that demonstrates offloading opportunities
// to Cira remote memory operations

module @tosa_neural_network {
  // Large matrix multiplication - candidate for CXL memory offloading
  func.func @large_matmul(%lhs: tensor<2048x2048xf32>, %rhs: tensor<2048x2048xf32>) -> tensor<2048x2048xf32> {
    // This will be converted to cira.offload "matmul" operation
    %result = tosa.matmul %lhs, %rhs : (tensor<2048x2048xf32>, tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
    return %result : tensor<2048x2048xf32>
  }

  // Large convolution - candidate for remote memory streaming
  func.func @large_conv2d(%input: tensor<1x512x512x64xf32>, %weight: tensor<128x3x3x64xf32>, %bias: tensor<128xf32>) -> tensor<1x510x510x128xf32> {
    // High memory footprint convolution suitable for CXL offloading
    %result = tosa.conv2d %input, %weight, %bias {
      dilation = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 0, 0>,
      stride = array<i64: 1, 1>
    } : (tensor<1x512x512x64xf32>, tensor<128x3x3x64xf32>, tensor<128xf32>) -> tensor<1x510x510x128xf32>
    return %result : tensor<1x510x510x128xf32>
  }

  // Large reduction - candidate for streaming remote access
  func.func @large_reduce(%input: tensor<1024x1024x1024xf32>) -> tensor<1024x1024xf32> {
    // Large reduction operation with streaming access pattern
    %result = tosa.reduce_sum %input {axis = 2 : i32} : (tensor<1024x1024x1024xf32>) -> tensor<1024x1024xf32>
    return %result : tensor<1024x1024xf32>
  }

  // Small operations - should remain local
  func.func @small_elementwise(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>) -> tensor<64x64xf32> {
    // Small elementwise operations stay in local memory
    %sum = tosa.add %a, %b : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    %scaled = tosa.mul %sum, %sum : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %scaled : tensor<64x64xf32>
  }

  // Chained operations - analyze for grouping
  func.func @chained_matmuls(%a: tensor<1024x1024xf32>, %b: tensor<1024x1024xf32>, %c: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
    // Chain of matrix multiplications - candidates for grouped offloading
    %ab = tosa.matmul %a, %b : (tensor<1024x1024xf32>, tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %abc = tosa.matmul %ab, %c : (tensor<1024x1024xf32>, tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    return %abc : tensor<1024x1024xf32>
  }

  // Mixed workload - some operations offloadable, others local
  func.func @mixed_workload(%input: tensor<256x256x256xf32>, %weights: tensor<512x3x3x256xf32>) -> tensor<254x254x512xf32> {
    // Large convolution - offload candidate
    %conv_result = tosa.conv2d %input, %weights, %bias {
      dilation = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 0, 0>,
      stride = array<i64: 1, 1>
    } : (tensor<256x256x256xf32>, tensor<512x3x3x256xf32>, tensor<512xf32>) -> tensor<254x254x512xf32>

    // Activation function - stays local
    %activated = tosa.clamp %conv_result {
      min_fp = 0.0 : f32,
      max_fp = 6.0 : f32,
      min_int = 0 : i64,
      max_int = 0 : i64
    } : (tensor<254x254x512xf32>) -> tensor<254x254x512xf32>

    return %activated : tensor<254x254x512xf32>
  }

  // Entry point demonstrating full neural network layer
  func.func @neural_network_layer(%input: tensor<1x224x224x3xf32>) -> tensor<1x112x112x64xf32> {
    %weights = arith.constant dense<1.0> : tensor<64x7x7x3xf32>
    %bias = arith.constant dense<0.1> : tensor<64xf32>

    // Large convolution with stride - offload to CXL memory
    %conv = tosa.conv2d %input, %weights, %bias {
      dilation = array<i64: 1, 1>,
      pad = array<i64: 3, 3, 3, 3>,
      stride = array<i64: 2, 2>
    } : (tensor<1x224x224x3xf32>, tensor<64x7x7x3xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>

    // Batch normalization - local operations
    %mean = arith.constant dense<0.0> : tensor<64xf32>
    %variance = arith.constant dense<1.0> : tensor<64xf32>
    %scale = arith.constant dense<1.0> : tensor<64xf32>
    %offset = arith.constant dense<0.0> : tensor<64xf32>

    // ReLU activation - local operation
    %relu = tosa.clamp %conv {
      min_fp = 0.0 : f32,
      max_fp = 3.4028235e+38 : f32,
      min_int = 0 : i64,
      max_int = 0 : i64
    } : (tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>

    return %relu : tensor<1x112x112x64xf32>
  }
}