// Expected output after TOSA to Cira offloading transformation
// This shows how large TOSA operations get converted to Cira offload operations

module @tosa_neural_network_offloaded {
  // Large matrix multiplication converted to Cira offload
  func.func @large_matmul(%lhs: tensor<2048x2048xf32>, %rhs: tensor<2048x2048xf32>) -> tensor<2048x2048xf32> {
    // Converted to offload operation targeting CXL memory
    %result = cira.offload "matmul" (%lhs, %rhs) : (tensor<2048x2048xf32>, tensor<2048x2048xf32>) -> tensor<2048x2048xf32> {
      // Offload body with optimized memory access patterns
      %lhs_memref = memref.alloc() : memref<2048x2048xf32, "cxl_attached">
      %rhs_memref = memref.alloc() : memref<2048x2048xf32, "cxl_attached">
      %result_memref = memref.alloc() : memref<2048x2048xf32, "local_dram">

      // Streaming matrix multiplication with prefetching
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index  // Tile size
      %c2048 = arith.constant 2048 : index

      scf.for %i = %c0 to %c2048 step %c64 {
        scf.for %j = %c0 to %c2048 step %c64 {
          // Load tiles with prefetching
          %lhs_tile = cira.offload.load_edge(%lhs_memref, %i, %c1) : memref<2048x2048xf32, "cxl_attached"> -> memref<64x2048xf32>
          %rhs_tile = cira.offload.load_edge(%rhs_memref, %j, %c1) : memref<2048x2048xf32, "cxl_attached"> -> memref<2048x64xf32>

          // Compute tile
          scf.for %k = %c0 to %c2048 step %c1 {
            // Matrix multiplication inner loop with optimized memory access
          }

          // Evict processed data
          cira.offload.evict_edge %lhs_memref, %i : memref<2048x2048xf32, "cxl_attached">, index
          cira.offload.evict_edge %rhs_memref, %j : memref<2048x2048xf32, "cxl_attached">, index
        }
      }
    }
    return %result : tensor<2048x2048xf32>
  }

  // Large convolution converted to streaming offload
  func.func @large_conv2d(%input: tensor<1x512x512x64xf32>, %weight: tensor<128x3x3x64xf32>, %bias: tensor<128xf32>) -> tensor<1x510x510x128xf32> {
    %result = cira.offload "conv2d" (%input, %weight, %bias) : (tensor<1x512x512x64xf32>, tensor<128x3x3x64xf32>, tensor<128xf32>) -> tensor<1x510x510x128xf32> {
      // Optimized convolution with input streaming from CXL memory
      %input_memref = memref.alloc() : memref<1x512x512x64xf32, "cxl_attached">
      %weight_memref = memref.alloc() : memref<128x3x3x64xf32, "local_dram">  // Weights stay local for reuse
      %result_memref = memref.alloc() : memref<1x510x510x128xf32, "local_dram">

      // Convolution with sliding window access pattern
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %c510 = arith.constant 510 : index

      // Stream input data with prefetching
      scf.for %h = %c0 to %c510 step %c1 {
        scf.for %w = %c0 to %c510 step %c1 {
          // Load input window with prefetch
          %input_window = cira.offload.load_edge(%input_memref, %h, %c3) : memref<1x512x512x64xf32, "cxl_attached"> -> memref<1x3x3x64xf32>

          // Convolution computation (stays local)
          scf.for %oc = %c0 to %c128 step %c1 {
            // Inner convolution loop with weight reuse
          }

          // Evict processed input
          cira.offload.evict_edge %input_memref, %h : memref<1x512x512x64xf32, "cxl_attached">, index
        }
      }
    }
    return %result : tensor<1x510x510x128xf32>
  }

  // Large reduction with streaming access
  func.func @large_reduce(%input: tensor<1024x1024x1024xf32>) -> tensor<1024x1024xf32> {
    %result = cira.offload "reduce_sum" (%input) : (tensor<1024x1024x1024xf32>) -> tensor<1024x1024xf32> {
      // Hierarchical reduction across memory tiers
      %input_memref = memref.alloc() : memref<1024x1024x1024xf32, "far_memory">  // Large data in far memory
      %temp_memref = memref.alloc() : memref<1024x1024xf32, "cxl_attached">     // Intermediate results in CXL
      %result_memref = memref.alloc() : memref<1024x1024xf32, "local_dram">     // Final result local

      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1024 = arith.constant 1024 : index

      // Stream reduction with memory tier hierarchy
      scf.for %i = %c0 to %c1024 step %c1 {
        scf.for %j = %c0 to %c1024 step %c1 {
          // Load slice from far memory
          %slice = cira.offload.load_edge(%input_memref, %i, %c1) : memref<1024x1024x1024xf32, "far_memory"> -> memref<1024xf32>

          // Reduce to CXL memory
          %partial = arith.constant 0.0 : f32
          scf.for %k = %c0 to %c1024 step %c1 {
            %val = memref.load %slice[%k] : memref<1024xf32>
            %partial = arith.addf %partial, %val : f32
          }
          memref.store %partial, %temp_memref[%i, %j] : memref<1024x1024xf32, "cxl_attached">

          // Evict processed slice
          cira.offload.evict_edge %input_memref, %i : memref<1024x1024x1024xf32, "far_memory">, index
        }
      }

      // Copy final result to local memory
      memref.copy %temp_memref, %result_memref : memref<1024x1024xf32, "cxl_attached"> to memref<1024x1024xf32, "local_dram">
    }
    return %result : tensor<1024x1024xf32>
  }

  // Small operations remain unchanged (no offloading)
  func.func @small_elementwise(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>) -> tensor<64x64xf32> {
    // These stay as TOSA operations - too small to benefit from offloading
    %sum = tosa.add %a, %b : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    %scaled = tosa.mul %sum, %sum : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %scaled : tensor<64x64xf32>
  }

  // Chained operations get grouped for efficient data flow
  func.func @chained_matmuls(%a: tensor<1024x1024xf32>, %b: tensor<1024x1024xf32>, %c: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
    // Grouped offload operation to minimize data movement
    %result = cira.offload "chained_matmul" (%a, %b, %c) : (tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
      // Optimized pipeline keeping intermediate results in CXL memory
      %a_memref = memref.alloc() : memref<1024x1024xf32, "cxl_attached">
      %b_memref = memref.alloc() : memref<1024x1024xf32, "cxl_attached">
      %c_memref = memref.alloc() : memref<1024x1024xf32, "cxl_attached">
      %ab_memref = memref.alloc() : memref<1024x1024xf32, "cxl_attached">  // Intermediate stays in CXL
      %result_memref = memref.alloc() : memref<1024x1024xf32, "local_dram">

      // First matmul: A * B
      // ... optimized matrix multiplication implementation ...

      // Second matmul: (A*B) * C, reusing intermediate result
      // ... optimized matrix multiplication implementation ...
    }
    return %result : tensor<1024x1024xf32>
  }

  // Mixed workload with selective offloading
  func.func @mixed_workload(%input: tensor<256x256x256xf32>, %weights: tensor<512x3x3x256xf32>) -> tensor<254x254x512xf32> {
    // Only the large convolution gets offloaded
    %conv_result = cira.offload "conv2d" (%input, %weights, %bias) : (tensor<256x256x256xf32>, tensor<512x3x3x256xf32>, tensor<512xf32>) -> tensor<254x254x512xf32> {
      // Optimized convolution implementation
    }

    // Activation function remains as TOSA (local operation)
    %activated = tosa.clamp %conv_result {
      min_fp = 0.0 : f32,
      max_fp = 6.0 : f32,
      min_int = 0 : i64,
      max_int = 0 : i64
    } : (tensor<254x254x512xf32>) -> tensor<254x254x512xf32>

    return %activated : tensor<254x254x512xf32>
  }
}