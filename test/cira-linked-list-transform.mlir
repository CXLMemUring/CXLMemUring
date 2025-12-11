// RUN: cira-opt %s -cir-to-cira | FileCheck %s

// Test the linked list pointer-chasing transformation from the paper (Listing 1):
//
// Original High-Level Logic:
//   while (node) {
//     val = node->data;
//     node = node->next;
//   }
//
// Transformed CIRA IR:
//   %stream = cira.stream_create_indirect %start_node, offset=8 : !cira.stream
//   cira.offload_start @vortex_core_0 {
//     cira.prefetch_chain %stream, depth=16
//   }
//   %loop:
//     %future = cira.peek_stream %stream
//     %data = cira.await %future : !cira.future
//     // Computation on %data...
//     cira.advance_stream %stream
//     br %loop

module {

// Test 1: Basic linked list traversal pattern
// CHECK-LABEL: func.func @linked_list_traverse
func.func @linked_list_traverse(%start: memref<?xi64>) -> i64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init_sum = arith.constant 0 : i64

  // CHECK: cira.stream_create_indirect
  // CHECK: cira.offload_start
  // CHECK: cira.prefetch_chain
  %result = scf.while (%ptr = %start, %sum = %init_sum) : (memref<?xi64>, i64) -> i64 {
    // Condition: check if pointer is valid (simplified)
    %size = memref.dim %ptr, %c0 : memref<?xi64>
    %has_more = arith.cmpi sgt, %size, %c0 : index
    scf.condition(%has_more) %sum : i64
  } do {
  ^bb0(%sum_arg: i64):
    // Load data from current node
    %data = memref.load %start[%c0] : memref<?xi64>

    // Add to sum
    %new_sum = arith.addi %sum_arg, %data : i64

    // CHECK: cira.peek_stream
    // CHECK: cira.future_await
    // CHECK: cira.advance_stream
    scf.yield %new_sum : i64
  }

  return %result : i64
}

// Test 2: Graph edge traversal (MCF-like pattern)
// CHECK-LABEL: func.func @graph_edge_traverse
func.func @graph_edge_traverse(%edges: memref<?x3xi64>, %num_edges: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  // CHECK: cira.offload
  scf.for %i = %c0 to %num_edges step %c1 {
    // Load edge: (from, to, weight)
    // CHECK: cira.prefetch_stream
    %from = memref.load %edges[%i, %c0] : memref<?x3xi64>
    %to = memref.load %edges[%i, %c1] : memref<?x3xi64>
    %weight = memref.load %edges[%i, %c2] : memref<?x3xi64>

    // Process edge (simplified)
    %sum = arith.addi %from, %to : i64
    %result = arith.addi %sum, %weight : i64

    // Nested loop for neighbor access (triggers graph pattern detection)
    scf.for %j = %c0 to %num_edges step %c1 {
      %neighbor = memref.load %edges[%j, %c0] : memref<?x3xi64>
    }
  }

  return
}

// Test 3: Demonstrate CIRA types directly
// CHECK-LABEL: func.func @cira_types_demo
func.func @cira_types_demo(%ptr: !cira.handle<i64>) -> i64 {
  // Async load returning a future
  // CHECK: cira.load_async
  %future = cira.load_async %ptr : !cira.handle<i64> -> !cira.future<i64>

  // Independent computation can happen here
  %c42 = arith.constant 42 : i64

  // Await the future to get the value
  // CHECK: cira.future_await
  %val = cira.future_await %future : !cira.future<i64> -> i64

  %result = arith.addi %val, %c42 : i64
  return %result : i64
}

// Test 4: Stream-based prefetching
// CHECK-LABEL: func.func @stream_prefetch_demo
func.func @stream_prefetch_demo(%start: !cira.handle<i64>) {
  // Create stream for indirect access pattern (linked list with next at offset 8)
  // CHECK: cira.stream_create_indirect
  %stream = cira.stream_create_indirect %start, offset=8 : !cira.handle<i64> -> !cira.stream<i64, offset=8>

  // Offload prefetching to Vortex accelerator
  // CHECK: cira.offload_start
  cira.offload_start @vortex_core_0 {
    // Chase pointers 16 steps ahead
    // CHECK: cira.prefetch_chain
    cira.prefetch_chain %stream, depth=16 : !cira.stream<i64, offset=8>
  }

  // Synchronize
  // CHECK: cira.barrier
  cira.barrier

  return
}

// Test 5: Full offload region with computation
// CHECK-LABEL: func.func @offload_region_demo
func.func @offload_region_demo(%data: !cira.handle<f32>, %n: index) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f32

  // CHECK: cira.offload
  %result = cira.offload {
    // This computation runs near CXL memory
    %sum = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> f32 {
      %fut = cira.load_async %data[%i] : !cira.handle<f32> -> !cira.future<f32>
      %val = cira.future_await %fut : !cira.future<f32> -> f32
      %new_acc = arith.addf %acc, %val : f32
      scf.yield %new_acc : f32
    }
    cira.yield %sum : f32
  } : f32

  return %result : f32
}

}
