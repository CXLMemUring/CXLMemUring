// RUN: cira %s --cira-region-formation='require-profitable=false' | FileCheck %s

func.func private @host_effect()
func.func private @host_index() -> index

// Two roots share the index load and cast, so their slices merge once.
// CHECK-LABEL: func.func @merge_same_loop
// CHECK-COUNT-1: cira.offload
// CHECK-SAME: cira.overlap_ops = 2
// CHECK-SAME: cira.root_count = 2
// CHECK: {{^ *\} *$}}
// CHECK-NEXT: %{{.*}} = memref.load %arg0
// CHECK-NEXT: %{{.*}} = arith.index_cast
// CHECK-NEXT: cira.barrier
// CHECK-NEXT: %{{.*}} = memref.load %arg1
// CHECK-NOT: cira.barrier
func.func @merge_same_loop(%idx: memref<?xi64, 1>, %data: memref<?xi64, 1>,
                           %n: index) -> i64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0 : i64
  %result = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (i64) {
    %next = memref.load %idx[%i] : memref<?xi64, 1>
    %next_i = arith.index_cast %next : i64 to index
    %first = memref.load %data[%next_i] : memref<?xi64, 1>
    %second_i = arith.addi %next_i, %c1 : index
    %second = memref.load %data[%second_i] : memref<?xi64, 1>
    %sum = arith.addi %first, %second : i64
    scf.yield %sum : i64
  }
  return %result : i64
}

// CHECK-LABEL: func.func @call_blocks_merge
// CHECK-COUNT-2: cira.offload
func.func @call_blocks_merge(%idx: memref<?xi64, 1>, %data: memref<?xi64, 1>,
                             %i: index) -> i64 {
  %next = memref.load %idx[%i] : memref<?xi64, 1>
  %next_i = arith.index_cast %next : i64 to index
  %first = memref.load %data[%next_i] : memref<?xi64, 1>
  func.call @host_effect() : () -> ()
  %c1 = arith.constant 1 : index
  %second_i = arith.addi %next_i, %c1 : index
  %second = memref.load %data[%second_i] : memref<?xi64, 1>
  %sum = arith.addi %first, %second : i64
  return %sum : i64
}

// CHECK-LABEL: func.func @store_blocks_merge
// CHECK-COUNT-2: cira.offload
func.func @store_blocks_merge(%idx: memref<?xi64, 1>, %data: memref<?xi64, 1>,
                              %local: memref<?xi64>, %i: index) -> i64 {
  %next = memref.load %idx[%i] : memref<?xi64, 1>
  %next_i = arith.index_cast %next : i64 to index
  %first = memref.load %data[%next_i] : memref<?xi64, 1>
  memref.store %first, %local[%i] : memref<?xi64>
  %c1 = arith.constant 1 : index
  %second_i = arith.addi %next_i, %c1 : index
  %second = memref.load %data[%second_i] : memref<?xi64, 1>
  %sum = arith.addi %first, %second : i64
  return %sum : i64
}

// Remote stores remain host-only in the read-only device milestone.
// CHECK-LABEL: func.func @remote_store_rejected
// CHECK-NOT: cira.offload
func.func @remote_store_rejected(%idx: memref<?xi64, 1>,
                                 %data: memref<?xi64, 1>, %i: index,
                                 %value: i64) {
  %next = memref.load %idx[%i] : memref<?xi64, 1>
  %next_i = arith.index_cast %next : i64 to index
  memref.store %value, %data[%next_i] : memref<?xi64, 1>
  return
}

// The submit must follow the host-only definition of its live-in.
// CHECK-LABEL: func.func @livein_dominates_submit
// CHECK: %[[SEED:.*]] = call @host_index
// CHECK-NEXT: cira.offload
// CHECK: cira.barrier
func.func @livein_dominates_submit(%idx: memref<?xi64, 1>,
                                   %data: memref<?xi64, 1>) -> i64 {
  %seed = func.call @host_index() : () -> index
  %next = memref.load %idx[%seed] : memref<?xi64, 1>
  %next_i = arith.index_cast %next : i64 to index
  %v = memref.load %data[%next_i] : memref<?xi64, 1>
  return %v : i64
}
