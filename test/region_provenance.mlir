// RUN: cira %s --cira-region-formation='require-profitable=false' | FileCheck %s
// RUN: cira %s --cira-region-formation='require-profitable=false allow-legacy-heap-remote=true' | FileCheck %s --check-prefix=LEGACY

!s64i = !cir.int<s, 64>
!node = !cir.record<struct "node" {!s64i, !s64i} #cir.record.decl.ast>

// CHECK-LABEL: cir.func @plain_heap
// CHECK-NOT: cira.offload
// LEGACY-LABEL: cir.func @plain_heap
// LEGACY: cira.offload
cir.func @plain_heap(%p: !cir.ptr<!s64i>) -> !s64i {
  %next = cir.load %p : !cir.ptr<!s64i>, !s64i
  %addr = cir.ptr_stride %p, %next : (!cir.ptr<!s64i>, !s64i) -> !cir.ptr<!s64i>
  %v = cir.load %addr : !cir.ptr<!s64i>, !s64i
  cir.return %v : !s64i
}

// CHECK-LABEL: cir.func @stack_local
// CHECK-NOT: cira.offload
cir.func @stack_local(%i: !s64i) -> !s64i {
  %slot = cir.alloca !s64i, !cir.ptr<!s64i>, ["slot"] {alignment = 8 : i64}
  %next = cir.load %slot : !cir.ptr<!s64i>, !s64i
  %addr = cir.ptr_stride %slot, %next : (!cir.ptr<!s64i>, !s64i) -> !cir.ptr<!s64i>
  %v = cir.load %addr : !cir.ptr<!s64i>, !s64i
  cir.return %v : !s64i
}

// CIR pointer provenance propagates through cast, member, and stride ops.
// CHECK-LABEL: cir.func @cir_derived_remote
// CHECK: cira.offload
cir.func @cir_derived_remote(%p: !cir.ptr<!node>) -> !s64i {
  %remote = cir.cast(bitcast, %p : !cir.ptr<!node>), !cir.ptr<!node> {cira.remote}
  %next_addr = cir.get_member %remote[0] {name = "next"} : !cir.ptr<!node> -> !cir.ptr<!s64i>
  %next = cir.load %next_addr : !cir.ptr<!s64i>, !s64i
  %node_addr = cir.ptr_stride %remote, %next : (!cir.ptr<!node>, !s64i) -> !cir.ptr<!node>
  %value_addr = cir.get_member %node_addr[1] {name = "value"} : !cir.ptr<!node> -> !cir.ptr<!s64i>
  %v = cir.load %value_addr : !cir.ptr<!s64i>, !s64i
  cir.return %v : !s64i
}

// CHECK-LABEL: func.func @annotated_remote
// CHECK: cira.offload
func.func @annotated_remote(%p: memref<?xi64>, %i: index) -> i64 {
  %next = memref.load %p[%i] {cira.remote} : memref<?xi64>
  %next_i = arith.index_cast %next : i64 to index
  %v = memref.load %p[%next_i] {cira.remote} : memref<?xi64>
  return %v : i64
}

// CHECK-LABEL: func.func @remote_memref
// CHECK: cira.offload
func.func @remote_memref(%p: memref<?xi64, 1>, %i: index) -> i64 {
  %next = memref.load %p[%i] : memref<?xi64, 1>
  %next_i = arith.index_cast %next : i64 to index
  %v = memref.load %p[%next_i] : memref<?xi64, 1>
  return %v : i64
}

// CHECK-LABEL: func.func @memref_cast_remote
// CHECK: cira.offload
func.func @memref_cast_remote(%p: memref<16xi64>, %i: index) -> i64 {
  %remote = memref.cast %p {cira.remote} : memref<16xi64> to memref<?xi64>
  %next = memref.load %remote[%i] : memref<?xi64>
  %next_i = arith.index_cast %next : i64 to index
  %v = memref.load %remote[%next_i] : memref<?xi64>
  return %v : i64
}

// Explicit local provenance overrides a remote-looking address-space type.
// CHECK-LABEL: func.func @explicit_local
// CHECK-NOT: cira.offload
func.func @explicit_local(%p: memref<16xi64, 1>, %i: index) -> i64 {
  %local = memref.cast %p {cira.local} : memref<16xi64, 1> to memref<?xi64, 1>
  %next = memref.load %local[%i] : memref<?xi64, 1>
  %next_i = arith.index_cast %next : i64 to index
  %v = memref.load %local[%next_i] : memref<?xi64, 1>
  return %v : i64
}

// Conflicting remote/local inputs join to Unknown.
// CHECK-LABEL: func.func @conflicting_select
// CHECK-NOT: cira.offload
func.func @conflicting_select(%p: memref<16xi64>, %choose: i1, %i: index) -> i64 {
  %remote = memref.cast %p {cira.remote} : memref<16xi64> to memref<?xi64>
  %local = memref.cast %p {cira.local} : memref<16xi64> to memref<?xi64>
  %selected = arith.select %choose, %remote, %local : memref<?xi64>
  %next = memref.load %selected[%i] : memref<?xi64>
  %next_i = arith.index_cast %next : i64 to index
  %v = memref.load %selected[%next_i] : memref<?xi64>
  return %v : i64
}

// CHECK-LABEL: func.func @loop_iter_remote
// CHECK: cira.offload
func.func @loop_iter_remote(%p: memref<16xi64>, %n: index) -> memref<?xi64> {
  %remote = memref.cast %p {cira.remote} : memref<16xi64> to memref<?xi64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result = scf.for %i = %c0 to %n step %c1 iter_args(%cursor = %remote) -> (memref<?xi64>) {
    %next = memref.load %cursor[%i] : memref<?xi64>
    %next_i = arith.index_cast %next : i64 to index
    %v = memref.load %cursor[%next_i] : memref<?xi64>
    scf.yield %cursor : memref<?xi64>
  }
  return %result : memref<?xi64>
}

func.func private @memref_source() -> memref<?xi64>

// An unknown loop backedge poisons otherwise-remote initial provenance.
// CHECK-LABEL: func.func @loop_unknown_backedge
// CHECK-NOT: cira.offload
func.func @loop_unknown_backedge(%p: memref<16xi64>, %n: index) -> memref<?xi64> {
  %remote = memref.cast %p {cira.remote} : memref<16xi64> to memref<?xi64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result = scf.for %i = %c0 to %n step %c1 iter_args(%cursor = %remote) -> (memref<?xi64>) {
    %next = memref.load %cursor[%i] : memref<?xi64>
    %next_i = arith.index_cast %next : i64 to index
    %v = memref.load %cursor[%next_i] : memref<?xi64>
    %unknown = func.call @memref_source() : () -> memref<?xi64>
    scf.yield %unknown : memref<?xi64>
  }
  return %result : memref<?xi64>
}

// CHECK-LABEL: func.func @branch_argument_remote
// CHECK: cira.offload
func.func @branch_argument_remote(%p: memref<16xi64>, %i: index) -> i64 {
  %remote = memref.cast %p {cira.remote} : memref<16xi64> to memref<?xi64>
  cf.br ^next(%remote : memref<?xi64>)
^next(%cursor: memref<?xi64>):
  %next = memref.load %cursor[%i] : memref<?xi64>
  %next_i = arith.index_cast %next : i64 to index
  %v = memref.load %cursor[%next_i] : memref<?xi64>
  return %v : i64
}

// CHECK-LABEL: llvm.func @llvm_gep_remote
// CHECK: cira.offload
llvm.func @llvm_gep_remote(%p: !llvm.ptr<1>) -> i64 {
  %next = llvm.load %p : !llvm.ptr<1> -> i64
  %addr = llvm.getelementptr %p[%next] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i64
  %v = llvm.load %addr : !llvm.ptr<1> -> i64
  llvm.return %v : i64
}

cir.func private @pointer_source(!cir.ptr<!s64i>) -> !cir.ptr<!s64i>

// A remote argument does not make an unannotated call result remote.
// CHECK-LABEL: cir.func @unknown_call_result
// CHECK-NOT: cira.offload
cir.func @unknown_call_result(%p: !cir.ptr<!s64i>) -> !s64i {
  %remote = cir.cast(bitcast, %p : !cir.ptr<!s64i>), !cir.ptr<!s64i> {cira.remote}
  %q = cir.call @pointer_source(%remote) : (!cir.ptr<!s64i>) -> !cir.ptr<!s64i>
  %next = cir.load %q : !cir.ptr<!s64i>, !s64i
  %addr = cir.ptr_stride %q, %next : (!cir.ptr<!s64i>, !s64i) -> !cir.ptr<!s64i>
  %v = cir.load %addr : !cir.ptr<!s64i>, !s64i
  cir.return %v : !s64i
}

// CHECK-LABEL: cir.func @annotated_call_result
// CHECK: cira.offload
cir.func @annotated_call_result(%p: !cir.ptr<!s64i>) -> !s64i {
  %q = cir.call @pointer_source(%p) {cira.remote_result} : (!cir.ptr<!s64i>) -> !cir.ptr<!s64i>
  %next = cir.load %q : !cir.ptr<!s64i>, !s64i
  %addr = cir.ptr_stride %q, %next : (!cir.ptr<!s64i>, !s64i) -> !cir.ptr<!s64i>
  %v = cir.load %addr : !cir.ptr<!s64i>, !s64i
  cir.return %v : !s64i
}
