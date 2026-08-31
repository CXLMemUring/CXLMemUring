// RUN: cira %s --cira-outline-rv64-kernels='manifest=/tmp/cira-provenance-kernels.json module-id=provenance' | FileCheck %s

!s8i = !cir.int<s, 8>

module {
  // CHECK-LABEL: func.func @erased_remote_pointer
  // CHECK: cira.offload
  // CHECK-SAME: cira.arg_layout = [
  // CHECK-SAME: {name = "arg0", offset = 0 : i64, type = "arena_offset"}
  func.func @erased_remote_pointer(%raw: !llvm.ptr) {
    %remote = builtin.unrealized_conversion_cast %raw : !llvm.ptr to !cir.ptr<!s8i, addrspace(target<1>)>
    %generic = builtin.unrealized_conversion_cast %remote : !cir.ptr<!s8i, addrspace(target<1>)> to !llvm.ptr
    cira.offload(%generic) attributes {cira.region_id = 0 : i32} : (!llvm.ptr) -> () {
    ^bb0(%device_pointer: !llvm.ptr):
      %value = llvm.load %device_pointer : !llvm.ptr -> i64
    } loc("gdk/gdk_select.c":1:1)
    return
  }
}
