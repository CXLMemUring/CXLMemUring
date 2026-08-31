// RUN: cira %s --cira-outline-rv64-kernels='manifest=/tmp/cira-kernels.json module-id=monetdb' | FileCheck %s

module {
  // CHECK-LABEL: func.func @join_probe
  // CHECK: cira.offload
  // CHECK-SAME: cira.arg_layout = [
  // CHECK-SAME: {name = "arg0", offset = 0 : i64, type = "arena_offset"}
  // CHECK-SAME: {name = "arg1", offset = 8 : i64, type = "i32"}
  // CHECK-SAME: cira.args_size = 16 : i64
  // CHECK-SAME: cira.kernel_id =
  func.func @join_probe(%links: memref<?xi64, 1>, %limit: i32) {
    cira.offload(%links, %limit) attributes {cira.query_tags = array<i32: 9, 11>, cira.region_id = 7 : i32} : (memref<?xi64, 1>, i32) -> () {
    ^bb0(%device_links: memref<?xi64, 1>, %device_limit: i32):
      %c0 = arith.constant 0 : index
      %value = memref.load %device_links[%c0] : memref<?xi64, 1>
      %wide_limit = arith.extsi %device_limit : i32 to i64
      %selected = arith.cmpi slt, %value, %wide_limit : i64
    } loc("gdk/gdk_join.c":4451:1)
    return
  }

  // CHECK: func.func private @__cira_rv64_[[ID:[0-9a-f]{16}]](%arg0: i64, %arg1: i64)
  // CHECK-SAME: cira.arg_layout = [
  // CHECK-SAME: cira.args_size = 16 : i64
  // CHECK-SAME: cira.kernel_id =
  // CHECK-SAME: cira.read_only
  // CHECK-SAME: cira.rv64_kernel
  // CHECK: call @cira_device_checked_load
  // CHECK: call @cira_device_checked_load
  // CHECK: call @cira_device_checked_load
  // CHECK-NOT: llvm.load
  // CHECK-NOT: memref.load
  // CHECK: return
}
