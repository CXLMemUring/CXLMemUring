// RUN: cira %s --convert-cira-to-llvm-hetero | FileCheck %s

module {
  llvm.func @materialize_remote(%bits: i64) -> !llvm.ptr<1> {
    %remote = builtin.unrealized_conversion_cast %bits : i64 to !llvm.ptr<1>
    llvm.return %remote : !llvm.ptr<1>
  }

  func.func @submit_kernel(%remote: !llvm.ptr<1>, %limit: i32) -> i64 {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    scf.for %iv = %zero to %one step %one {
      cira.offload(%remote, %limit) attributes {
        cira.arg_layout = [
          {name = "arg0", offset = 0 : i64, type = "arena_offset"},
          {name = "arg1", offset = 8 : i64, type = "i32"}
        ],
        cira.args_size = 16 : i64,
        cira.kernel_id = 81985529216486895 : i64,
        cira.read_only
      } : (!llvm.ptr<1>, i32) -> () {
      ^bb0(%device_remote: !llvm.ptr<1>, %device_limit: i32):
        %value = llvm.load %device_remote : !llvm.ptr<1> -> i64
        %wide_limit = arith.extsi %device_limit : i32 to i64
        %selected = arith.cmpi slt, %value, %wide_limit : i64
      }
      cira.barrier
    }
    %host_value = llvm.load %remote : !llvm.ptr<1> -> i64
    return %host_value : i64
  }
}

// CHECK-LABEL: func.func @submit_kernel
// CHECK: llvm.alloca
// CHECK: scf.for
// CHECK-NOT: llvm.alloca
// CHECK: llvm.store
// CHECK: llvm.store
// CHECK: %[[EXTENT:.+]] = llvm.mlir.constant(8 : i64)
// CHECK-NEXT: llvm.call @cira_offload_arg_offset({{.*}}, %[[EXTENT]])
// CHECK: llvm.mlir.constant(81985529216486895 : i64)
// CHECK-NOT: llvm.call @cira_get_device_func_addr
// CHECK: %[[SUBMIT:.+]] = llvm.call @cira_offload_submit_kernel
// CHECK: llvm.load
// CHECK: llvm.call @cira_future_await
// CHECK-NEXT: llvm.call @cira_offload_require_success(%[[SUBMIT]], {{.*}})
// CHECK-NEXT: llvm.call @cira_future_free
// CHECK-NOT: llvm.call @cira_get_device_func_addr
// CHECK-NOT: func.func private @__vortex_kernel_
