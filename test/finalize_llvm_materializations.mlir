// RUN: cira %s --cira-finalize-llvm-materializations | FileCheck %s

module {
  llvm.func @consume(!llvm.ptr<1>)

  llvm.func @finalize(%pointer: !llvm.ptr, %index: i64) {
    %remote = builtin.unrealized_conversion_cast %pointer : !llvm.ptr to !llvm.ptr<1>
    llvm.call @consume(%remote) : (!llvm.ptr<1>) -> ()
    %unused = builtin.unrealized_conversion_cast %index : i64 to index
    llvm.return
  }
}

// CHECK-LABEL: llvm.func @finalize
// CHECK: llvm.addrspacecast %{{.*}} : !llvm.ptr to !llvm.ptr<1>
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
