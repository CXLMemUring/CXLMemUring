// RUN: cira %s --convert-cira-to-llvm-x86 | FileCheck %s

module {
  func.func @llc_tile_runtime_hooks() {
    %c128 = arith.constant 128 : index
    %tile = cira.alloc_cxl %c128 : !cira.handle<i8>
    cira.install_cacheline %tile, %c128 level = 3 : !cira.handle<i8>, index
    cira.release %tile : !cira.handle<i8>
    return
  }
}

// CHECK-LABEL: func.func @llc_tile_runtime_hooks
// CHECK: llvm.call @cira_llc_tile_alloc
// CHECK: llvm.call @cira_install_cacheline_x86
// CHECK: llvm.call @cira_llc_tile_free
// CHECK-NOT: llvm.call @malloc
// CHECK-NOT: llvm.call @free
