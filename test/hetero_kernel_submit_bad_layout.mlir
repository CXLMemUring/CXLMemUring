module {
  func.func @bad_alignment(%remote: !llvm.ptr<1>, %limit: i64) {
    cira.offload(%remote, %limit) attributes {
      cira.arg_layout = [
        {name = "arg0", offset = 0 : i64, type = "arena_offset"},
        {name = "arg1", offset = 9 : i64, type = "i64"}
      ],
      cira.args_size = 24 : i64,
      cira.kernel_id = 81985529216486895 : i64,
      cira.read_only
    } : (!llvm.ptr<1>, i64) -> () {
    ^bb0(%device_remote: !llvm.ptr<1>, %device_limit: i64):
      %value = llvm.load %device_remote : !llvm.ptr<1> -> i64
      %selected = arith.cmpi slt, %value, %device_limit : i64
    }
    cira.barrier
    return
  }
}
