module {
  func.func private @host_lookup(i64) -> i64

  func.func @bad_kernel(%links: memref<?xi64, 1>) {
    cira.offload(%links) attributes {cira.region_id = 0 : i32} : (memref<?xi64, 1>) -> () {
    ^bb0(%device_links: memref<?xi64, 1>):
      %c0 = arith.constant 0 : index
      %value = memref.load %device_links[%c0] : memref<?xi64, 1>
      %ignored = func.call @host_lookup(%value) : (i64) -> i64
    } loc("gdk/gdk_join.c":99:3)
    return
  }
}
