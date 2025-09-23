module @"/home/victoryang00/CXLMemUring/build/simple_add.c" attributes {cir.lang = #cir.lang<c>, cir.sob = #cir.signed_overflow_behavior<undefined>, cir.triple = "x86_64-unknown-linux-gnu", cir.type_size_info = #cir.type_size_info<char = 8, int = 32, size_t = 64>, dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>} {
  func.func @add_floats(%arg0: f32, %arg1: f32) -> f32 {
    %alloca = memref.alloca() {alignment = 4 : i64} : memref<f32>
    %alloca_0 = memref.alloca() {alignment = 4 : i64} : memref<f32>
    %alloca_1 = memref.alloca() {alignment = 4 : i64} : memref<f32>
    memref.store %arg0, %alloca[] : memref<f32>
    memref.store %arg1, %alloca_0[] : memref<f32>
    %0 = memref.load %alloca[] : memref<f32>
    %1 = memref.load %alloca_0[] : memref<f32>
    %2 = arith.addf %0, %1 : f32
    memref.store %2, %alloca_1[] : memref<f32>
    %3 = memref.load %alloca_1[] : memref<f32>
    return %3 : f32
  }
  func.func @add_ints(%arg0: i32, %arg1: i32) -> i32 {
    %alloca = memref.alloca() {alignment = 4 : i64} : memref<i32>
    %alloca_0 = memref.alloca() {alignment = 4 : i64} : memref<i32>
    %alloca_1 = memref.alloca() {alignment = 4 : i64} : memref<i32>
    memref.store %arg0, %alloca[] : memref<i32>
    memref.store %arg1, %alloca_0[] : memref<i32>
    %0 = memref.load %alloca[] : memref<i32>
    %1 = memref.load %alloca_0[] : memref<i32>
    %2 = arith.addi %0, %1 : i32
    memref.store %2, %alloca_1[] : memref<i32>
    %3 = memref.load %alloca_1[] : memref<i32>
    return %3 : i32
  }
}

