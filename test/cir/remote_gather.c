// End-to-end source input for the ClangIR -> core MLIR -> CIRA region-forming
// path. Address space 1 denotes CXL-attached memory.
typedef unsigned long __attribute__((address_space(1))) *remote_index_ptr;
typedef float __attribute__((address_space(1))) *remote_data_ptr;

float remote_gather(remote_index_ptr index, remote_data_ptr data,
                    unsigned long i) {
  return data[index[i]];
}
