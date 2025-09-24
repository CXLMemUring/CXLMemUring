cmake_minimum_required(VERSION 3.10)

find_program(BASH_EXECUTABLE bash)
if(NOT BASH_EXECUTABLE)
  message(FATAL_ERROR "bash not found; required to run tests")
endif()

add_test(
  NAME CXLMemUring_RunAll
  COMMAND ${BASH_EXECUTABLE} ${CMAKE_SOURCE_DIR}/test/run_tests.sh
)
set_tests_properties(CXLMemUring_RunAll PROPERTIES TIMEOUT 240)

# Optional: add FileCheck-specific tests if available
find_program(FILECHECK_EXECUTABLE FileCheck)
if(FILECHECK_EXECUTABLE)
  add_test(
    NAME CXLMemUring_Check_RemoteExtract
    COMMAND ${BASH_EXECUTABLE} -lc "${CMAKE_BINARY_DIR}/bin/cira ${CMAKE_SOURCE_DIR}/test/mlir/check_remote_extract.mlir --rmem-search-remote | ${FILECHECK_EXECUTABLE} ${CMAKE_SOURCE_DIR}/test/mlir/check_remote_extract.mlir"
  )
  add_test(
    NAME CXLMemUring_Check_CiraAttrs
    COMMAND ${BASH_EXECUTABLE} -lc "${CMAKE_BINARY_DIR}/bin/cira ${CMAKE_SOURCE_DIR}/test/mlir/check_cira_attrs.mlir --convert-target-to-remote | ${FILECHECK_EXECUTABLE} ${CMAKE_SOURCE_DIR}/test/mlir/check_cira_attrs.mlir"
  )
endif()

# Per-MLIR tests for finer-grained reporting
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/test_ctest_out)
set(_CIRA ${CMAKE_BINARY_DIR}/bin/cira)
set(_OUT ${CMAKE_BINARY_DIR}/test_ctest_out)
function(add_mlir_pipeline_test name in)
  add_test(
    NAME ${name}
    COMMAND ${BASH_EXECUTABLE} -lc "\
      ${_CIRA} ${in} --rmem-search-remote -o ${_OUT}/${name}_remote.mlir && \
      ${_CIRA} ${_OUT}/${name}_remote.mlir --convert-cira-to-llvm-hetero --convert-func-to-llvm --reconcile-unrealized-casts -o ${_OUT}/${name}_hetero.ll"
  )
endfunction()

add_mlir_pipeline_test(types_i32            ${CMAKE_SOURCE_DIR}/test/mlir/types_i32.mlir)
add_mlir_pipeline_test(types_f64            ${CMAKE_SOURCE_DIR}/test/mlir/types_f64.mlir)
add_mlir_pipeline_test(indirect_store       ${CMAKE_SOURCE_DIR}/test/mlir/indirect_store.mlir)
add_mlir_pipeline_test(mixed                ${CMAKE_SOURCE_DIR}/test/mlir/mixed.mlir)
add_mlir_pipeline_test(bounds_var           ${CMAKE_SOURCE_DIR}/test/mlir/bounds_var.mlir)
add_mlir_pipeline_test(nested_shapes        ${CMAKE_SOURCE_DIR}/test/mlir/nested_shapes.mlir)
add_mlir_pipeline_test(aliasing             ${CMAKE_SOURCE_DIR}/test/mlir/aliasing.mlir)
add_mlir_pipeline_test(aliasing_two         ${CMAKE_SOURCE_DIR}/test/mlir/aliasing_two.mlir)
add_mlir_pipeline_test(mixed_sizes          ${CMAKE_SOURCE_DIR}/test/mlir/mixed_sizes.mlir)
add_mlir_pipeline_test(mixed_precision      ${CMAKE_SOURCE_DIR}/test/mlir/mixed_precision.mlir)
add_mlir_pipeline_test(deep_nest            ${CMAKE_SOURCE_DIR}/test/mlir/deep_nest.mlir)
add_mlir_pipeline_test(deep_nest_branch     ${CMAKE_SOURCE_DIR}/test/mlir/deep_nest_branch.mlir)
add_mlir_pipeline_test(reduction            ${CMAKE_SOURCE_DIR}/test/mlir/reduction.mlir)

# Negative test: expect verifier error on malformed cast
add_test(
  NAME negative_cast
  COMMAND ${_CIRA} ${CMAKE_SOURCE_DIR}/test/mlir/negative_cast.mlir --rmem-search-remote -o ${_OUT}/negative_remote.mlir
)
set_tests_properties(negative_cast PROPERTIES WILL_FAIL TRUE PASS_REGULAR_EXPRESSION "")
