add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(Lowering)
# add_subdirectory(Analysis)  # Temporarily disabled
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
  ${dialect_libs}
  ${conversion_libs}

  MLIRArithDialect
  MLIROptLib
  MLIRDialect
  MLIRIR
  MLIRRemoteMem
  # CXLMemUringAnalysis  # Temporarily disabled
)

add_llvm_executable(cira driver.cpp)
llvm_update_compile_flags(cira)
target_link_libraries(cira PRIVATE ${LIBS})
