add_mlir_library(CXLMemUringAnalysis
  TOSAOffloadAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Analysis

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRFuncDialect
  MLIRTosaDialect
  MLIRSupport
  MLIRIR
)