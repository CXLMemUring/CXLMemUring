add_mlir_conversion_library(CXLMemUring
  CIRA.cpp
  SCFCIRAPattern.cpp
  CiraToLLVM.cpp
  SCFToCira.cpp
  TOSAToCira.cpp
  CIRToCira.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include

  DEPENDS
  CIRAPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRArithDialect
  MLIRDataLayoutInterfaces
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRLLVMDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTosaDialect
  MLIRTransforms
  MLIRRemoteMem
)