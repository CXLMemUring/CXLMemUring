add_mlir_conversion_library(CXLMemUring
  CIRA.cpp
  SCFCIRAPattern.cpp
  CiraToLLVM.cpp
  SCFToCira.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include

  DEPENDS
  CIRAPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRDataLayoutInterfaces
  MLIRMemRefDialect
  MLIRLLVMDialect
  MLIRSCFDialect
  MLIRTransforms
  MLIRRemoteMem
)