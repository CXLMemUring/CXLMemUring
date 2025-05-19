add_mlir_dialect_library(MLIRPolygeist
Dialect.cpp
Ops.cpp
PolygeistOpCanonicalizers.cpp

ADDITIONAL_HEADER_DIRS
${PROJECT_SOURCE_DIR}/include/polygeist

DEPENDS

LINK_LIBS PUBLIC
MLIRIR
MLIRMemRefDialect
MLIRLLVMDialect
MLIROpenMPDialect
MLIRAffineDialect
MLIRSupport
MLIRSCFTransforms
)
add_subdirectory(Passes)
