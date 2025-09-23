# Add compiler flags to work around MLIR/LLVM compatibility issues
add_compile_options(-fpermissive -Wno-error)

add_mlir_dialect_library(MLIRRemoteMem
        RemoteMemDialect.cpp
        RemoteMemRef.cpp
        FunctionUtils.cpp
        WorkloadAnalysis.cpp
        ReadFromProfile.cpp
        RemoteMemTypeLowerer.cpp
        OffloadOp.cpp
        CiraOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include

        DEPENDS
        MLIRRemoteMemIncGen
        MLIRCiraOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRMemRefDialect
        MLIRInferTypeOpInterface
)

add_subdirectory(Transforms)