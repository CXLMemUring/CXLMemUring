add_mlir_dialect_library(MLIRRemoteMem
        RemoteMemDialect.cpp
        RemoteMemRef.cpp
        FunctionUtils.cpp
        WorkloadAnalysis.cpp
        ReadFromProfile.cpp
        RemoteMemTypeLowerer.cpp
        OffloadOp.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include

        DEPENDS
        MLIRRemoteMemIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRMemRefDialect
        MLIRInferTypeOpInterface
)

add_subdirectory(Transforms)