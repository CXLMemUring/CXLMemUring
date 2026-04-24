# Add compiler flags to work around MLIR/LLVM compatibility issues
add_compile_options(-fpermissive -Wno-error -w)

add_mlir_dialect_library(MLIRRemoteMem
        RemoteMemDialect.cpp
        RemoteMemRef.cpp
        FunctionUtils.cpp
        WorkloadAnalysis.cpp
        ReadFromProfile.cpp
        OffloadProfile.cpp
        OffloadOverheadAnalysis.cpp
        RemoteMemTypeLowerer.cpp
        OffloadOp.cpp
        CiraOps.cpp
        TwoPassTimingAnalysis.cpp
        GpuOffloadDecision.cpp
        GpuKernelGeneration.cpp
        GpuMemoryOptimization.cpp
        GpuRuntimeLowering.cpp
        VortexKernelGen.cpp
        Type2GpuCodeGen.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include

        DEPENDS
        MLIRRemoteMemIncGen
        MLIRCiraOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRMemRefDialect
        MLIRInferTypeOpInterface
        MLIRAnalysis
        MLIRSCFDialect
        MLIRArithDialect
        MLIRLinalgDialect
        MLIRFuncDialect
)

add_subdirectory(Transforms)