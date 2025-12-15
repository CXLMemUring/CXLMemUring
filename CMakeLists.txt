cmake_minimum_required(VERSION 3.10)
project(CIRA)

set(CMAKE_CXX_STANDARD 20)

# Find packages - LLVM first, then MLIR, then Clang
find_package(LLVM REQUIRED CONFIG)

# Try to find MLIR - it may fail if CIR targets are missing from ClangIR install
# In that case, we manually define the CIR targets
set(MLIR_FOUND_WITH_CIR ON)
find_package(MLIR CONFIG)
if(NOT MLIR_FOUND)
  message(WARNING "MLIR package not found or missing CIR targets, attempting manual setup")
  set(MLIR_FOUND_WITH_CIR OFF)
  # Force MLIR_DIR and try again without imported targets check
  set(MLIR_DIR "${LLVM_DIR}/../mlir" CACHE PATH "MLIR cmake directory" FORCE)
  include(${MLIR_DIR}/MLIRConfig.cmake OPTIONAL)
endif()

find_package(Clang REQUIRED CONFIG)

# Manually create CIR library targets if they don't exist (common with ClangIR installs)
if(NOT TARGET MLIRCIR)
  message(STATUS "Creating MLIRCIR imported target manually")
  add_library(MLIRCIR STATIC IMPORTED)
  set_target_properties(MLIRCIR PROPERTIES
    IMPORTED_LOCATION "${LLVM_LIBRARY_DIR}/libMLIRCIR.a"
    INTERFACE_INCLUDE_DIRECTORIES "${LLVM_INCLUDE_DIRS};${CLANG_INCLUDE_DIRS}"
  )
endif()

if(NOT TARGET MLIRCIRTransforms)
  message(STATUS "Creating MLIRCIRTransforms imported target manually")
  add_library(MLIRCIRTransforms STATIC IMPORTED)
  set_target_properties(MLIRCIRTransforms PROPERTIES
    IMPORTED_LOCATION "${LLVM_LIBRARY_DIR}/libMLIRCIRTransforms.a"
    INTERFACE_INCLUDE_DIRECTORIES "${LLVM_INCLUDE_DIRS}"
  )
endif()

if(NOT TARGET clangCIRLoweringDirectToLLVM)
  message(STATUS "Creating clangCIRLoweringDirectToLLVM imported target manually")
  add_library(clangCIRLoweringDirectToLLVM STATIC IMPORTED)
  set_target_properties(clangCIRLoweringDirectToLLVM PROPERTIES
    IMPORTED_LOCATION "${LLVM_LIBRARY_DIR}/libclangCIRLoweringDirectToLLVM.a"
    INTERFACE_INCLUDE_DIRECTORIES "${LLVM_INCLUDE_DIRS};${CLANG_INCLUDE_DIRS}"
  )
endif()

if(NOT TARGET clangCIRLoweringThroughMLIR)
  message(STATUS "Creating clangCIRLoweringThroughMLIR imported target manually")
  add_library(clangCIRLoweringThroughMLIR STATIC IMPORTED)
  set_target_properties(clangCIRLoweringThroughMLIR PROPERTIES
    IMPORTED_LOCATION "${LLVM_LIBRARY_DIR}/libclangCIRLoweringThroughMLIR.a"
    INTERFACE_INCLUDE_DIRECTORIES "${LLVM_INCLUDE_DIRS};${CLANG_INCLUDE_DIRS}"
  )
endif()

message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using ClangConfig.cmake in: ${Clang_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/lib/Polygeist/include/)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(src)
# Polygeist disabled - using ClangIR integration instead
# add_subdirectory(lib/polygeist)

# Add runtime library
add_subdirectory(runtime)

# Testing
include(CTest)
if(BUILD_TESTING)
  add_subdirectory(test)
endif()

# Optional: CIR dialect registration (from clangir checkout)
message(STATUS "Enabling CIR dialect registration from ${CLANGIR_ROOT}")
include_directories(${CLANGIR_ROOT}/llvm/include)
include_directories(${CLANGIR_ROOT}/clang/include)
# Common build layouts
link_directories(${CLANGIR_ROOT}/llvm/build/lib)
