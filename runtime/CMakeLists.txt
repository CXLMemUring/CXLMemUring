cmake_minimum_required(VERSION 3.10)
project(CiraRuntime VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Find required packages
find_package(Threads REQUIRED)

# Check for NUMA support
find_library(NUMA_LIBRARY numa)
if(NOT NUMA_LIBRARY)
    message(WARNING "libnuma not found, building without NUMA support")
    set(NUMA_SUPPORT OFF)
else()
    set(NUMA_SUPPORT ON)
    message(STATUS "Found libnuma: ${NUMA_LIBRARY}")
endif()

# Vortex SDK (optional, for hardware support)
set(VORTEX_HOME "$ENV{HOME}/vortex" CACHE PATH "Vortex SDK root directory")
if(EXISTS "${VORTEX_HOME}/runtime/include")
    set(VORTEX_FOUND ON)
    message(STATUS "Found Vortex SDK at: ${VORTEX_HOME}")
else()
    set(VORTEX_FOUND OFF)
    message(STATUS "Vortex SDK not found at ${VORTEX_HOME}, building with simulation stubs")
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
if(VORTEX_FOUND)
    include_directories(${VORTEX_HOME}/runtime/include)
    include_directories(${VORTEX_HOME}/build/hw)  # For VX_config.h
endif()

# Source files for the runtime library
set(RUNTIME_SOURCES
    src/CiraRuntime.cpp
    vortex_verilator_sim.cpp
    vortex_device.cpp
    vortex_offload_backend.cpp
)

# Create shared library
add_library(cira_runtime SHARED ${RUNTIME_SOURCES})

# Link libraries
target_link_libraries(cira_runtime
    PUBLIC
        Threads::Threads
)

if(NUMA_SUPPORT)
    target_link_libraries(cira_runtime PUBLIC ${NUMA_LIBRARY})
    target_compile_definitions(cira_runtime PUBLIC NUMA_SUPPORT)
endif()

if(VORTEX_FOUND)
    target_compile_definitions(cira_runtime PUBLIC VORTEX_FOUND)
    target_link_libraries(cira_runtime PUBLIC dl)
    # Link to Vortex runtime library
    target_link_libraries(cira_runtime PUBLIC ${VORTEX_HOME}/build/runtime/libvortex.so)
    target_link_libraries(cira_runtime PUBLIC ${VORTEX_HOME}/build/runtime/libsimx.so)
endif()

# Create static library
add_library(cira_runtime_static STATIC ${RUNTIME_SOURCES})
set_target_properties(cira_runtime_static PROPERTIES OUTPUT_NAME cira_runtime)

target_link_libraries(cira_runtime_static
    PUBLIC
        Threads::Threads
)

if(NUMA_SUPPORT)
    target_link_libraries(cira_runtime_static PUBLIC ${NUMA_LIBRARY})
    target_compile_definitions(cira_runtime_static PUBLIC NUMA_SUPPORT)
endif()

if(VORTEX_FOUND)
    target_compile_definitions(cira_runtime_static PUBLIC VORTEX_FOUND)
    target_link_libraries(cira_runtime_static PUBLIC dl)
    target_link_libraries(cira_runtime_static PUBLIC ${VORTEX_HOME}/build/runtime/libvortex.so)
    target_link_libraries(cira_runtime_static PUBLIC ${VORTEX_HOME}/build/runtime/libsimx.so)
endif()

# Set library properties
set_target_properties(cira_runtime PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER include/CiraRuntime.h
)

# Installation rules
install(TARGETS cira_runtime cira_runtime_static
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include/cira
)

# Install all header files
install(DIRECTORY include/
    DESTINATION include/cira
    FILES_MATCHING PATTERN "*.h"
)

# Print configuration summary
message(STATUS "")
message(STATUS "CiraRuntime Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  NUMA Support: ${NUMA_SUPPORT}")
message(STATUS "  Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")