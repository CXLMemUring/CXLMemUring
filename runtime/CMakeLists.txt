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

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files for the runtime library
set(RUNTIME_SOURCES
    src/CiraRuntime.cpp
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