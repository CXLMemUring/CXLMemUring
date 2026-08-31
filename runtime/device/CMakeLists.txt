cmake_minimum_required(VERSION 3.16)
project(CiraRv64Device LANGUAGES C ASM)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(CIRA_KERNEL_MANIFEST "" CACHE FILEPATH "CIRA ABI-v2 kernel manifest")
set(CIRA_KERNEL_SOURCES "" CACHE STRING "Semicolon-separated RV64 kernel source/object files")
set(CIRA_DEVICE_CONTROL_BASE "0x40000000" CACHE STRING "RV64 control-window base")
set(CIRA_DEVICE_ARENA_BASE "0x80000000" CACHE STRING "RV64 shared-arena aperture base")
set(CIRA_DEVICE_ARENA_SIZE "0x10000000" CACHE STRING "RV64 shared-arena aperture size")

if(NOT CIRA_KERNEL_MANIFEST OR NOT EXISTS "${CIRA_KERNEL_MANIFEST}")
    message(FATAL_ERROR "CIRA_KERNEL_MANIFEST must name an existing ABI-v2 manifest")
endif()
if(NOT CIRA_KERNEL_SOURCES)
    message(FATAL_ERROR "CIRA_KERNEL_SOURCES must provide the real RV64 kernel implementations")
endif()

set(DISPATCH_HEADER "${CMAKE_CURRENT_BINARY_DIR}/cira_rv64_dispatch.generated.h")
add_custom_command(
    OUTPUT "${DISPATCH_HEADER}"
    COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/gen_dispatch_table.py"
            "${CIRA_KERNEL_MANIFEST}" "${DISPATCH_HEADER}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/gen_dispatch_table.py" "${CIRA_KERNEL_MANIFEST}"
    VERBATIM
)

add_executable(cira-rv64-device
    cira_rv64_start.S
    cira_rv64_firmware.c
    ${CIRA_KERNEL_SOURCES}
    "${DISPATCH_HEADER}"
)
set_target_properties(cira-rv64-device PROPERTIES SUFFIX ".elf")
target_include_directories(cira-rv64-device PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    "${CMAKE_CURRENT_BINARY_DIR}"
)
target_compile_definitions(cira-rv64-device PRIVATE
    CIRA_DEVICE_CONTROL_BASE=${CIRA_DEVICE_CONTROL_BASE}
    CIRA_DEVICE_ARENA_BASE=${CIRA_DEVICE_ARENA_BASE}
    CIRA_DEVICE_ARENA_SIZE=${CIRA_DEVICE_ARENA_SIZE}
)
target_compile_options(cira-rv64-device PRIVATE
    -march=rv64imac
    -mabi=lp64
    -mcmodel=medany
    -O2
    -ffreestanding
    -fno-builtin
    -fno-stack-protector
    -fno-pic
    -fno-pie
    -fdata-sections
    -ffunction-sections
)
target_link_options(cira-rv64-device PRIVATE
    -nostdlib
    -static
    -no-pie
    -march=rv64imac
    -mabi=lp64
    -mcmodel=medany
    "LINKER:-T,${CMAKE_CURRENT_SOURCE_DIR}/cira_rv64_link.ld"
    LINKER:--gc-sections
    LINKER:--no-undefined
    LINKER:--build-id=none
)
set_property(TARGET cira-rv64-device APPEND PROPERTY
    LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cira_rv64_link.ld"
)
