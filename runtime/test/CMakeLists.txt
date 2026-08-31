cmake_minimum_required(VERSION 3.10)

# One helper so every test gets the same include paths and link line.
function(cira_add_runtime_test name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE cira_runtime)
    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/..
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

# Cost model: pure decision logic, no LLVM and no device required.
cira_add_runtime_test(test_cira_jit test_cira_jit.cpp)

# Dispatch path: cache-resident wait + CXL control window + doorbell protocol.
# Runs against an emulated (anonymous-memory) window, so no FPGA is needed.
cira_add_runtime_test(test_cira_offload_path test_cira_offload_path.cpp)
cira_add_runtime_test(test_cira_cache_completion_path test_cira_cache_completion_path.cpp)
cira_add_runtime_test(test_cira_kernel_call_abi test_cira_kernel_call_abi.cpp)
cira_add_runtime_test(test_cira_shared_arena test_cira_shared_arena.cpp)

find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(
    NAME test_rv64_dispatch_table
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test_rv64_dispatch_table.py
            ${CMAKE_CURRENT_SOURCE_DIR}/../device/gen_dispatch_table.py
)

find_program(RISCV64_GCC riscv64-linux-gnu-gcc REQUIRED)
find_program(RISCV64_READELF riscv64-linux-gnu-readelf REQUIRED)
find_program(RISCV64_NM riscv64-linux-gnu-nm REQUIRED)
find_program(RISCV64_OBJDUMP riscv64-linux-gnu-objdump REQUIRED)
add_test(
    NAME test_rv64_device_elf
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test_rv64_device_elf.py
            ${CMAKE_CURRENT_SOURCE_DIR}/../device
            ${CMAKE_CURRENT_BINARY_DIR}/rv64-device
            ${RISCV64_GCC} ${RISCV64_READELF} ${RISCV64_NM} ${RISCV64_OBJDUMP}
)

# Offload engine / two-pass profiling.
cira_add_runtime_test(test_offload_engine test_offload_engine.cpp)
cira_add_runtime_test(test_twopass_runtime test_twopass_runtime.cpp)

# ORC specializer — only built when the runtime was configured with LLVM.
if(CIRA_JIT_HAS_LLVM)
    cira_add_runtime_test(test_cira_jit_engine test_cira_jit_engine.cpp)
endif()

# Cost-model microbenchmark — built, but not part of the test suite.
add_executable(bench_cira_jit bench_cira_jit.cpp)
target_link_libraries(bench_cira_jit PRIVATE cira_runtime)
target_include_directories(bench_cira_jit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
