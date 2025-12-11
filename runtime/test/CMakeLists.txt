cmake_minimum_required(VERSION 3.10)

# Test executables
add_executable(test_offload_engine test_offload_engine.cpp)
target_link_libraries(test_offload_engine cira_runtime)

add_executable(test_memory_manager test_memory_manager.cpp)
target_link_libraries(test_memory_manager cira_runtime)

add_executable(test_graph_runtime test_graph_runtime.cpp)
target_link_libraries(test_graph_runtime cira_runtime)

add_executable(test_twopass_runtime test_twopass_runtime.cpp)
target_link_libraries(test_twopass_runtime cira_runtime)
target_include_directories(test_twopass_runtime PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Add tests
add_test(NAME OffloadEngineTest COMMAND test_offload_engine)
add_test(NAME MemoryManagerTest COMMAND test_memory_manager)
add_test(NAME GraphRuntimeTest COMMAND test_graph_runtime)
add_test(NAME TwoPassRuntimeTest COMMAND test_twopass_runtime)