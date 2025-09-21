cmake_minimum_required(VERSION 3.10)

# Example programs
add_executable(graph_processing graph_processing.cpp)
target_link_libraries(graph_processing cira_runtime)

add_executable(pagerank pagerank.cpp)
target_link_libraries(pagerank cira_runtime)

add_executable(bfs_traversal bfs_traversal.cpp)
target_link_libraries(bfs_traversal cira_runtime)