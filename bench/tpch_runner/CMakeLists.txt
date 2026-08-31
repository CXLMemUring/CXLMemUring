cmake_minimum_required(VERSION 3.16)
project(monetdb_tpch_cira C)

if(NOT MONETDB_SOURCE_DIR OR NOT MONETDB_BUILD_DIR OR NOT CIRA_ROOT)
  message(FATAL_ERROR "MONETDB_SOURCE_DIR, MONETDB_BUILD_DIR, and CIRA_ROOT are required")
endif()

set(MONETDBE_LIBRARY "${MONETDB_BUILD_DIR}/tools/monetdbe/libmonetdbe.so")
set(BAT_LIBRARY "${MONETDB_BUILD_DIR}/gdk/libbat-11.52.0.so.29.0.0")
set(CIRA_RUNTIME_LIBRARY "${CIRA_ROOT}/build/runtime/libcira_runtime.so.1.0.0")
foreach(required IN ITEMS MONETDBE_LIBRARY BAT_LIBRARY CIRA_RUNTIME_LIBRARY)
  if(NOT EXISTS "${${required}}")
    message(FATAL_ERROR "missing required library: ${${required}}")
  endif()
endforeach()

add_executable(monetdb-tpch-cira monetdb_tpch_cira.c result_writer.c)
target_compile_features(monetdb-tpch-cira PRIVATE c_std_11)
target_compile_options(monetdb-tpch-cira PRIVATE -Wall -Wextra -Werror)
target_include_directories(monetdb-tpch-cira PRIVATE
  "${MONETDB_BUILD_DIR}"
  "${MONETDB_SOURCE_DIR}/tools/monetdbe"
  "${CIRA_ROOT}/runtime/include")
target_link_libraries(monetdb-tpch-cira PRIVATE
  "${MONETDBE_LIBRARY}" "${BAT_LIBRARY}" "${CIRA_RUNTIME_LIBRARY}")
set_target_properties(monetdb-tpch-cira PROPERTIES BUILD_RPATH
  "${MONETDB_BUILD_DIR}/tools/monetdbe;${MONETDB_BUILD_DIR}/sql/backends/monet5;${MONETDB_BUILD_DIR}/monetdb5/tools;${MONETDB_BUILD_DIR}/gdk;${MONETDB_BUILD_DIR}/clients/mapilib;${MONETDB_BUILD_DIR}/common/stream;${CIRA_ROOT}/build/runtime")
