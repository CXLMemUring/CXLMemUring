cmake_minimum_required(VERSION 3.10)
project(cgeist)

# Create a simple executable without any dependencies
add_executable(cgeist
  driver.cc
)

install(TARGETS cgeist
  RUNTIME DESTINATION bin
  COMPONENT cgeist)
