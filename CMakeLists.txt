cmake_minimum_required(VERSION 3.12.4)
project(SuperPoint)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(include)
add_executable(SuperPoint main.cpp)

enable_testing()
add_subdirectory(tests)
