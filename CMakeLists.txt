cmake_minimum_required(VERSION 3.12.4)
project(SuperPoint)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find torch in third party folder
set(CMAKE_PREFIX_PATH third_party/libtorch)
find_package(Torch  REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(third_party/CImg/ include)
add_executable(SuperPoint main.cpp imageGenerator.cpp)
target_link_libraries(SuperPoint ${TORCH_LIBRARIES})

enable_testing()
add_subdirectory(tests)
