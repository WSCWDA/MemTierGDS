cmake_minimum_required(VERSION 3.18)
project(MemTier LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)

option(MEMTIER_BUILD_TESTS "Build tests" ON)
option(MEMTIER_BUILD_EXAMPLES "Build examples" ON)

add_library(memtier STATIC
  src/api.cpp src/context.cpp src/log.cpp src/stats.cpp src/cache_directory.cpp
  src/posix_backend.cpp src/gds_backend_stub.cpp src/cuda_backend_stub.cpp
)
target_include_directories(memtier PUBLIC include PRIVATE src)

if(MEMTIER_BUILD_EXAMPLES)
  add_executable(runtime_info examples/runtime_info.cpp)
  target_link_libraries(runtime_info PRIVATE memtier)
endif()

if(MEMTIER_BUILD_TESTS)
  enable_testing()
  add_executable(test_runtime tests/test_runtime.cpp)
  target_link_libraries(test_runtime PRIVATE memtier)
  add_test(NAME test_runtime COMMAND test_runtime)
endif()
