cmake_minimum_required(VERSION 3.18)
project(MemTier VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MEMTIER_ENABLE_CUDA "Enable CUDA backend" OFF)
option(MEMTIER_ENABLE_GDS "Enable GDS backend" OFF)
option(MEMTIER_STRICT_GDS "Fail configure if GDS requested but not found" OFF)
option(MEMTIER_BUILD_TESTS "Build tests" ON)
option(MEMTIER_BUILD_EXAMPLES "Build examples" ON)
option(MEMTIER_BUILD_BENCHMARKS "Build benchmarks" ON)

include(CheckIncludeFileCXX)

if(MEMTIER_ENABLE_CUDA)
  find_package(CUDAToolkit QUIET)
  if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "MEMTIER_ENABLE_CUDA=ON but CUDAToolkit was not found")
  endif()
endif()

set(MEMTIER_GDS_AVAILABLE OFF)
if(MEMTIER_ENABLE_GDS)
  find_path(CUFILE_INCLUDE_DIR cufile.h)
  find_library(CUFILE_LIBRARY cufile)
  if(CUFILE_INCLUDE_DIR AND CUFILE_LIBRARY)
    set(MEMTIER_GDS_AVAILABLE ON)
    message(STATUS "GDS enabled with cufile: ${CUFILE_LIBRARY}")
  else()
    if(MEMTIER_STRICT_GDS)
      message(FATAL_ERROR "MEMTIER_ENABLE_GDS=ON but cufile.h/libcufile not found")
    else()
      message(WARNING "GDS requested but cufile.h/libcufile not found; disabling GDS and using fallback")
      set(MEMTIER_ENABLE_GDS OFF)
    endif()
  endif()
endif()

add_library(memtier STATIC
  src/api.cpp
  src/posix_backend.cpp
  src/logger.cpp
  src/cache_directory.cpp
  src/context.cpp
  src/path_selector.cpp
  src/gds_backend_stub.cpp
)

target_include_directories(memtier PUBLIC include)

target_link_libraries(memtier PUBLIC pthread)

if(MEMTIER_ENABLE_CUDA)
  target_compile_definitions(memtier PUBLIC MEMTIER_HAS_CUDA=1)
  target_link_libraries(memtier PUBLIC CUDA::cudart)
else()
  target_compile_definitions(memtier PUBLIC MEMTIER_HAS_CUDA=0)
endif()

if(MEMTIER_GDS_AVAILABLE)
  target_compile_definitions(memtier PUBLIC MEMTIER_HAS_GDS=1)
  target_include_directories(memtier PRIVATE ${CUFILE_INCLUDE_DIR})
  target_link_libraries(memtier PRIVATE ${CUFILE_LIBRARY})
else()
  target_compile_definitions(memtier PUBLIC MEMTIER_HAS_GDS=0)
endif()

if(MEMTIER_BUILD_EXAMPLES)
  add_executable(memtier_example examples/basic_read.cpp)
  target_link_libraries(memtier_example PRIVATE memtier)
  add_executable(cache_reuse examples/cache_reuse.cpp)
  target_link_libraries(cache_reuse PRIVATE memtier)
  add_executable(path_policy_demo examples/path_policy_demo.cpp)
  target_link_libraries(path_policy_demo PRIVATE memtier)
endif()

if(MEMTIER_BUILD_TESTS)
  enable_testing()
  add_executable(test_basic tests/test_basic.cpp)
  target_link_libraries(test_basic PRIVATE memtier)
  add_test(NAME test_basic COMMAND test_basic)
  add_executable(test_cache tests/test_cache.cpp)
  target_link_libraries(test_cache PRIVATE memtier)
  add_test(NAME test_cache COMMAND test_cache)
  add_executable(test_path_selector tests/test_path_selector.cpp)
  target_link_libraries(test_path_selector PRIVATE memtier)
  add_test(NAME test_path_selector COMMAND test_path_selector)
endif()

if(MEMTIER_BUILD_BENCHMARKS)
  add_executable(bench_stub benchmarks/bench_stub.cpp)
  target_link_libraries(bench_stub PRIVATE memtier)
endif()
