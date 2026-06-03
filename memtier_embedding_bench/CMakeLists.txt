cmake_minimum_required(VERSION 3.18)
project(memtier_embedding_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(ENABLE_GDS "Build with experimental GDS/cuFile path selection hooks" OFF)

add_library(memtier_embedding SHARED
  src/memtier_api.cpp
  src/residency_table.cpp
  src/posix_loader.cpp
  src/gds_loader_stub.cpp
  src/dram_cache.cpp
  src/metrics.cpp
  cuda/embedding_lookup.cu
  cuda/embedding_bag_kernel.cu
)

target_include_directories(memtier_embedding PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(memtier_embedding PUBLIC CUDA::cudart)

if(ENABLE_GDS)
  target_compile_definitions(memtier_embedding PRIVATE ENABLE_GDS=1)
endif()

set_target_properties(memtier_embedding PROPERTIES
  OUTPUT_NAME memtier_embedding
  CUDA_SEPARABLE_COMPILATION ON
)
