cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(hashTablesCuda LANGUAGES CXX CUDA)

OPTION(USE_CUDA "Build with GPU acceleration" ON)
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_language(CUDA)

INCLUDE_DIRECTORIES(
	src
)

FILE(GLOB_RECURSE SLABHASH_SOURCES
	src/SlabHash/*.cpp
	src/SlabHash/*.h
	src/*.h
	src/SlabHash/*.cu
	src/SlabHash/*.cuh)
	
#include(${CMAKE_BINARY_DIR}/build/conanbuildinfo.cmake)
#conan_basic_setup()

FILE(GLOB_RECURSE TEST_SOURCES
	tests/SlabHash/*.cu
	tests/SlabHash/*.cuh
	tests/SlabHash/*.h)

set(CMAKE_CUDA_FLAGS "-G -g")
add_executable(test_slab ${TEST_SOURCES} ${SLABHASH_SOURCES})
#target_link_libraries(test_slab ${CONAN_LIBS})  
set_target_properties(test_slab PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
