cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(hashTablesCuda LANGUAGES CXX CUDA)

OPTION(USE_CUDA "Build with GPU acceleration" ON)
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_language(CUDA)

INCLUDE_DIRECTORIES(
	src
	render/include
)

FILE(GLOB_RECURSE SLABHASH_SOURCES
	src/SlabHash/*.cpp
	src/SlabHash/*.h
	src/*.h
	src/SlabHash/*.cu
	src/SlabHash/*.cuh
)

FILE(GLOB_RECURSE RENDER_SOURCES
	render/src/*.cpp
	render/src/*.cu
	render/include/*.cuh
)

#include(${CMAKE_BINARY_DIR}/build/conanbuildinfo.cmake)
#conan_basic_setup()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/assimp/")
find_package(ASSIMP REQUIRED)
include_directories(${ASSIMP_INCLUDE_DIRS})

find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

FILE(GLOB_RECURSE TEST_SOURCES
	tests/SlabHash/*.cu
	tests/SlabHash/*.cuh
	tests/SlabHash/*.h)

set(CMAKE_CUDA_FLAGS "-gencode arch=compute_61,code=sm_61 -lineinfo -maxrregcount=32 -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
add_executable(test_slab ${RENDER_SOURCES} ${TEST_SOURCES} ${SLABHASH_SOURCES})
target_link_libraries(test_slab ${ASSIMP_LIBRARIES})
target_link_libraries(test_slab glfw)
target_link_libraries(test_slab glm)
set_target_properties(test_slab PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(WIN32)
	set_target_properties(test_slab PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
	ASSIMP_COPY_BINARIES(${CMAKE_BINARY_DIR})
endif(WIN32)
set_property(TARGET test_slab PROPERTY CUDA_ARCHITECTURES 61)