cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(hashTablesCuda LANGUAGES CXX CUDA)

OPTION(USE_CUDA "Build with GPU acceleration" ON)
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_language(CUDA)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
INCLUDE_DIRECTORIES(includes ${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})
	
#set(CMAKE_CUDA_FLAGS "-G -g")
add_executable(test simpleGL.cu)
target_link_libraries(test ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})
set_target_properties(test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
