cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

project(graphics_test LANGUAGES CXX)
add_executable(test main.cpp)

find_package(PCL 1.10 REQUIRED COMPONENTS common io visualization)

if (PCL_FOUND)
	include_directories( ${PCL_INCLUDE_DIRS} )
	add_definitions( ${PCL_DEFINITIONS} )
	link_directories( ${PCL_LIBRARY_DIRS} )
	target_link_libraries( test ${PCL_LIBRARIES} )
endif()
