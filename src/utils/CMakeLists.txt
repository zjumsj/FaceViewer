cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set(lib_name "utils")

file(GLOB_RECURSE src_files ${CMAKE_CURRENT_SOURCE_DIR}/*.*)

#include("${CMAKE_SOURCE_DIR}/cmake/FindFreeImage.cmake")

find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})

find_package(FreeImage REQUIRED)
include_directories(${FREEIMAGE_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/third_party/)

add_library(${lib_name} STATIC ${src_files})

