cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set(lib_name "gsutils")

file(GLOB_RECURSE src_files ${CMAKE_CURRENT_SOURCE_DIR}/*.*)

find_package(ZLIB REQUIRED)
include_directories(${ZLIB_INCLUDE_DIRS})

find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})

include_directories(${CMAKE_SOURCE_DIR}/third_party/)
include_directories(${CMAKE_SOURCE_DIR}/src/)

add_library(${lib_name} STATIC ${src_files})
