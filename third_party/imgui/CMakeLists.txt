cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set(lib_name "imgui")

file(GLOB src_files ${CMAKE_CURRENT_SOURCE_DIR}/*.*)

find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})

add_library(${lib_name} STATIC ${src_files})
