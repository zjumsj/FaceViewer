#CMake minimum requirement 
cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set(lib_name "cuGaussianSplatting")

#GLOB source files 
#file(GLOB_RECURSE src_files ${CMAKE_CURRENT_SOURCE_DIR}/include/*.*)#<TODO> change this to add your files 
file(GLOB_RECURSE src_files ${CMAKE_CURRENT_SOURCE_DIR}/*.*)#<TODO> change this to add your files 

find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})

#Executable
# if(${MY_BUILD_SHARED_LIBS})
# 	add_library(${lib_name} SHARED ${src_files})
# else()
# 	add_library(${lib_name} STATIC ${src_files})
# endif()

add_library(${lib_name} STATIC ${src_files})

#message("CMAKE_SOURCE_DIR=${CMAKE_SOURCE_DIR}")
#target_include_directories(${lib_name} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")
target_include_directories(${lib_name} PUBLIC "${CMAKE_SOURCE_DIR}/third_party/")

#Pass the flags to the library
target_link_libraries( ${lib_name}
	PUBLIC $<BUILD_INTERFACE:developer_flags>	
)
