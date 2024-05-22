#CMake minimum requirement 
cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set(exec_name "FaceViewer_FLAME")

#GLOB source files 
#file(GLOB_RECURSE src_files ${CMAKE_CURRENT_SOURCE_DIR}/include/*.*)#<TODO> change this to add your files 

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(FreeImage REQUIRED)
find_package(glfw3 REQUIRED)
find_package(ZLIB REQUIRED)

include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${ZLIB_INCLUDE_DIRS})

#Executable
add_executable(${exec_name} FaceViewer_FLAME.cpp)

include_directories(${CMAKE_SOURCE_DIR}/third_party/)
include_directories(${CMAKE_SOURCE_DIR}/src/)

#Libs linked to the executable
# target_link_libraries( ${exec_name}   
# 	utils
# 	$<BUILD_INTERFACE:developer_flags>	
# 	cuGaussianSplatting
#     imgui
# )

target_link_libraries(${exec_name}
    utils
    gsutils
    cnpy
    imgui
    freeimage::FreeImage
    OpenGL::GL
    GLU
    glfw
    ${GLEW_LIBRARIES}
    cuGaussianSplatting
)

#gtest_discover_tests(${exec_name})
