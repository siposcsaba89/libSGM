cmake_minimum_required(VERSION 2.8)

project(sgm_demo)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

message(STATUS ${CMAKE_MODULE_PATH})

FIND_PACKAGE(CUDA  REQUIRED)

find_package(OpenCV REQUIRED core) # dammy 
if(OpenCV_VERSION VERSION_LESS 3.0) 
	find_package(OpenCV REQUIRED core highgui imgproc)  # 2.x
else()
	find_package(OpenCV REQUIRED) # 3.x
endif()

find_package(OpenGL REQUIRED)
find_package(GLFW REQUIRED)
find_package(glew REQUIRED CONFIG)
find_package(aw-io REQUIRED CONFIG)


include_directories(../../include)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${GLFW_INCLUDE_DIRS})

CUDA_ADD_EXECUTABLE(stereo_movie
stereosgm_movie.cpp
demo.cpp
demo.h
renderer.cpp
renderer.h
cuda_gl.cu
)

TARGET_LINK_LIBRARIES(stereo_movie sgm  ${CUDA_LIBRARIES} ${CUDA_nppi_LIBRARY} ${OpenCV_LIBS} ::glew ${GLFW_LIBRARY} ${OPENGL_LIBRARY} ::aw-io)
