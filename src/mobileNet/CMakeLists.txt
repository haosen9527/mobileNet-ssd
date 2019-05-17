cmake_minimum_required(VERSION 2.8.3)
project(mobileNet)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11 -O2)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
)
#set(OpenCV_DIR /opt/ros/kinetic/include/opencv-3.3.1-dev/opencv2)
#find_package(OpenCV 3.3.1 EXACT REQUIRED)
set(OpenCV_DIR /usr/local/opencv341/share/OpenCV)
## System dependencies are found with CMake's conventions
find_package(OpenCV 3.4.1 EXACT REQUIRED)
catkin_package(
  INCLUDE_DIRS include
#  LIBRARIES mobileNet
#  CATKIN_DEPENDS roscpp rospy
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories( "/home/haosen/tensorflow/tensorflow/contrib/makefile/downloads/eigen" )

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  "include"
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  "/usr/local/include/eigen3"
  "~/tensorflow"
  "~/tensorflow/bazel-genfiles"
  "~/tensorflow/tensorflow/contrib/makefile/gen/protobuf/include"
  "~/tensorflow/tensorflow/contrib/makefile/downloads/nsync/public"
)
link_directories("~/tensorflow/bazel-bin/tensorflow")

## Declare a C++ library
##mobilenet
add_library(${PROJECT_NAME}
  src/mobileNet.cpp
)

target_link_libraries(${PROJECT_NAME}
  tensorflow_cc
  tensorflow_framework
  ${catkin_LIBRARIES}
)
## SSD
add_library(${PROJECT_NAME}_ssd
  src/ssd.cpp
)

target_link_libraries(${PROJECT_NAME}_ssd
  tensorflow_cc
  tensorflow_framework
  ${catkin_LIBRARIES}
)
## mobilenet test
add_executable(${PROJECT_NAME}_test
    src/mobileNet.cpp
    src/mobilenet_test.cpp
    src/ssd.cpp
)

target_link_libraries(${PROJECT_NAME}_test
  tensorflow_cc
  tensorflow_framework
  ${catkin_LIBRARIES}
)

## mobilenet read PB
add_executable(${PROJECT_NAME}_pb
    src/mobilenet-ssd_readpb.cpp
)

target_link_libraries(${PROJECT_NAME}_pb
  tensorflow_cc
  tensorflow_framework
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
)
##mobilenet opencv dnn
add_executable(${PROJECT_NAME}_opencv
    src/mobilenet-ssd-opencv.cpp
    )
target_link_libraries(${PROJECT_NAME}_opencv
    tensorflow_cc
    tensorflow_framework
    ${OpenCV_LIBS}
    ${catkin_LIBRARIES}
    )


add_library(lib${PROJECT_NAME}
    src/pbtotensorflow.cpp
)

target_link_libraries(lib${PROJECT_NAME}
  tensorflow_cc
  tensorflow_framework
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
)

## mobilenet read PB test
add_executable(${PROJECT_NAME}_api
    src/pbtotensorflow.cpp
    test/apiTest.cpp
)

target_link_libraries(${PROJECT_NAME}_api
  tensorflow_cc
  tensorflow_framework
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
)    
    
    
    
    
