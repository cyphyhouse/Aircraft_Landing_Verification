cmake_minimum_required(VERSION 2.8.3)
project(magicc_sim)

add_definitions(-std=c++11)

find_package(catkin 
  REQUIRED COMPONENTS
  gazebo_ros
  gazebo_msgs 
  geometry_msgs 
  roscpp 
  sensor_msgs
  message_generation
  std_msgs
)

find_package(cmake_modules REQUIRED)
find_package(Eigen REQUIRED)
find_package(gazebo REQUIRED)

link_directories(${GAZEBO_LIBRARY_DIRS})
include_directories(include ${catkin_INCLUDE_DIRS} ${GAZEBO_INCLUDE_DIRS})

# Configure Build
catkin_package(
  INCLUDE_DIRS ${EIGEN_INCLUDE_DIRS}
  CATKIN_DEPENDS
    gazebo_msgs
    geometry_msgs
    roscpp
    sensor_msgs
    message_runtime
  DEPENDS 
    Eigen
    gazebo_ros
)
