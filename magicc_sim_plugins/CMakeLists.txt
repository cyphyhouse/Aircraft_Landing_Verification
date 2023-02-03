cmake_minimum_required(VERSION 2.8.3)
project(magicc_sim_plugins)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")

# To enable assertions when compiled in release mode.
add_definitions(-DROS_ASSERT_ENABLED)

find_package(catkin REQUIRED COMPONENTS
  # cmake_modules
  cv_bridge
  geometry_msgs
  magicc_sim
  # rosbag
  roscpp
  std_srvs
  tf
  # rosflight_msgs
  # rosflight_utils
  sensor_msgs
  std_msgs
  tf2_ros
  image_transport
  polled_camera
  camera_info_manager
  # nav_msgs
  gazebo_ros
  gazebo_plugins
)

find_package(Eigen3 REQUIRED)
find_package(gazebo REQUIRED)
link_directories(${GAZEBO_LIBRARY_DIRS})
include_directories(${GAZEBO_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

catkin_package(
  INCLUDE_DIRS include ${Eigen_INCLUDE_DIRS}
  CATKIN_DEPENDS cv_bridge geometry_msgs magicc_sim rosbag roscpp
                 magicc_sim std_srvs tf relative_nav
		 cv_bridge geometry_msgs magicc_sim rosbag roscpp gazebo_plugins gazebo_ros
		 tf2_ros image_transport polled_camera camera_info_manager std_msgs nav_msgs sensor_msgs
  DEPENDS eigen gazebo opencv rosflight_msgs SDF roscpp gazebo_ros
)

include_directories(include ${catkin_INCLUDE_DIRS})
include_directories(${Eigen_INCLUDE_DIRS})

add_library(gimbal_plugin
  src/gimbal_plugin.cpp
  include/magicc_sim_plugins/gimbal_plugin.h)
target_link_libraries(gimbal_plugin ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})
add_dependencies(gimbal_plugin ${catkin_EXPORTED_TARGETS})

add_library(autolevel_plugin
  src/autolevel_plugin.cpp
  include/magicc_sim_plugins/autolevel_plugin.h)
target_link_libraries(autolevel_plugin ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})
add_dependencies(autolevel_plugin ${catkin_EXPORTED_TARGETS})

add_library(magicc_sim_camera_utils
  src/gazebo_ros_camera_utils.cpp)
target_link_libraries(magicc_sim_camera_utils ${GAZEBO_LIBRARIES} ${catkin_LIBRARIES} ${Boost_LIBRARIES})
add_dependencies(magicc_sim_camera_utils ${catkin_EXPORTED_TARGETS})

add_library(magicc_sim_camera_plugin
  src/gazebo_ros_camera.cpp)
target_link_libraries(magicc_sim_camera_plugin magicc_sim_camera_utils ${GAZEBO_LIBRARIES} CameraPlugin ${catkin_LIBRARIES})
add_dependencies(magicc_sim_camera_plugin ${catkin_EXPORTED_TARGETS})

add_library(step_camera
  src/step_camera.cpp
  include/magicc_sim_plugins/step_camera.h)
target_link_libraries(step_camera ${catkin_LIBRARIES} ${GAZEBO_libraries} CameraPlugin)
add_dependencies(step_camera ${catkin_EXPORTED_TARGETS})

# add_library(world_utilities
#   src/world_utilities.cpp
#   include/magicc_sim_plugins/world_utilities.h)
# target_link_libraries(world_utilities ${catkin_LIBRARIES} ${GAZEBO_libraries})
# add_dependencies(world_utilities ${catkin_EXPORTED_TARGETS})


install(
  TARGETS
    step_camera
    # world_utilities
    autolevel_plugin
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)
