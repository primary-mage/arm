cmake_minimum_required(VERSION 3.8)
project(realsense_v4l_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(realsense_v4l_camera_node
  src/realsense_v4l_camera_node.cpp
)
target_compile_features(realsense_v4l_camera_node PUBLIC cxx_std_17)
ament_target_dependencies(realsense_v4l_camera_node
  rclcpp
  sensor_msgs
  cv_bridge
)
target_link_libraries(realsense_v4l_camera_node
  ${OpenCV_LIBS}
)

install(TARGETS
  realsense_v4l_camera_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
