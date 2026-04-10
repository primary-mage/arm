cmake_minimum_required(VERSION 3.10)
project(dm_motor_sdk_ros)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(robot_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBUSB REQUIRED libusb-1.0)

set(DM_SDK_SO "${CMAKE_CURRENT_SOURCE_DIR}/third_party/dm_sdk/linux/libdm_device.so")
if(NOT EXISTS ${DM_SDK_SO})
  message(FATAL_ERROR "libdm_device.so not found at ${DM_SDK_SO}")
endif()

include_directories(
  include
  third_party/dm_sdk/include
)

add_library(dm_sdk_can_bsp
  src/can_bsp_sdk.cpp
)
target_link_libraries(dm_sdk_can_bsp
  ${DM_SDK_SO}
  ${LIBUSB_LIBRARIES}
  pthread
)
target_include_directories(dm_sdk_can_bsp PRIVATE ${LIBUSB_INCLUDE_DIRS})

add_library(dm_motor_protocol
  src/dm_motor_drv.c
  src/dm_motor_ctrl.c
)

add_executable(dm_motor_minimal_node
  src/dm_motor_minimal_node.cpp
)
ament_target_dependencies(dm_motor_minimal_node
  rclcpp
  std_msgs
)
target_link_libraries(dm_motor_minimal_node
  dm_motor_protocol
  dm_sdk_can_bsp
  ${LIBUSB_LIBRARIES}
)

add_executable(dm_motor_robot_driver_node
  src/dm_motor_robot_driver_node.cpp
)
ament_target_dependencies(dm_motor_robot_driver_node
  rclcpp
  robot_msgs
  std_msgs
)
target_link_libraries(dm_motor_robot_driver_node
  dm_motor_protocol
  dm_sdk_can_bsp
  ${LIBUSB_LIBRARIES}
)

add_executable(dm_motor_robot_stress_node
  src/dm_motor_robot_stress_node.cpp
)
ament_target_dependencies(dm_motor_robot_stress_node
  rclcpp
  robot_msgs
  std_msgs
)

add_executable(dm_motor_sdk_example
  src/dm_motor_sdk_example.cpp
)
target_link_libraries(dm_motor_sdk_example
  dm_motor_protocol
  dm_sdk_can_bsp
  ${LIBUSB_LIBRARIES}
)

add_executable(dm_motor_sdk_stress_test
  src/dm_motor_sdk_stress_test.cpp
)
target_link_libraries(dm_motor_sdk_stress_test
  dm_motor_protocol
  dm_sdk_can_bsp
  ${LIBUSB_LIBRARIES}
)

set_target_properties(dm_sdk_can_bsp dm_motor_minimal_node dm_motor_robot_driver_node dm_motor_robot_stress_node dm_motor_sdk_example dm_motor_sdk_stress_test PROPERTIES
  BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/dm_sdk/linux"
  INSTALL_RPATH "\$ORIGIN"
)

install(TARGETS
  dm_sdk_can_bsp
  dm_motor_protocol
  dm_motor_minimal_node
  dm_motor_robot_driver_node
  dm_motor_robot_stress_node
  dm_motor_sdk_example
  dm_motor_sdk_stress_test
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(FILES
  third_party/dm_sdk/linux/libdm_device.so
  DESTINATION lib
)

install(FILES
  launch/dm_motor_minimal.launch.py
  launch/dm_motor_robot_driver.launch.py
  launch/dm_motor_robot_stress.launch.py
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

ament_package()
