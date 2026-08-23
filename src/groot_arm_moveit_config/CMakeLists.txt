cmake_minimum_required(VERSION 3.16)
project(groot_arm_moveit_config)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config srdf launch rviz
  DESTINATION share/${PROJECT_NAME}
)
install(
  FILES .setup_assistant
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
