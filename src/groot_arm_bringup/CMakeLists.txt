cmake_minimum_required(VERSION 3.16)
project(groot_arm_bringup)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS scripts/kill_stack.sh scripts/collect_parallel.sh
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
