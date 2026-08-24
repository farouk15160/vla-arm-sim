cmake_minimum_required(VERSION 3.16)
project(groot_arm_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY urdf config worlds launch rviz materials meshes
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
