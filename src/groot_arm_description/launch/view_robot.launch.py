"""Description-only view: robot_state_publisher + joint sliders + RViz.

No Gazebo, no controllers - use this to sanity-check the URDF, the gripper
kinematics and the camera frames after editing any xacro.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ur_type = LaunchConfiguration("ur_type")

    robot_description = {
        "robot_description": Command(
            [
                FindExecutable(name="xacro"),
                " ",
                PathJoinSubstitution(
                    [FindPackageShare("groot_arm_description"), "urdf", "groot_arm.urdf.xacro"]
                ),
                " ur_type:=", ur_type,
                " name:=groot_arm",
                # Gazebo tags are irrelevant here and force absolute mesh paths.
                " sim_gazebo:=false",
            ]
        )
    }

    return LaunchDescription(
        [
            DeclareLaunchArgument("ur_type", default_value="ur5e"),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                output="screen",
                parameters=[robot_description],
            ),
            Node(
                package="joint_state_publisher_gui",
                executable="joint_state_publisher_gui",
                output="screen",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=[
                    "-d",
                    PathJoinSubstitution(
                        [FindPackageShare("groot_arm_description"), "rviz", "view_robot.rviz"]
                    ),
                ],
            ),
        ]
    )
