"""RViz with the MoveIt MotionPlanning panel wired to this cell."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    # Absolute path: each package gets its own install prefix under colcon,
    # so a path relative to the moveit_config share dir would not resolve.
    description_xacro = os.path.join(
        get_package_share_directory("groot_arm_description"),
        "urdf",
        "groot_arm.urdf.xacro",
    )

    ur_type = LaunchConfiguration("ur_type")
    use_sim_time = LaunchConfiguration("use_sim_time")

    moveit_config = (
        MoveItConfigsBuilder("groot_arm", package_name="groot_arm_moveit_config")
        .robot_description(
            file_path=description_xacro,
            mappings={"ur_type": ur_type, "name": "groot_arm", "sim_gazebo": "false"},
        )
        .robot_description_semantic(
            file_path="srdf/groot_arm.srdf.xacro", mappings={"name": "groot_arm"}
        )
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"])
        .to_moveit_configs()
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="moveit_rviz",
        output="log",
        arguments=[
            "-d",
            PathJoinSubstitution(
                [FindPackageShare("groot_arm_moveit_config"), "rviz", "moveit.rviz"]
            ),
        ],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            {"use_sim_time": use_sim_time},
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("ur_type", default_value="ur5e"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            rviz_node,
        ]
    )
