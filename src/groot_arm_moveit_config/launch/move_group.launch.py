"""Start move_group for the GR00T arm cell.

Intentionally does NOT start robot_state_publisher or ros2_control - those come
from groot_arm_bringup/sim.launch.py, so this file can also be pointed at a
different backend (mock hardware, real UR driver) without change.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
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

    # ur_type is a LaunchConfiguration, so the xacro is resolved lazily via
    # ParameterValue(Xacro(...)) inside the builder.
    moveit_config = (
        MoveItConfigsBuilder("groot_arm", package_name="groot_arm_moveit_config")
        .robot_description(
            file_path=description_xacro,
            mappings={
                "ur_type": ur_type,
                "name": "groot_arm",
                "sim_gazebo": "false",
            },
        )
        .robot_description_semantic(
            file_path="srdf/groot_arm.srdf.xacro",
            mappings={"name": "groot_arm"},
        )
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"])
        .pilz_cartesian_limits(file_path="config/pilz_cartesian_limits.yaml")
        .planning_scene_monitor(
            publish_robot_description=True,
            publish_robot_description_semantic=True,
            publish_planning_scene=True,
        )
        .to_moveit_configs()
    )

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"use_sim_time": use_sim_time},
            # Without this move_group refuses trajectories whose start state
            # differs slightly from the controller's, which happens constantly
            # when the VLA and MoveIt share the arm.
            {"trajectory_execution.allowed_start_tolerance": 0.05},
            {"trajectory_execution.allowed_execution_duration_scaling": 2.5},
            {"publish_robot_description_semantic": True},
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("ur_type", default_value="ur5e"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            move_group_node,
        ]
    )
