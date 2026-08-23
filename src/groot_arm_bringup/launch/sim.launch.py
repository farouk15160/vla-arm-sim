"""Gazebo simulation of the GR00T arm cell: physics, controllers, cameras.

Brings up, in order:
  * robot_state_publisher (Gazebo-flavoured description)
  * gz sim with worlds/tabletop.sdf
  * the robot spawned into that world
  * ros_gz bridges: /clock, camera_info, and the two RGB image streams
  * controller spawners

MoveIt is deliberately NOT started here - see demo.launch.py.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    FindExecutable,
    IfElseSubstitution,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ur_type = LaunchConfiguration("ur_type")
    gazebo_gui = LaunchConfiguration("gazebo_gui")
    world_file = LaunchConfiguration("world_file")

    description_pkg = FindPackageShare("groot_arm_description")
    controllers_file = PathJoinSubstitution(
        [description_pkg, "config", "ros2_controllers.yaml"]
    )

    robot_description_content = Command(
        [
            FindExecutable(name="xacro"), " ",
            PathJoinSubstitution([description_pkg, "urdf", "groot_arm.urdf.xacro"]),
            " name:=groot_arm",
            " ur_type:=", ur_type,
            " sim_gazebo:=true",
            " simulation_controllers:=", controllers_file,
        ]
    )
    robot_description = {
        "robot_description": ParameterValue(robot_description_content, value_type=str)
    }

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description, {"use_sim_time": True}],
    )

    # -r starts unpaused; -s runs headless (server only).
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("ros_gz_sim"), "/launch/gz_sim.launch.py"]
        ),
        launch_arguments={
            "gz_args": IfElseSubstitution(
                gazebo_gui,
                if_value=[" -r -v 3 ", world_file],
                else_value=[" -s -r -v 3 ", world_file],
            )
        }.items(),
    )

    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-string", robot_description_content,
            "-name", "groot_arm",
            "-allow_renaming", "true",
            # Base is at world origin; the pedestal already lifts it to table height.
            "-x", "0", "-y", "0", "-z", "0",
        ],
    )

    # /clock and camera_info. Images go through image_bridge below, which is
    # markedly cheaper than routing them via parameter_bridge.
    gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="gz_bridge",
        output="screen",
        parameters=[
            {
                "config_file": os.path.join(
                    get_package_share_directory("groot_arm_description"),
                    "config",
                    "gz_bridge.yaml",
                ),
                "use_sim_time": True,
            }
        ],
    )

    image_bridge = Node(
        package="ros_gz_image",
        executable="image_bridge",
        name="camera_image_bridge",
        output="screen",
        arguments=["/wrist_camera/image_raw", "/scene_camera/image_raw"],
        parameters=[{"use_sim_time": True}],
    )

    def spawner(name, *extra):
        return Node(
            package="controller_manager",
            executable="spawner",
            arguments=[name, "-c", "/controller_manager", *extra],
            output="screen",
        )

    joint_state_broadcaster = spawner("joint_state_broadcaster")
    arm_controller = spawner("arm_controller")
    gripper_controller = spawner("gripper_controller")
    # Loaded but inactive: only moveit_servo (eef_delta mode) claims it, and it
    # cannot coexist with arm_controller on the same joints.
    forward_position_controller = spawner("forward_position_controller", "--inactive")

    # Controllers must not be spawned before the broadcaster has claimed the
    # state interfaces, or the spawner races the plugin's startup.
    controllers_after_jsb = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster,
            on_exit=[arm_controller, gripper_controller, forward_position_controller],
        )
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "ur_type",
                default_value="ur5e",
                choices=["ur3e", "ur5e", "ur10e", "ur16e"],
                description="UR series to simulate.",
            ),
            DeclareLaunchArgument(
                "gazebo_gui",
                default_value="true",
                description="Run the Gazebo GUI. Set false for headless / CI.",
            ),
            DeclareLaunchArgument(
                "world_file",
                default_value=PathJoinSubstitution(
                    [description_pkg, "worlds", "tabletop.sdf"]
                ),
                description="SDF world to load.",
            ),
            robot_state_publisher,
            gz_sim,
            spawn_robot,
            gz_bridge,
            image_bridge,
            joint_state_broadcaster,
            controllers_after_jsb,
        ]
    )
