"""Start the GR00T policy node against a running simulation."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    parameter_file = PathJoinSubstitution(
        [FindPackageShare("groot_vla"), "config", "groot_policy.yaml"]
    )

    policy_node = Node(
        package="groot_vla",
        executable="policy_node",
        name="groot_policy",
        output="screen",
        parameters=[
            parameter_file,
            {
                "policy_host": LaunchConfiguration("policy_host"),
                "policy_port": LaunchConfiguration("policy_port"),
                "instruction": LaunchConfiguration("instruction"),
                "action_space": LaunchConfiguration("action_space"),
                "start_enabled": LaunchConfiguration("start_enabled"),
                "use_sim_time": True,
            },
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("policy_host", default_value="127.0.0.1"),
            DeclareLaunchArgument("policy_port", default_value="5555"),
            DeclareLaunchArgument(
                "instruction",
                default_value="pick up the red cube and place it in the tray",
            ),
            DeclareLaunchArgument(
                "action_space",
                default_value="joint_position",
                choices=["joint_position", "joint_delta", "eef_delta"],
            ),
            DeclareLaunchArgument("start_enabled", default_value="false"),
            policy_node,
        ]
    )
