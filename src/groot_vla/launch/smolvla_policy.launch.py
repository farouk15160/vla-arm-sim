"""Run the VLA bridge against a SmolVLA server.

Identical to groot_policy.launch.py apart from the defaults, because the ROS
side does not know or care which policy is behind the socket. Only the timing
differs: SmolVLA is ~1 Hz on a 6 GB card, GR00T is faster on a big one.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    parameter_file = PathJoinSubstitution(
        [FindPackageShare("groot_vla"), "config", "smolvla_policy.yaml"]
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
                "control_rate": LaunchConfiguration("control_rate"),
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
            DeclareLaunchArgument("control_rate", default_value="1.0"),
            policy_node,
        ]
    )
