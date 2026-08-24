"""ROS 2 side of the Unity bridge.

Starts ros_tcp_endpoint, which Unity's ROS-TCP-Connector connects to over a
plain TCP socket. Unity then publishes camera images and joint states, and
subscribes to joint trajectories, on exactly the same topics Gazebo uses - so
the policy stack does not know or care which simulator is behind them.

    # Unity as the simulator, everything else unchanged
    ros2 launch groot_arm_bringup unity_bridge.launch.py
    ros2 launch groot_arm_bringup system.launch.py policy:=smolvla simulator:=none

Set ros_ip to this machine's LAN address if Unity runs on another computer;
the default binds every interface.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    endpoint = Node(
        package="ros_tcp_endpoint",
        executable="default_server_endpoint",
        name="unity_endpoint",
        output="screen",
        parameters=[{
            "ROS_IP": LaunchConfiguration("ros_ip"),
            "ROS_TCP_PORT": LaunchConfiguration("ros_tcp_port"),
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "ros_ip", default_value="0.0.0.0",
            description="interface to bind; 0.0.0.0 accepts Unity from anywhere"),
        DeclareLaunchArgument("ros_tcp_port", default_value="10000"),
        LogInfo(msg="[unity] endpoint starting - point Unity's ROS Settings at this host:port"),
        endpoint,
    ])
