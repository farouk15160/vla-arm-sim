"""Everything: simulation + MoveIt + the GR00T policy loop.

Typical uses:

  # 1. smoke test with the bundled mock policy server (no GPU needed)
  ros2 run groot_vla mock_policy_server --port 5555
  ros2 launch groot_arm_bringup vla.launch.py

  # 2. against a real GR00T N1.7 server on another machine
  ros2 launch groot_arm_bringup vla.launch.py \
      policy_host:=10.0.0.42 policy_port:=5555 \
      instruction:="pick up the red cube and put it in the tray"

The policy starts disabled; enable it once the scene looks right:
  ros2 service call /groot_policy/enable std_srvs/srv/SetBool "{data: true}"
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ur_type = LaunchConfiguration("ur_type")
    launch_rviz = LaunchConfiguration("launch_rviz")
    gazebo_gui = LaunchConfiguration("gazebo_gui")

    demo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("groot_arm_bringup"), "launch", "demo.launch.py"]
            )
        ),
        launch_arguments={
            "ur_type": ur_type,
            "launch_rviz": launch_rviz,
            "gazebo_gui": gazebo_gui,
        }.items(),
    )

    policy = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("groot_vla"), "launch", "groot_policy.launch.py"]
            )
        ),
        launch_arguments={
            "policy_host": LaunchConfiguration("policy_host"),
            "policy_port": LaunchConfiguration("policy_port"),
            "instruction": LaunchConfiguration("instruction"),
            "action_space": LaunchConfiguration("action_space"),
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("ur_type", default_value="ur5e"),
            DeclareLaunchArgument("launch_rviz", default_value="true"),
            DeclareLaunchArgument("gazebo_gui", default_value="true"),
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
            demo,
            # Give the controllers and move_group time to come up before the
            # policy starts sampling observations.
            TimerAction(period=12.0, actions=[policy]),
        ]
    )
