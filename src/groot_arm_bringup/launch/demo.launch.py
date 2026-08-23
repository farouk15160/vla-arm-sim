"""Full stack without the policy: Gazebo + controllers + MoveIt + RViz.

This is the layer you drive by hand (RViz MotionPlanning panel, or the
scripted pick-and-place in groot_vla). Add the policy with vla.launch.py.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ur_type = LaunchConfiguration("ur_type")
    launch_rviz = LaunchConfiguration("launch_rviz")
    gazebo_gui = LaunchConfiguration("gazebo_gui")

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("groot_arm_bringup"), "launch", "sim.launch.py"]
            )
        ),
        launch_arguments={"ur_type": ur_type, "gazebo_gui": gazebo_gui}.items(),
    )

    move_group = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("groot_arm_moveit_config"),
                    "launch",
                    "move_group.launch.py",
                ]
            )
        ),
        launch_arguments={"ur_type": ur_type, "use_sim_time": "true"}.items(),
    )

    rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("groot_arm_moveit_config"),
                    "launch",
                    "moveit_rviz.launch.py",
                ]
            )
        ),
        launch_arguments={"ur_type": ur_type, "use_sim_time": "true"}.items(),
        condition=IfCondition(launch_rviz),
    )

    # move_group's planning scene monitor needs /joint_states and a running
    # controller_manager to advertise its controller list; starting it too
    # early leaves it in a permanent "no controllers" state.
    delayed_moveit = TimerAction(period=6.0, actions=[move_group, rviz])

    return LaunchDescription(
        [
            DeclareLaunchArgument("ur_type", default_value="ur5e"),
            DeclareLaunchArgument("launch_rviz", default_value="true"),
            DeclareLaunchArgument("gazebo_gui", default_value="true"),
            sim,
            delayed_moveit,
        ]
    )
