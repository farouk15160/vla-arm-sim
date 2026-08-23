"""moveit_servo node, for groot_vla's eef_delta action space.

Start this only when the policy emits Cartesian velocities.

Servo is configured to publish JointTrajectory to /arm_controller/joint_trajectory,
i.e. the same controller MoveIt uses, so NO controller switch is needed - the
arm_controller stays active and Servo and move_group simply take turns. (The
alternative, streaming raw positions to forward_position_controller, requires
deactivating arm_controller and breaks MoveIt planning while active.)

After launching, Servo is running but has NO command type selected and will
reject every message with "Command type has not been set". Select one first:

    ros2 service call /servo_node/switch_command_type \
        moveit_msgs/srv/ServoCommandType "{command_type: 1}"   # 1 = TWIST

then stream to /servo_node/delta_twist_cmds. groot_vla's policy_node does this
call for you when action_space:=eef_delta. (There is no start_servo service in
Jazzy - that was the Humble API; use pause_servo/unpause_servo.)
"""

import os

import yaml

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
        .to_moveit_configs()
    )

    # moveit_servo expects its whole config nested under a "moveit_servo" key.
    # A ParameterFile object cannot be nested inside a dict, so the YAML is
    # loaded here and passed as plain values.
    servo_yaml = os.path.join(
        get_package_share_directory("groot_arm_moveit_config"), "config", "servo.yaml"
    )
    with open(servo_yaml) as handle:
        servo_params = {"moveit_servo": yaml.safe_load(handle)}

    servo_node = Node(
        package="moveit_servo",
        executable="servo_node",
        name="servo_node",
        output="screen",
        parameters=[
            servo_params,
            # Required by online_signal_smoothing::AccelerationLimitedPlugin.
            # It is read from the node's root namespace, not from inside the
            # "moveit_servo" block, so it must be passed separately or the node
            # aborts with ParameterUninitializedException.
            {"update_period": 0.01},
            # Also read from the root namespace, separately from
            # moveit_servo.move_group_name, by the smoothing plugin.
            {"planning_group_name": "ur_manipulator"},
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
            {"use_sim_time": use_sim_time},
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("ur_type", default_value="ur5e"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            servo_node,
        ]
    )
