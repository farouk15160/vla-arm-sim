"""One launch file for the whole cell: sim, MoveIt, RViz, policy server, GUI.

Replaces the three-terminal dance. The policy server runs in its own venv (it
needs torch, which must not go into the ROS environment), started here as a
plain subprocess.

    # everything, with the mock policy - no GPU needed
    ros2 launch groot_arm_bringup system.launch.py

    # everything, with real SmolVLA on the GPU
    ros2 launch groot_arm_bringup system.launch.py policy:=smolvla

    # OpenVLA 4-bit, which emits end-effector deltas, so Servo comes up too
    ros2 launch groot_arm_bringup system.launch.py policy:=openvla

    # no policy at all - just the robot, MoveIt, RViz and the GUI
    ros2 launch groot_arm_bringup system.launch.py policy:=none

Useful arguments:
    policy:=none|mock|smolvla|openvla   which server to start   (default mock)
    gui:=true|false                     Qt control panel        (default true)
    rviz:=true|false                    RViz                    (default true)
    gazebo_gui:=true|false              Gazebo GUI              (default true)
    goal_marker:=true|false             draggable RViz goal     (default true)
    instruction:="..."                  the task
    policy_host / policy_port           point at a remote server instead
    venv_python:=~/vla_venv/bin/python  interpreter for smolvla
    openvla_python:=~/openvla_venv/bin/python   interpreter for openvla
    model_path:=...                     override the checkpoint

OpenVLA gets its own venv on purpose: it pins transformers 4.40, while LeRobot
requires >= 4.57. The two cannot share an environment, and since each policy
server is a separate process that costs nothing.

Set policy_host to something other than 127.0.0.1 and no local server is
started - the stack just connects to yours.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# Which action space each policy speaks. OpenVLA emits 7-DoF end-effector
# deltas, so it needs moveit_servo; the others emit joint targets.
POLICY_ACTION_SPACE = {
    "mock": "joint_position",
    "smolvla": "joint_position",
    "openvla": "eef_delta",
    "none": "joint_position",
}

POLICY_CONFIG = {
    "mock": "groot_policy.yaml",
    "smolvla": "smolvla_policy.yaml",
    "openvla": "openvla_policy.yaml",
}


def launch_setup(context, *args, **kwargs):
    policy = LaunchConfiguration("policy").perform(context)
    policy_host = LaunchConfiguration("policy_host").perform(context)
    policy_port = LaunchConfiguration("policy_port").perform(context)
    venv_python = os.path.expanduser(LaunchConfiguration("venv_python").perform(context))
    openvla_python = os.path.expanduser(LaunchConfiguration("openvla_python").perform(context))
    model_path = LaunchConfiguration("model_path").perform(context)
    instruction = LaunchConfiguration("instruction").perform(context)

    if policy not in POLICY_ACTION_SPACE:
        raise RuntimeError(
            f"policy:={policy} is not one of {sorted(POLICY_ACTION_SPACE)}"
        )

    vla_share = get_package_share_directory("groot_vla")
    server_dir = os.path.join(vla_share, "servers")
    actions = []

    # ---------------------------------------------------------------- server
    # Only start a server when the policy is local. A remote host means the
    # user is running it themselves.
    is_local = policy_host in ("127.0.0.1", "localhost", "0.0.0.0")
    if policy != "none" and is_local:
        if policy == "mock":
            # The mock has no heavy dependencies, so the ROS interpreter is fine.
            command = [
                "python3", os.path.join(server_dir, "mock_policy_server.py"),
                "--port", policy_port, "--behaviour", "wave",
            ]
        else:
            script = f"{policy}_server.py"
            # OpenVLA pins transformers 4.40; LeRobot needs >= 4.57. Separate
            # interpreters, one per policy.
            interpreter = openvla_python if policy == "openvla" else venv_python
            if not os.path.exists(interpreter):
                raise RuntimeError(
                    f"policy:={policy} needs {interpreter}, which does not exist. "
                    "Create it (see README) or pass "
                    f"{'openvla_python' if policy == 'openvla' else 'venv_python'}"
                    ":=/path/to/python."
                )
            command = [interpreter, os.path.join(server_dir, script), "--port", policy_port]
            if model_path:
                command += ["--model-path", model_path]

        actions.append(LogInfo(msg=f"[system] starting {policy} policy server on port {policy_port}"))
        actions.append(
            ExecuteProcess(cmd=command, output="screen", name=f"{policy}_server", shell=False)
        )
    elif policy != "none":
        actions.append(
            LogInfo(msg=f"[system] using remote policy server at {policy_host}:{policy_port}")
        )

    # ---------------------------------------------------------------- servo
    action_space = POLICY_ACTION_SPACE[policy]
    execution_backend = LaunchConfiguration("execution_backend").perform(context)
    if execution_backend == "moveit" and action_space == "eef_delta":
        raise RuntimeError(
            f"policy:={policy} streams eef_delta to moveit_servo, which "
            "bypasses move_group by design; execution_backend:=moveit plans "
            "to joint goals instead. The two cannot both drive the arm."
        )
    if action_space == "eef_delta":
        actions.append(
            TimerAction(period=10.0, actions=[
                LogInfo(msg="[system] eef_delta policy: starting moveit_servo"),
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        PathJoinSubstitution([
                            FindPackageShare("groot_arm_moveit_config"),
                            "launch", "servo.launch.py",
                        ])
                    ),
                    launch_arguments={"use_sim_time": "true"}.items(),
                ),
            ])
        )

    # --------------------------------------------------------------- policy
    if policy != "none":
        parameter_file = os.path.join(vla_share, "config", POLICY_CONFIG[policy])
        # The model needs longer than the sim to come up, especially OpenVLA,
        # which loads 7B of weights before it answers a ping.
        delay = 45.0 if policy == "openvla" else (25.0 if policy == "smolvla" else 14.0)
        actions.append(
            TimerAction(period=delay, actions=[
                Node(
                    package="groot_vla",
                    executable="policy_node",
                    name="groot_policy",
                    output="screen",
                    parameters=[
                        parameter_file,
                        {
                            "policy_host": policy_host,
                            "policy_port": int(policy_port),
                            "instruction": instruction,
                            "action_space": action_space,
                            "execution_backend": execution_backend,
                            "use_sim_time": True,
                        },
                    ],
                ),
            ])
        )

    return actions


def generate_launch_description():
    ur_type = LaunchConfiguration("ur_type")

    demo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("groot_arm_bringup"), "launch", "demo.launch.py"]
            )
        ),
        launch_arguments={
            "ur_type": ur_type,
            "launch_rviz": LaunchConfiguration("rviz"),
            "gazebo_gui": LaunchConfiguration("gazebo_gui"),
        }.items(),
    )

    # Draws the Gazebo world in RViz and registers the static parts with the
    # planner, so the two views agree.
    world_publisher = TimerAction(period=8.0, actions=[
        Node(
            package="groot_vla",
            executable="world_publisher",
            name="world_publisher",
            output="screen",
            parameters=[{"use_sim_time": True}],
        ),
    ])

    goal_marker = TimerAction(period=10.0, actions=[
        Node(
            package="groot_vla",
            executable="goal_marker",
            name="goal_marker",
            output="screen",
            parameters=[{"use_sim_time": True}],
            condition=IfCondition(LaunchConfiguration("goal_marker")),
        ),
    ])

    gui = TimerAction(period=12.0, actions=[
        Node(
            package="groot_vla",
            executable="control_gui",
            name="control_gui",
            output="screen",
            parameters=[{"use_sim_time": True}],
            condition=IfCondition(LaunchConfiguration("gui")),
        ),
    ])

    return LaunchDescription([
        DeclareLaunchArgument("ur_type", default_value="ur5e"),
        DeclareLaunchArgument(
            "policy", default_value="mock",
            choices=["none", "mock", "smolvla", "openvla"],
            description="which policy server to start",
        ),
        DeclareLaunchArgument("policy_host", default_value="127.0.0.1"),
        DeclareLaunchArgument("policy_port", default_value="5555"),
        DeclareLaunchArgument(
            "venv_python", default_value="~/vla_venv/bin/python",
            description="interpreter that has torch + lerobot (smolvla)",
        ),
        DeclareLaunchArgument(
            "openvla_python", default_value="~/openvla_venv/bin/python",
            description="interpreter with transformers 4.40 (openvla)",
        ),
        DeclareLaunchArgument("model_path", default_value=""),
        DeclareLaunchArgument(
            "instruction",
            default_value="pick up the red cube and place it in the tray",
        ),
        DeclareLaunchArgument(
            "execution_backend",
            default_value="direct",
            description=(
                "How the policy's joint targets reach the arm. 'direct' "
                "publishes them to the trajectory controller. 'moveit' hands "
                "each one to move_group as a goal, so the path is planned "
                "around the planning scene and checked for collisions."
            ),
            choices=["direct", "moveit"],
        ),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("gazebo_gui", default_value="true"),
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("goal_marker", default_value="true"),
        demo,
        world_publisher,
        goal_marker,
        gui,
        OpaqueFunction(function=launch_setup),
    ])
