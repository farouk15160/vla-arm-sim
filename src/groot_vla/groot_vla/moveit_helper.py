"""Minimal MoveIt 2 client built on actions and services only.

``moveit_py`` is not packaged for Jazzy, so this talks to move_group directly:

    /move_action              (action)   plan + execute a joint-space goal
    /compute_ik               (service)  Cartesian pose -> joint values
    /compute_cartesian_path   (service)  straight-line segments
    /execute_trajectory       (action)   run a precomputed trajectory
    /gripper_controller/follow_joint_trajectory (action)

Pose goals go through IK rather than pose constraints deliberately: it fails
loudly and immediately when a pose is unreachable, instead of burning the
planner's whole time budget first.
"""

from __future__ import annotations

import math
import time
from typing import Sequence

from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    CollisionObject,
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    PlanningOptions,
    PlanningScene,
    PositionIKRequest,
    RobotState,
)
from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath, GetPositionIK
from shape_msgs.msg import SolidPrimitive
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

GRIPPER_OPEN = 0.04
# Commanded finger position for a grasp. The 40 mm cubes stall the fingers at
# about 8 mm, so this leaves ~2 mm of squeeze rather than 8 mm.
GRASP_SQUEEZE = 0.006

ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
GRIPPER_JOINTS = ("left_finger_joint", "right_finger_joint")


class MoveItError(RuntimeError):
    """Planning, IK or execution failed."""


def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """RPY (XYZ, fixed axes) -> (x, y, z, w). Avoids a transforms3d dependency."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def make_pose(
    x: float, y: float, z: float, roll: float = math.pi, pitch: float = 0.0, yaw: float = 0.0
) -> Pose:
    """Pose with a top-down default orientation (tool +z pointing at the table)."""
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = x, y, z
    qx, qy, qz, qw = quaternion_from_euler(roll, pitch, yaw)
    pose.orientation.x, pose.orientation.y = qx, qy
    pose.orientation.z, pose.orientation.w = qz, qw
    return pose


class MoveItHelper:
    """Blocking convenience wrapper. Call from a thread that is NOT spinning."""

    def __init__(
        self,
        node: Node,
        group_name: str = "ur_manipulator",
        ik_link: str = "tcp_link",
        base_frame: str = "base_link",
        planning_time: float = 5.0,
        velocity_scaling: float = 0.25,
        acceleration_scaling: float = 0.25,
    ) -> None:
        self.node = node
        self.group_name = group_name
        self.ik_link = ik_link
        self.base_frame = base_frame
        self.planning_time = planning_time
        self.velocity_scaling = velocity_scaling
        self.acceleration_scaling = acceleration_scaling

        # One executor for the helper's whole lifetime. rclpy.spin_until_future
        # _complete() builds a throwaway executor per call; across the two waits
        # an action needs (goal response, then result) the second executor never
        # sees the first's subscriptions, and goal responses are dropped with
        # "Ignoring unexpected goal response".
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(node)
        group = ReentrantCallbackGroup()

        self._move_group = ActionClient(node, MoveGroup, "/move_action", callback_group=group)
        self._execute = ActionClient(
            node, ExecuteTrajectory, "/execute_trajectory", callback_group=group
        )
        self._gripper = ActionClient(
            node,
            FollowJointTrajectory,
            "/gripper_controller/follow_joint_trajectory",
            callback_group=group,
        )
        self._ik = node.create_client(GetPositionIK, "/compute_ik", callback_group=group)
        self._cartesian = node.create_client(
            GetCartesianPath, "/compute_cartesian_path", callback_group=group
        )
        self._apply_scene = node.create_client(
            ApplyPlanningScene, "/apply_planning_scene", callback_group=group
        )

        self._joint_state: JointState | None = None
        node.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)

    def _on_joint_state(self, msg: JointState) -> None:
        self._joint_state = msg

    # ------------------------------------------------------------------ #
    def wait_for_services(self, timeout: float = 30.0) -> None:
        checks = (
            (self._move_group.wait_for_server, "/move_action"),
            (self._execute.wait_for_server, "/execute_trajectory"),
            (self._gripper.wait_for_server, "/gripper_controller/follow_joint_trajectory"),
            (self._ik.wait_for_service, "/compute_ik"),
            (self._cartesian.wait_for_service, "/compute_cartesian_path"),
        )
        for wait, name in checks:
            if not wait(timeout_sec=timeout):
                raise MoveItError(f"timed out waiting for {name}")
        # A joint state is required before any goal can carry a start state.
        # This must spin: without it the subscription callback never runs and
        # the loop would wait forever. Wall time is used for the deadline so
        # the check does not depend on /clock already flowing.
        deadline = time.monotonic() + timeout
        while self._joint_state is None:
            if time.monotonic() > deadline:
                raise MoveItError("no /joint_states received")
            self._executor.spin_once(timeout_sec=0.1)

    def current_joint_positions(self) -> dict[str, float]:
        if self._joint_state is None:
            raise MoveItError("no /joint_states received yet")
        return dict(zip(self._joint_state.name, self._joint_state.position))

    def _robot_state(self) -> RobotState:
        state = RobotState()
        if self._joint_state is not None:
            state.joint_state = self._joint_state
        return state

    # ------------------------------------------------------------------ #
    def move_to_joints(self, targets: dict[str, float], tolerance: float = 0.01) -> None:
        """Plan and execute a joint-space goal, blocking until it finishes."""
        constraints = Constraints()
        for joint, value in targets.items():
            constraint = JointConstraint()
            constraint.joint_name = joint
            constraint.position = float(value)
            constraint.tolerance_above = tolerance
            constraint.tolerance_below = tolerance
            constraint.weight = 1.0
            constraints.joint_constraints.append(constraint)

        request = MotionPlanRequest()
        request.group_name = self.group_name
        request.goal_constraints.append(constraints)
        request.num_planning_attempts = 10
        request.allowed_planning_time = self.planning_time
        request.max_velocity_scaling_factor = self.velocity_scaling
        request.max_acceleration_scaling_factor = self.acceleration_scaling
        request.start_state = self._robot_state()

        goal = MoveGroup.Goal()
        goal.request = request
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = False

        result = self._call_action(self._move_group, goal)
        if result.error_code.val != 1:  # moveit_msgs/MoveItErrorCodes.SUCCESS
            raise MoveItError(f"joint goal failed, MoveItErrorCode={result.error_code.val}")

    def compute_ik(self, pose: Pose, timeout: float = 2.0) -> dict[str, float]:
        request = GetPositionIK.Request()
        request.ik_request = PositionIKRequest()
        request.ik_request.group_name = self.group_name
        request.ik_request.ik_link_name = self.ik_link
        request.ik_request.robot_state = self._robot_state()
        request.ik_request.avoid_collisions = True
        request.ik_request.timeout = _duration(timeout)

        target = PoseStamped()
        target.header.frame_id = self.base_frame
        target.pose = pose
        request.ik_request.pose_stamped = target

        response = self._call_service(self._ik, request)
        if response.error_code.val != 1:
            raise MoveItError(
                f"no IK solution for pose "
                f"({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f}); "
                f"MoveItErrorCode={response.error_code.val}"
            )
        solution = dict(
            zip(response.solution.joint_state.name, response.solution.joint_state.position)
        )
        return {joint: solution[joint] for joint in ARM_JOINTS if joint in solution}

    def move_to_pose(self, pose: Pose) -> None:
        self.move_to_joints(self.compute_ik(pose))

    def cartesian_move(
        self, waypoints: Sequence[Pose], step: float = 0.005, min_fraction: float = 0.9
    ) -> None:
        """Straight-line motion through ``waypoints``; used for approach/retreat."""
        request = GetCartesianPath.Request()
        request.header.frame_id = self.base_frame
        request.start_state = self._robot_state()
        request.group_name = self.group_name
        request.link_name = self.ik_link
        request.waypoints = list(waypoints)
        request.max_step = step
        request.avoid_collisions = True
        request.max_velocity_scaling_factor = self.velocity_scaling
        request.max_acceleration_scaling_factor = self.acceleration_scaling

        response = self._call_service(self._cartesian, request)
        if response.fraction < min_fraction:
            raise MoveItError(
                f"Cartesian path only {response.fraction:.0%} complete "
                f"(needed {min_fraction:.0%}); the segment is blocked or unreachable"
            )

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = response.solution
        result = self._call_action(self._execute, goal)
        if result.error_code.val != 1:
            raise MoveItError(
                f"Cartesian execution failed, MoveItErrorCode={result.error_code.val}"
            )

    # ------------------------------------------------------------------ #
    def add_collision_box(
        self,
        object_id: str,
        size: tuple[float, float, float],
        position: tuple[float, float, float],
        frame: str | None = None,
    ) -> None:
        """Put a static box into the planning scene.

        Without this the planner has no idea the table exists and will happily
        route the gripper straight through it. Positions are given in
        ``frame`` (default: the planning base frame), at the box CENTRE.
        """
        if not self._apply_scene.wait_for_service(timeout_sec=10.0):
            raise MoveItError("timed out waiting for /apply_planning_scene")

        collision_object = CollisionObject()
        collision_object.id = object_id
        collision_object.header.frame_id = frame or self.base_frame
        collision_object.operation = CollisionObject.ADD

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [float(v) for v in size]
        collision_object.primitives.append(primitive)

        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = position
        pose.orientation.w = 1.0
        collision_object.primitive_poses.append(pose)

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.append(collision_object)

        request = ApplyPlanningScene.Request()
        request.scene = scene
        response = self._call_service(self._apply_scene, request)
        if not response.success:
            raise MoveItError(f"failed to add collision object {object_id!r}")

    def add_default_scene(self, pedestal_height: float = 0.6) -> None:
        """Register the work surface from worlds/tabletop.sdf with the planner.

        The box is deliberately NARROWER in x than the real table. The SDF table
        spans x = -0.05..0.95 and so interpenetrates the arm's pedestal at the
        origin - harmless between two static bodies in Gazebo, but as a
        collision object it would put the robot permanently in self-collision
        and every plan would be rejected. Starting the box at x = 0.12 clears
        the pedestal while still covering the whole reachable workspace
        (cubes at x = 0.38..0.55, tray at x = 0.5).
        """
        x_min, x_max = 0.12, 0.95
        width = x_max - x_min
        centre_x = (x_min + x_max) / 2.0

        # Work surface: top face at world z = 0.60, i.e. base_link z = 0.
        self.add_collision_box(
            "work_table",
            size=(width, 1.4, 0.05),
            position=(centre_x, 0.0, 0.575 - pedestal_height),
        )
        # Solid volume beneath it, so the planner cannot route under the top.
        self.add_collision_box(
            "table_base",
            size=(width, 1.25, 0.55),
            position=(centre_x, 0.0, 0.275 - pedestal_height),
        )

    # ------------------------------------------------------------------ #
    def set_gripper(
        self, position: float, duration: float = 1.0, verify: bool = True
    ) -> float:
        """Command both fingers and return the finger position actually reached.

        A stalled finger and a dead finger both abort with
        GOAL_TOLERANCE_VIOLATED, so the error code alone cannot tell a
        successful grasp from a gripper that never moved. When ``verify`` is
        set, the joint state is compared before and after and a complete
        failure to move is raised rather than silently accepted.
        """
        before = self.current_joint_positions().get(GRIPPER_JOINTS[0])

        trajectory = JointTrajectory()
        trajectory.joint_names = list(GRIPPER_JOINTS)
        point = JointTrajectoryPoint()
        point.positions = [float(position)] * len(GRIPPER_JOINTS)
        point.time_from_start = _duration(duration)
        trajectory.points.append(point)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        result = self._call_action(self._gripper, goal)

        # PATH/GOAL_TOLERANCE_VIOLATED is the NORMAL outcome of a grasp: the
        # fingers stall on the object short of the commanded position.
        if result.error_code not in (
            FollowJointTrajectory.Result.SUCCESSFUL,
            FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED,
            FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED,
        ):
            raise MoveItError(
                f"gripper command failed, code={result.error_code} "
                f"({result.error_string})"
            )

        # Let the joint settle before reading back.
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            self._executor.spin_once(timeout_sec=0.02)
        after = self.current_joint_positions().get(GRIPPER_JOINTS[0])

        if verify and before is not None and after is not None:
            travelled = abs(after - before)
            requested = abs(position - before)
            # Moved essentially nothing while a real move was asked for.
            if requested > 0.005 and travelled < 0.002:
                raise MoveItError(
                    f"gripper did not move: commanded {position:.4f}, still at "
                    f"{after:.4f} (controller said '{result.error_string}'). "
                    "The finger joints are jammed - restart the simulation."
                )
        return after if after is not None else float("nan")

    def open_gripper(self) -> float:
        return self.set_gripper(GRIPPER_OPEN)

    def close_gripper(self, width: float = GRASP_SQUEEZE) -> float:
        """Close onto an object.

        ``width`` is deliberately NOT 0.0. Commanding the hard stop leaves the
        position controller driving the full 8 mm of error into a stiff
        contact for as long as the object is held, which destabilises the
        contact solver and can leave the prismatic joints jammed for the rest
        of the session. A target just inside the object's half-width gives a
        firm grip with a couple of millimetres of squeeze.
        """
        return self.set_gripper(width)

    # ------------------------------------------------------------------ #
    # blocking call helpers - all waits go through the one shared executor
    # ------------------------------------------------------------------ #
    def _spin_until(self, future, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() > deadline:
                return False
            self._executor.spin_once(timeout_sec=0.05)
        return True

    def _call_service(self, client, request, timeout: float = 30.0):
        if not client.service_is_ready():
            raise MoveItError(f"service {client.srv_name} is not available")
        future = client.call_async(request)
        if not self._spin_until(future, timeout):
            raise MoveItError(f"service {client.srv_name} timed out")
        return future.result()

    def _call_action(self, client: ActionClient, goal, timeout: float = 120.0):
        send_future = client.send_goal_async(goal)
        if not self._spin_until(send_future, 30.0):
            raise MoveItError("timed out waiting for the goal to be accepted")
        handle = send_future.result()
        if handle is None or not handle.accepted:
            raise MoveItError("goal was rejected by the action server")

        result_future = handle.get_result_async()
        if not self._spin_until(result_future, timeout):
            raise MoveItError("action timed out")
        return result_future.result().result


# --------------------------------------------------------------------------- #
# blocking helpers
# --------------------------------------------------------------------------- #
def _duration(seconds: float):
    from builtin_interfaces.msg import Duration

    return Duration(sec=int(seconds), nanosec=int((seconds % 1.0) * 1e9))
