"""Closed-loop GR00T N1.7 controller for the simulated UR5e cell.

    cameras + /joint_states  ->  observation  ->  GR00T server  ->  action chunk
                                                        |
                             JointTrajectory / TwistStamped  ->  ros2_control

Design notes
------------
* Inference runs on its own thread. A GR00T forward pass takes tens to hundreds
  of milliseconds; doing it in a timer callback would stall the executor and
  starve the subscriptions that produce the next observation.
* The node starts **disabled**. Enable it deliberately once the scene is set:
      ros2 service call /groot_policy/enable std_srvs/srv/SetBool "{data: true}"
* Every command passes through ActionMapper's clamps, plus a workspace box
  check and a stale-observation watchdog here. Treat these as load-bearing:
  the policy is free to emit anything at all.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from moveit_msgs.srv import ServoCommandType
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from groot_vla.action_mapper import ARM_JOINTS, ActionDecodeError, ActionMapper, joint_positions_to_array
from groot_vla.groot_client import GrootClient, PolicyServerError
from groot_vla.observation_builder import CameraSpec, ImageDecodeError, ObservationBuilder, image_to_rgb, resize_rgb

# Gazebo publishes images best-effort; a reliable subscription would silently
# receive nothing.
SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class GrootPolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("groot_policy")

        # ------------------------------- parameters ------------------------ #
        self.declare_parameter("policy_host", "127.0.0.1")
        self.declare_parameter("policy_port", 5555)
        self.declare_parameter("policy_timeout_ms", 15000)
        self.declare_parameter("api_token", "")

        self.declare_parameter("instruction", "pick up the red cube and place it in the tray")
        self.declare_parameter("observation_schema", "nested")
        self.declare_parameter("action_space", "joint_position")
        self.declare_parameter("arm_action_key", "single_arm")
        self.declare_parameter("gripper_action_key", "gripper")

        self.declare_parameter("wrist_camera_topic", "/wrist_camera/image_raw")
        self.declare_parameter("scene_camera_topic", "/scene_camera/image_raw")
        self.declare_parameter("wrist_camera_key", "wrist_view")
        self.declare_parameter("scene_camera_key", "ego_view")
        self.declare_parameter("image_width", 224)
        self.declare_parameter("image_height", 224)
        self.declare_parameter("use_wrist_camera", True)
        self.declare_parameter("use_scene_camera", True)

        self.declare_parameter("control_rate", 10.0)
        # How many steps of each returned chunk to actually execute before
        # re-inferring. Smaller = more reactive, more inference calls.
        self.declare_parameter("execution_horizon", 8)
        self.declare_parameter("action_dt", 0.1)
        self.declare_parameter("max_joint_step", 0.15)
        self.declare_parameter("gripper_open_position", 0.04)
        self.declare_parameter("gripper_closed_position", 0.0)
        self.declare_parameter("invert_gripper_action", False)

        self.declare_parameter("arm_command_topic", "/arm_controller/joint_trajectory")
        self.declare_parameter("gripper_command_topic", "/gripper_controller/joint_trajectory")
        self.declare_parameter("servo_twist_topic", "/servo_node/delta_twist_cmds")
        # moveit_servo halts if commands stop arriving (incoming_command_timeout,
        # 0.25 s by default). A slow policy - OpenVLA is ~5 s per forward pass -
        # would therefore move the arm in tiny stutters. The most recent twist
        # is republished at this rate to keep Servo fed between inferences.
        self.declare_parameter("twist_republish_rate", 20.0)
        # Safety cap: how long a single twist may keep being replayed before it
        # is treated as stale and the arm is commanded to stop.
        self.declare_parameter("twist_hold_time", 1.5)

        self.declare_parameter("observation_timeout", 1.0)
        self.declare_parameter("workspace_frame", "base_link")
        self.declare_parameter("tcp_frame", "tcp_link")
        # [xmin, xmax, ymin, ymax, zmin, zmax] in workspace_frame. The arm base
        # sits at table height, so z is measured from the work surface.
        self.declare_parameter("workspace_bounds", [-0.2, 0.95, -0.75, 0.75, -0.10, 0.85])
        self.declare_parameter("enforce_workspace", True)
        self.declare_parameter("start_enabled", False)
        self.declare_parameter("dry_run", False)

        self._load_parameters()

        # ------------------------------- state ----------------------------- #
        self._lock = threading.Lock()
        self._latest_images: dict[str, tuple[np.ndarray, float]] = {}
        self._joint_positions: dict[str, float] = {}
        self._joint_stamp = 0.0
        self._enabled = bool(self.get_parameter("start_enabled").value)
        self._shutdown = threading.Event()
        self._inference_count = 0
        self._failure_count = 0
        self._last_latency_ms = 0.0
        self._last_error = ""
        self._last_twist: np.ndarray | None = None
        self._last_twist_time = 0.0

        # ------------------------------- ROS I/O --------------------------- #
        sensor_group = ReentrantCallbackGroup()
        service_group = MutuallyExclusiveCallbackGroup()

        if self.use_wrist_camera:
            self.create_subscription(
                Image,
                self.wrist_camera_topic,
                lambda msg: self._on_image(msg, self.wrist_camera_key),
                SENSOR_QOS,
                callback_group=sensor_group,
            )
        if self.use_scene_camera:
            self.create_subscription(
                Image,
                self.scene_camera_topic,
                lambda msg: self._on_image(msg, self.scene_camera_key),
                SENSOR_QOS,
                callback_group=sensor_group,
            )
        self.create_subscription(
            JointState, "/joint_states", self._on_joint_state, 10, callback_group=sensor_group
        )
        self.create_subscription(
            String, "~/instruction", self._on_instruction, 10, callback_group=sensor_group
        )

        latching = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self._arm_publisher = self.create_publisher(JointTrajectory, self.arm_command_topic, 10)
        self._gripper_publisher = self.create_publisher(JointTrajectory, self.gripper_command_topic, 10)
        self._twist_publisher = self.create_publisher(TwistStamped, self.servo_twist_topic, 10)
        self._status_publisher = self.create_publisher(String, "~/status", latching)

        self.create_service(SetBool, "~/enable", self._on_enable, callback_group=service_group)
        self.create_service(Trigger, "~/halt", self._on_halt, callback_group=service_group)
        self.create_service(Trigger, "~/reset_policy", self._on_reset_policy, callback_group=service_group)

        # eef_delta streams through moveit_servo, which rejects all input until
        # a command type is selected. Done on enable, not here, so the node
        # still starts cleanly when servo is not running.
        # A dedicated reentrant group is required, NOT service_group: the
        # ~/enable callback blocks waiting on this client's response, and a
        # MutuallyExclusiveCallbackGroup cannot process that response while one
        # of its own callbacks is still running. Sharing the group deadlocks
        # until the call times out.
        self._servo_command_type = self.create_client(
            ServoCommandType,
            "/servo_node/switch_command_type",
            callback_group=ReentrantCallbackGroup(),
        )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.create_timer(1.0, self._publish_status, callback_group=service_group)

        if self.action_space == "eef_delta" and self.twist_republish_rate > 0.0:
            self.create_timer(
                1.0 / self.twist_republish_rate,
                self._republish_twist,
                callback_group=sensor_group,
            )

        # ------------------------------- pipeline -------------------------- #
        cameras: list[CameraSpec] = []
        if self.use_wrist_camera:
            cameras.append(
                CameraSpec(self.wrist_camera_topic, self.wrist_camera_key,
                           self.image_width, self.image_height)
            )
        if self.use_scene_camera:
            cameras.append(
                CameraSpec(self.scene_camera_topic, self.scene_camera_key,
                           self.image_width, self.image_height)
            )
        if not cameras:
            raise RuntimeError("at least one camera must be enabled")

        self._observation_builder = ObservationBuilder(
            cameras=cameras,
            schema=self.observation_schema,
            gripper_joint_range=(self.gripper_closed_position, self.gripper_open_position),
        )

        gripper_range = (self.gripper_closed_position, self.gripper_open_position)
        if bool(self.get_parameter("invert_gripper_action").value):
            gripper_range = (self.gripper_open_position, self.gripper_closed_position)

        self._action_mapper = ActionMapper(
            action_space=self.action_space,
            arm_action_key=self.arm_action_key,
            gripper_action_key=self.gripper_action_key,
            max_joint_step=float(self.get_parameter("max_joint_step").value),
            gripper_range=gripper_range,
        )

        self._client = GrootClient(
            host=self.policy_host,
            port=self.policy_port,
            timeout_ms=int(self.get_parameter("policy_timeout_ms").value),
            api_token=(self.get_parameter("api_token").value or None),
        )

        self._worker = threading.Thread(target=self._control_loop, daemon=True, name="groot_inference")
        self._worker.start()

        self.get_logger().info(
            f"groot_policy up. server={self._client.endpoint} "
            f"action_space={self.action_space} schema={self.observation_schema} "
            f"cameras={[c.modality_key for c in cameras]} "
            f"enabled={self._enabled}"
        )
        if not self._enabled:
            self.get_logger().info(
                "Policy is DISABLED. Start it with: "
                "ros2 service call /groot_policy/enable std_srvs/srv/SetBool \"{data: true}\""
            )

    # ------------------------------------------------------------------ #
    # parameters
    # ------------------------------------------------------------------ #
    def _load_parameters(self) -> None:
        get = lambda name: self.get_parameter(name).value  # noqa: E731
        self.policy_host = str(get("policy_host"))
        self.policy_port = int(get("policy_port"))
        self.instruction = str(get("instruction"))
        self.observation_schema = str(get("observation_schema"))
        self.action_space = str(get("action_space"))
        self.arm_action_key = str(get("arm_action_key"))
        self.gripper_action_key = str(get("gripper_action_key"))
        self.wrist_camera_topic = str(get("wrist_camera_topic"))
        self.scene_camera_topic = str(get("scene_camera_topic"))
        self.wrist_camera_key = str(get("wrist_camera_key"))
        self.scene_camera_key = str(get("scene_camera_key"))
        self.image_width = int(get("image_width"))
        self.image_height = int(get("image_height"))
        self.use_wrist_camera = bool(get("use_wrist_camera"))
        self.use_scene_camera = bool(get("use_scene_camera"))
        self.control_rate = float(get("control_rate"))
        self.execution_horizon = int(get("execution_horizon"))
        self.action_dt = float(get("action_dt"))
        self.gripper_open_position = float(get("gripper_open_position"))
        self.gripper_closed_position = float(get("gripper_closed_position"))
        self.arm_command_topic = str(get("arm_command_topic"))
        self.gripper_command_topic = str(get("gripper_command_topic"))
        self.servo_twist_topic = str(get("servo_twist_topic"))
        self.twist_republish_rate = float(get("twist_republish_rate"))
        self.twist_hold_time = float(get("twist_hold_time"))
        self.observation_timeout = float(get("observation_timeout"))
        self.workspace_frame = str(get("workspace_frame"))
        self.tcp_frame = str(get("tcp_frame"))
        self.workspace_bounds = [float(v) for v in get("workspace_bounds")]
        self.enforce_workspace = bool(get("enforce_workspace"))
        self.dry_run = bool(get("dry_run"))

    # ------------------------------------------------------------------ #
    # subscriptions
    # ------------------------------------------------------------------ #
    def _on_image(self, msg: Image, key: str) -> None:
        try:
            rgb = resize_rgb(image_to_rgb(msg), self.image_width, self.image_height)
        except ImageDecodeError as exc:
            self.get_logger().warn(f"dropping frame on {key}: {exc}", throttle_duration_sec=5.0)
            return
        with self._lock:
            self._latest_images[key] = (rgb, time.monotonic())

    def _on_joint_state(self, msg: JointState) -> None:
        with self._lock:
            for name, position in zip(msg.name, msg.position):
                self._joint_positions[name] = float(position)
            self._joint_stamp = time.monotonic()

    def _on_instruction(self, msg: String) -> None:
        text = msg.data.strip()
        if not text:
            return
        self.instruction = text
        self.get_logger().info(f"instruction -> {text!r}")

    # ------------------------------------------------------------------ #
    # services
    # ------------------------------------------------------------------ #
    def _on_enable(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        if request.data and not self._client.ping():
            response.success = False
            response.message = (
                f"GR00T server at {self._client.endpoint} is not answering; "
                "refusing to enable"
            )
            self.get_logger().error(response.message)
            return response

        if request.data and self.action_space == "eef_delta":
            if not self._select_servo_twist_mode():
                response.success = False
                response.message = (
                    "could not put moveit_servo into TWIST mode; is "
                    "servo.launch.py running?"
                )
                self.get_logger().error(response.message)
                return response

        self._enabled = bool(request.data)
        if not self._enabled:
            self._halt()
        response.success = True
        response.message = f"policy {'enabled' if self._enabled else 'disabled'}"
        self.get_logger().info(response.message)
        return response

    def _select_servo_twist_mode(self, timeout: float = 5.0) -> bool:
        """Put moveit_servo into TWIST mode so it accepts our commands."""
        if not self._servo_command_type.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(
                "/servo_node/switch_command_type is not available"
            )
            return False
        request = ServoCommandType.Request()
        request.command_type = ServoCommandType.Request.TWIST
        future = self._servo_command_type.call_async(request)
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not future.done():
            self.get_logger().error("switch_command_type timed out")
            return False
        return bool(future.result().success)

    def _on_halt(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        self._enabled = False
        self._halt()
        response.success = True
        response.message = "policy disabled and arm halted"
        return response

    def _on_reset_policy(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """Clear the server's action-chunk history between episodes."""
        try:
            self._client.reset()
            response.success = True
            response.message = "server state reset"
        except PolicyServerError as exc:
            response.success = False
            response.message = f"reset failed: {exc}"
        return response

    # ------------------------------------------------------------------ #
    # control loop (worker thread)
    # ------------------------------------------------------------------ #
    def _control_loop(self) -> None:
        period = 1.0 / max(self.control_rate, 1e-3)
        while not self._shutdown.is_set():
            started = time.monotonic()
            if self._enabled:
                try:
                    self._step()
                except Exception as exc:  # noqa: BLE001 - a policy fault must
                    # disable the arm, never kill the control thread
                    self._failure_count += 1
                    self._last_error = str(exc)
                    self.get_logger().error(f"inference step failed: {exc}")
                    self._enabled = False
                    self._halt()
            remaining = period - (time.monotonic() - started)
            self._shutdown.wait(max(remaining, 0.0))

    def _step(self) -> None:
        observation, joint_positions = self._collect_observation()
        current_arm = joint_positions_to_array(joint_positions, ARM_JOINTS)

        started = time.monotonic()
        raw_action = self._client.get_action(observation)
        self._last_latency_ms = (time.monotonic() - started) * 1000.0
        self._inference_count += 1

        arm_chunk, gripper_chunk = self._action_mapper.decode(raw_action)

        if self.action_space == "eef_delta":
            twist = self._action_mapper.to_twist(arm_chunk)
            with self._lock:
                self._last_twist = twist
                self._last_twist_time = time.monotonic()
            self._publish_twist(twist)
        else:
            targets = self._action_mapper.to_joint_targets(arm_chunk, current_arm)
            targets = targets[: max(self.execution_horizon, 1)]
            self._check_workspace()
            self._publish_arm_trajectory(targets)

        if gripper_chunk.size:
            fingers = self._action_mapper.to_finger_targets(gripper_chunk)
            self._publish_gripper(float(fingers[0]))

    def _collect_observation(self) -> tuple[dict[str, Any], dict[str, float]]:
        now = time.monotonic()
        with self._lock:
            images = {key: value for key, (value, _stamp) in self._latest_images.items()}
            stamps = {key: stamp for key, (_value, stamp) in self._latest_images.items()}
            joint_positions = dict(self._joint_positions)
            joint_stamp = self._joint_stamp

        expected = {c.modality_key for c in self._observation_builder.cameras}
        missing = expected - set(images)
        if missing:
            raise RuntimeError(f"no frames yet from camera(s): {sorted(missing)}")

        stale = [k for k, s in stamps.items() if now - s > self.observation_timeout]
        if stale:
            raise RuntimeError(
                f"camera stream(s) stale by >{self.observation_timeout}s: {sorted(stale)}"
            )
        if not joint_positions:
            raise RuntimeError("no /joint_states received yet")
        if now - joint_stamp > self.observation_timeout:
            raise RuntimeError(f"/joint_states stale by >{self.observation_timeout}s")

        observation = self._observation_builder.build(
            images=images, joint_positions=joint_positions, instruction=self.instruction
        )
        return observation, joint_positions

    # ------------------------------------------------------------------ #
    # safety
    # ------------------------------------------------------------------ #
    def _check_workspace(self) -> None:
        """Raise if the tool has left the allowed box.

        A missing transform is not treated as a violation - TF can lag at
        startup - but it is logged, because silently skipping a safety check
        is worse than a noisy log.
        """
        if not self.enforce_workspace:
            return
        try:
            transform = self._tf_buffer.lookup_transform(
                self.workspace_frame, self.tcp_frame, rclpy.time.Time()
            )
        except Exception as exc:  # noqa: BLE001 - tf2 raises several types
            self.get_logger().warn(
                f"workspace check skipped, no {self.workspace_frame}->{self.tcp_frame} "
                f"transform: {exc}",
                throttle_duration_sec=10.0,
            )
            return

        t = transform.transform.translation
        xmin, xmax, ymin, ymax, zmin, zmax = self.workspace_bounds
        if not (xmin <= t.x <= xmax and ymin <= t.y <= ymax and zmin <= t.z <= zmax):
            raise RuntimeError(
                f"TCP left the workspace box at "
                f"({t.x:.3f}, {t.y:.3f}, {t.z:.3f}) in {self.workspace_frame}"
            )

    def _halt(self) -> None:
        """Freeze the arm at its current measured position."""
        with self._lock:
            joint_positions = dict(self._joint_positions)
        if not all(j in joint_positions for j in ARM_JOINTS):
            return
        current = joint_positions_to_array(joint_positions, ARM_JOINTS)
        self._publish_arm_trajectory(current[np.newaxis, :], hold=True)
        if self.action_space == "eef_delta":
            with self._lock:
                self._last_twist = None
            self._publish_twist(np.zeros(6))
        self.get_logger().warn("arm halted")

    # ------------------------------------------------------------------ #
    # publishing
    # ------------------------------------------------------------------ #
    def _publish_arm_trajectory(self, targets: np.ndarray, hold: bool = False) -> None:
        if self.dry_run:
            self.get_logger().info(f"[dry_run] arm target: {np.round(targets[-1], 4)}")
            return

        trajectory = JointTrajectory()
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.joint_names = list(ARM_JOINTS)
        for index, target in enumerate(targets):
            point = JointTrajectoryPoint()
            point.positions = [float(v) for v in target]
            point.velocities = [0.0] * len(ARM_JOINTS)
            seconds = 0.25 if hold else self.action_dt * (index + 1)
            point.time_from_start.sec = int(seconds)
            point.time_from_start.nanosec = int((seconds % 1.0) * 1e9)
            trajectory.points.append(point)
        self._arm_publisher.publish(trajectory)

    def _publish_gripper(self, position: float) -> None:
        if self.dry_run:
            self.get_logger().info(f"[dry_run] gripper -> {position:.4f}")
            return
        low = min(self.gripper_closed_position, self.gripper_open_position)
        high = max(self.gripper_closed_position, self.gripper_open_position)
        position = float(np.clip(position, low, high))

        trajectory = JointTrajectory()
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.joint_names = ["left_finger_joint", "right_finger_joint"]
        point = JointTrajectoryPoint()
        point.positions = [position, position]
        point.velocities = [0.0, 0.0]
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = int(0.3 * 1e9)
        trajectory.points.append(point)
        self._gripper_publisher.publish(trajectory)

    def _publish_twist(self, twist: np.ndarray) -> None:
        if self.dry_run:
            self.get_logger().info(f"[dry_run] twist: {np.round(twist, 4)}")
            return
        message = TwistStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.workspace_frame
        message.twist.linear.x, message.twist.linear.y, message.twist.linear.z = twist[:3]
        message.twist.angular.x, message.twist.angular.y, message.twist.angular.z = twist[3:6]
        self._twist_publisher.publish(message)

    def _republish_twist(self) -> None:
        """Keep moveit_servo fed between inferences.

        Replays the most recent twist until it ages past ``twist_hold_time``,
        then sends zeros once so Servo decelerates cleanly instead of being
        cut off mid-motion by its own timeout.
        """
        if not self._enabled:
            return
        with self._lock:
            twist = self._last_twist
            age = time.monotonic() - self._last_twist_time
        if twist is None:
            return
        if age > self.twist_hold_time:
            with self._lock:
                self._last_twist = None
            self._publish_twist(np.zeros(6))
            return
        self._publish_twist(twist)

    def _publish_status(self) -> None:
        status = {
            "enabled": self._enabled,
            "server": self._client.endpoint,
            "instruction": self.instruction,
            "action_space": self.action_space,
            "inferences": self._inference_count,
            "failures": self._failure_count,
            "last_latency_ms": round(self._last_latency_ms, 1),
            "last_error": self._last_error,
        }
        self._status_publisher.publish(String(data=json.dumps(status)))

    def destroy_node(self) -> bool:
        self._shutdown.set()
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        self._client.close()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = GrootPolicyNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
