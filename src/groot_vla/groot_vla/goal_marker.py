"""A draggable 3D goal marker in RViz that commands MoveIt.

Adds an interactive marker at the tool tip. Drag it anywhere in the workspace,
then either release it (with auto_go enabled) or pick "Move here" from its
right-click menu: the node runs IK for the grasp frame, plans, and executes.

Add it in RViz with:  Displays -> Add -> InteractiveMarkers,
                      topic /goal_marker/update

Menu entries:
  Move here            plan and execute to the marker pose
  Reset marker to TCP  snap the marker back onto the current tool position
  Open / Close gripper
  Auto-go on release   toggle planning automatically when you let go

Motion runs on a worker thread so the marker stays responsive while the arm is
moving, and a second request is refused while one is in flight rather than
queueing up behind it.
"""

from __future__ import annotations

import threading

import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from interactive_markers import InteractiveMarkerServer, MenuHandler
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import (
    InteractiveMarker,
    InteractiveMarkerControl,
    InteractiveMarkerFeedback,
    Marker,
)

from groot_vla.moveit_helper import MoveItError, MoveItHelper


class GoalMarkerNode(Node):
    def __init__(self) -> None:
        super().__init__("goal_marker")

        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("tcp_frame", "tcp_link")
        self.declare_parameter("auto_go", False)
        self.declare_parameter("marker_scale", 0.12)
        # Fallback pose if TF is not up yet: a sensible spot above the table.
        self.declare_parameter("initial_xyz", [0.45, 0.0, 0.15])

        self.base_frame = str(self.get_parameter("base_frame").value)
        self.tcp_frame = str(self.get_parameter("tcp_frame").value)
        self.auto_go = bool(self.get_parameter("auto_go").value)
        self.scale = float(self.get_parameter("marker_scale").value)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._busy = threading.Lock()
        self._status = self.create_publisher(String, "~/status", 10)
        # Current marker pose, so the GUI can show the target without having to
        # subscribe to the interactive-marker protocol itself.
        self._pose_publisher = self.create_publisher(PoseStamped, "~/goal_pose", 10)
        self.create_timer(0.2, self._publish_pose)

        # Lets the GUI (or any script) trigger the move without a right-click.
        # Reentrant: the callback blocks on the motion thread's lock check, and
        # a mutually-exclusive group would stall the node's other callbacks.
        self.create_service(
            Trigger, "~/go_to_marker", self._on_go_service,
            callback_group=ReentrantCallbackGroup(),
        )

        # Separate node for MoveIt: MoveItHelper spins its own executor inside
        # blocking calls, and this node is driven by a MultiThreadedExecutor.
        # One node cannot belong to both.
        self._moveit_node = rclpy.create_node("goal_marker_moveit")
        self._moveit = MoveItHelper(self._moveit_node, base_frame=self.base_frame)
        self._server = InteractiveMarkerServer(self, "goal_marker")
        self._menu = MenuHandler()
        self._build_menu()
        self._make_marker()

        self.get_logger().info(
            "goal_marker ready. In RViz: Add -> InteractiveMarkers -> "
            "topic /goal_marker/update. Drag the marker, then right-click -> Move here."
        )

    # ------------------------------------------------------------------ #
    def _build_menu(self) -> None:
        self._menu.insert("Move here", callback=self._on_move_here)
        self._menu.insert("Reset marker to TCP", callback=self._on_reset_marker)
        self._menu.insert("Open gripper", callback=self._on_open)
        self._menu.insert("Close gripper", callback=self._on_close)
        self._auto_entry = self._menu.insert("Auto-go on release", callback=self._on_toggle_auto)
        self._menu.setCheckState(
            self._auto_entry,
            MenuHandler.CHECKED if self.auto_go else MenuHandler.UNCHECKED,
        )

    def _current_tcp_pose(self) -> Pose:
        pose = Pose()
        try:
            transform = self._tf_buffer.lookup_transform(
                self.base_frame, self.tcp_frame, rclpy.time.Time()
            ).transform
            pose.position.x = transform.translation.x
            pose.position.y = transform.translation.y
            pose.position.z = transform.translation.z
            pose.orientation = transform.rotation
        except Exception:  # noqa: BLE001 - tf2 raises several types
            x, y, z = [float(v) for v in self.get_parameter("initial_xyz").value]
            pose.position.x, pose.position.y, pose.position.z = x, y, z
            # Tool pointing down, matching the top-down grasp convention.
            pose.orientation.x = 1.0
            pose.orientation.w = 0.0
        return pose

    def _make_marker(self) -> None:
        marker = InteractiveMarker()
        marker.header.frame_id = self.base_frame
        marker.name = "goal"
        marker.description = "MoveIt goal"
        marker.scale = self.scale
        marker.pose = self._current_tcp_pose()

        # Visible body: a small sphere so the marker reads as a point in space.
        sphere = Marker()
        sphere.type = Marker.SPHERE
        sphere.scale.x = sphere.scale.y = sphere.scale.z = self.scale * 0.35
        sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = 0.1, 0.8, 1.0, 0.9

        visual = InteractiveMarkerControl()
        visual.always_visible = True
        visual.interaction_mode = InteractiveMarkerControl.MENU
        visual.markers.append(sphere)
        marker.controls.append(visual)

        # Six-DOF handles. The quaternions below select the axis each ring and
        # arrow pair acts on; this is the standard RViz 6-DOF control set.
        axes = (
            ("x", 1.0, 0.0, 0.0),
            ("z", 0.0, 1.0, 0.0),
            ("y", 0.0, 0.0, 1.0),
        )
        for name, qx, qy, qz in axes:
            for mode in (
                InteractiveMarkerControl.ROTATE_AXIS,
                InteractiveMarkerControl.MOVE_AXIS,
            ):
                control = InteractiveMarkerControl()
                control.orientation.w = 1.0
                control.orientation.x = qx
                control.orientation.y = qy
                control.orientation.z = qz
                control.name = (
                    f"{'rotate' if mode == InteractiveMarkerControl.ROTATE_AXIS else 'move'}_{name}"
                )
                control.interaction_mode = mode
                # Without this the handles rotate with the marker and become
                # awkward to grab once the goal is tilted.
                control.orientation_mode = InteractiveMarkerControl.FIXED
                marker.controls.append(control)

        self._server.insert(marker, feedback_callback=self._on_feedback)
        self._menu.apply(self._server, marker.name)
        self._server.applyChanges()

    # ------------------------------------------------------------------ #
    def _publish_pose(self) -> None:
        pose = self._marker_pose()
        if pose is None:
            return
        message = PoseStamped()
        message.header.frame_id = self.base_frame
        message.header.stamp = self.get_clock().now().to_msg()
        message.pose = pose
        self._pose_publisher.publish(message)

    def _on_go_service(self, _request, response):
        """Move to wherever the marker currently sits."""
        pose = self._marker_pose()
        if pose is None:
            response.success = False
            response.message = "no marker pose available"
            return response
        if self._busy.locked():
            response.success = False
            response.message = "a motion is already running"
            return response
        self._start_move(pose)
        response.success = True
        response.message = (
            f"moving to ({pose.position.x:.3f}, {pose.position.y:.3f}, "
            f"{pose.position.z:.3f})"
        )
        return response

    def _publish_status(self, text: str) -> None:
        self._status.publish(String(data=text))
        self.get_logger().info(text)

    def _marker_pose(self) -> Pose | None:
        marker = self._server.get("goal")
        return marker.pose if marker is not None else None

    def _on_feedback(self, feedback: InteractiveMarkerFeedback) -> None:
        if feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP and self.auto_go:
            self._start_move(feedback.pose)

    def _on_move_here(self, feedback: InteractiveMarkerFeedback) -> None:
        # Every feedback event carries the marker's current pose, so use that
        # rather than the server's stored copy: the stored pose only tracks
        # drags that produced a POSE_UPDATE, which makes a menu click act on a
        # stale position.
        pose = feedback.pose if feedback.marker_name == "goal" else None
        if pose is None:
            pose = self._marker_pose()
        if pose is not None:
            # Keep the server in step with what we just acted on.
            self._server.setPose("goal", pose)
            self._server.applyChanges()
            self._start_move(pose)

    def _on_reset_marker(self, _feedback: InteractiveMarkerFeedback) -> None:
        pose = self._current_tcp_pose()
        self._server.setPose("goal", pose)
        self._server.applyChanges()
        self._publish_status("marker reset to the current tool position")

    def _on_toggle_auto(self, _feedback: InteractiveMarkerFeedback) -> None:
        self.auto_go = not self.auto_go
        self._menu.setCheckState(
            self._auto_entry,
            MenuHandler.CHECKED if self.auto_go else MenuHandler.UNCHECKED,
        )
        self._menu.reApply(self._server)
        self._server.applyChanges()
        self._publish_status(f"auto-go {'enabled' if self.auto_go else 'disabled'}")

    def _on_open(self, _feedback: InteractiveMarkerFeedback) -> None:
        self._run_async(lambda: self._moveit.open_gripper(), "open gripper")

    def _on_close(self, _feedback: InteractiveMarkerFeedback) -> None:
        self._run_async(lambda: self._moveit.close_gripper(), "close gripper")

    # ------------------------------------------------------------------ #
    def _start_move(self, pose: Pose) -> None:
        target = Pose()
        target.position.x = pose.position.x
        target.position.y = pose.position.y
        target.position.z = pose.position.z
        target.orientation = pose.orientation
        self._run_async(
            lambda: self._moveit.move_to_pose(target),
            f"move to ({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f})",
        )

    def _run_async(self, action, label: str) -> None:
        """Run a blocking MoveIt call off the executor thread.

        Refuses rather than queues: clicking "Move here" twice should not
        commit the arm to two motions in sequence.
        """
        if not self._busy.acquire(blocking=False):
            self._publish_status(f"busy, ignoring: {label}")
            return

        def worker() -> None:
            try:
                self._publish_status(f"{label} ...")
                action()
                self._publish_status(f"{label}: done")
            except MoveItError as exc:
                self._publish_status(f"{label}: FAILED - {exc}")
            except Exception as exc:  # noqa: BLE001 - never kill the thread
                self._publish_status(f"{label}: error - {type(exc).__name__}: {exc}")
            finally:
                self._busy.release()

        threading.Thread(target=worker, daemon=True, name="goal_marker_move").start()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = GoalMarkerNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._moveit_node.destroy_node()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
