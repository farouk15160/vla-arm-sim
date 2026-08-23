"""Make RViz show the same world Gazebo is simulating.

The table, cubes and tray live in worlds/tabletop.sdf, which only Gazebo reads.
RViz renders the robot description and the planning scene, so without this node
it shows a robot floating in an empty grid while Gazebo shows a furnished
table - the two views disagree, which makes it impossible to judge what the
policy is looking at.

This node publishes two things from one source of truth:

  * a MarkerArray on /world_markers  - what RViz draws
  * MoveIt collision objects         - what the planner avoids

Cube poses are read live from Gazebo's /world/<world>/dynamic_pose/info topic
when the ros_gz bridge provides it, so cubes that the robot moves are followed
rather than drawn at their start positions. Static geometry is drawn from the
table of constants below, which mirrors the SDF.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass, field

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile
from shape_msgs.msg import SolidPrimitive
from visualization_msgs.msg import Marker, MarkerArray

# Height of base_link above the world origin, from groot_arm.urdf.xacro.
PEDESTAL_HEIGHT = 0.6


@dataclass
class WorldBox:
    """A box in worlds/tabletop.sdf, described in WORLD coordinates."""

    name: str
    size: tuple[float, float, float]
    position: tuple[float, float, float]
    colour: tuple[float, float, float, float]
    # Static geometry is registered with the planner; movable objects are not,
    # because a cube the gripper is about to close on must not read as a
    # collision. They are still drawn.
    collision: bool = True
    dynamic: bool = False


WORLD_BOXES: list[WorldBox] = [
    WorldBox("table_top", (1.0, 1.4, 0.05), (0.45, 0.0, 0.575), (0.65, 0.50, 0.35, 1.0)),
    WorldBox("table_base", (0.85, 1.25, 0.55), (0.45, 0.0, 0.275), (0.30, 0.30, 0.32, 1.0)),
    WorldBox("tray", (0.22, 0.22, 0.01), (0.50, -0.35, 0.605), (0.25, 0.25, 0.30, 1.0)),
    WorldBox("red_cube", (0.04, 0.04, 0.04), (0.45, 0.16, 0.62),
             (0.90, 0.10, 0.10, 1.0), collision=False, dynamic=True),
    WorldBox("green_cube", (0.04, 0.04, 0.04), (0.55, -0.05, 0.62),
             (0.10, 0.80, 0.10, 1.0), collision=False, dynamic=True),
    WorldBox("blue_cube", (0.04, 0.04, 0.04), (0.38, -0.18, 0.62),
             (0.10, 0.20, 0.90, 1.0), collision=False, dynamic=True),
]


def world_to_base(xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    """World coordinates -> base_link, which sits on top of the pedestal."""
    return (xyz[0], xyz[1], xyz[2] - PEDESTAL_HEIGHT)


class WorldPublisher(Node):
    def __init__(self) -> None:
        super().__init__("world_publisher")

        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("world_name", "tabletop")
        self.declare_parameter("publish_rate", 5.0)
        self.declare_parameter("track_dynamic_objects", True)
        self.declare_parameter("register_collision_objects", True)
        # The planner's table is narrower than the real one so it does not
        # engulf the arm pedestal; see MoveItHelper.add_default_scene.
        self.declare_parameter("collision_table_x_min", 0.12)

        self.frame_id = str(self.get_parameter("frame_id").value)
        self.world_name = str(self.get_parameter("world_name").value)
        self.track_dynamic = bool(self.get_parameter("track_dynamic_objects").value)

        latching = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self._markers = self.create_publisher(MarkerArray, "/world_markers", latching)
        self._scene = self.create_client(ApplyPlanningScene, "/apply_planning_scene")

        self._live_poses: dict[str, tuple[float, float, float]] = {}
        rate = float(self.get_parameter("publish_rate").value)
        self.create_timer(1.0 / max(rate, 0.1), self._publish_markers)
        if self.track_dynamic:
            # Polling the CLI is cheap at 2 Hz and avoids requiring the pose
            # topic to be bridged, which it is not by default.
            self.create_timer(0.5, self._poll_gazebo_poses)

        if bool(self.get_parameter("register_collision_objects").value):
            self.create_timer(2.0, self._register_collision_once)
        self._collision_registered = False

        self.get_logger().info(
            f"world_publisher up: {len(WORLD_BOXES)} objects -> /world_markers "
            f"in frame {self.frame_id!r}"
        )

    # ------------------------------------------------------------------ #
    def _poll_gazebo_poses(self) -> None:
        """Read live cube poses out of Gazebo.

        Uses `gz topic -e` rather than a ROS subscription because
        /world/<name>/dynamic_pose/info is not bridged by default and adding it
        to the bridge would cost bandwidth for every link in the scene.
        """
        try:
            result = subprocess.run(
                ["gz", "topic", "-e", "-t", f"/world/{self.world_name}/dynamic_pose/info", "-n", "1"],
                capture_output=True, text=True, timeout=2.0,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return
        if result.returncode != 0:
            return

        name = None
        pending: dict[str, list[float]] = {}
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith('name: "'):
                name = line.split('"')[1]
            elif name and line.startswith("x: "):
                pending.setdefault(name, []).append(float(line[3:]))
            elif name and line.startswith("y: "):
                pending.setdefault(name, []).append(float(line[3:]))
            elif name and line.startswith("z: "):
                pending.setdefault(name, []).append(float(line[3:]))
        for key, values in pending.items():
            if len(values) >= 3:
                self._live_poses[key] = (values[0], values[1], values[2])

    def _pose_of(self, box: WorldBox) -> tuple[float, float, float]:
        if box.dynamic and box.name in self._live_poses:
            return self._live_poses[box.name]
        return box.position

    # ------------------------------------------------------------------ #
    def _publish_markers(self) -> None:
        array = MarkerArray()
        for index, box in enumerate(WORLD_BOXES):
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "world"
            marker.id = index
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            x, y, z = world_to_base(self._pose_of(box))
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = z
            marker.pose.orientation.w = 1.0

            marker.scale.x, marker.scale.y, marker.scale.z = box.size
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = box.colour
            array.markers.append(marker)
        self._markers.publish(array)

    # ------------------------------------------------------------------ #
    def _register_collision_once(self) -> None:
        """Add the static geometry to the planning scene, once."""
        if self._collision_registered:
            return
        if not self._scene.service_is_ready():
            return  # move_group not up yet; the timer will retry

        x_min = float(self.get_parameter("collision_table_x_min").value)
        scene = PlanningScene()
        scene.is_diff = True

        for box in WORLD_BOXES:
            if not box.collision:
                continue
            size = list(box.size)
            position = list(world_to_base(box.position))
            if box.name.startswith("table"):
                # Trim the -x side so the box does not swallow the pedestal,
                # which would put the robot permanently in self-collision.
                x_max = box.position[0] + size[0] / 2.0
                size[0] = x_max - x_min
                position[0] = (x_min + x_max) / 2.0

            collision_object = CollisionObject()
            collision_object.id = box.name
            collision_object.header.frame_id = self.frame_id
            collision_object.operation = CollisionObject.ADD

            primitive = SolidPrimitive()
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = [float(v) for v in size]
            collision_object.primitives.append(primitive)

            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = position
            pose.orientation.w = 1.0
            collision_object.primitive_poses.append(pose)
            scene.world.collision_objects.append(collision_object)

        request = ApplyPlanningScene.Request()
        request.scene = scene
        future = self._scene.call_async(request)
        future.add_done_callback(self._on_scene_applied)
        self._collision_registered = True

    def _on_scene_applied(self, future) -> None:
        try:
            if future.result() is not None and future.result().success:
                self.get_logger().info("static world geometry registered with move_group")
                return
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"planning scene update failed: {exc}")
            self._collision_registered = False  # allow a retry
            return
        self.get_logger().warn("move_group rejected the planning scene update")
        self._collision_registered = False


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = WorldPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
