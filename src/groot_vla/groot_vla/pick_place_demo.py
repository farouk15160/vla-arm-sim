"""Scripted MoveIt pick-and-place: the baseline the VLA is measured against.

Run it with the simulation and MoveIt already up:

    ros2 launch groot_arm_bringup demo.launch.py
    ros2 run groot_vla pick_place_demo --ros-args -p cube:=red_cube

Cube positions are read from the world file's defaults; pass ``-p
object_xyz:="[0.45, 0.16, 0.62]"`` to target something else. Positions are in
the *world* frame and converted to the arm's base_link here, since base_link
sits on top of the 0.6 m pedestal.
"""

from __future__ import annotations

import math
import sys

import rclpy
from rclpy.node import Node

from groot_vla.moveit_helper import MoveItError, MoveItHelper, make_pose

# Poses in worlds/tabletop.sdf, world frame.
CUBES = {
    "red_cube": (0.45, 0.16, 0.62),
    "green_cube": (0.55, -0.05, 0.62),
    "blue_cube": (0.38, -0.18, 0.62),
}
TRAY_XYZ = (0.5, -0.35, 0.62)
PEDESTAL_HEIGHT = 0.6  # world z of base_link


def world_to_base(xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    """base_link is directly above the world origin, on the pedestal."""
    return (xyz[0], xyz[1], xyz[2] - PEDESTAL_HEIGHT)


def pick_place_sequence(
    moveit: MoveItHelper,
    object_world: tuple[float, float, float],
    tray_world: tuple[float, float, float] = TRAY_XYZ,
    approach_height: float = 0.12,
    grasp_clearance: float = 0.012,
    on_step=None,
) -> None:
    """Run the full pick-and-place. Raises MoveItError on any failed step.

    Extracted so demonstration collection (collect_demos) drives exactly the
    same motion the standalone demo does - a dataset recorded from a different
    code path would be quietly training on different behaviour.
    """
    x, y, z = world_to_base(object_world)
    yaw = math.atan2(y, x)
    grasp_z = z + grasp_clearance
    approach_z = grasp_z + approach_height

    tray_x, tray_y, tray_z = world_to_base(tray_world)
    place_z = tray_z + 0.04
    tray_yaw = math.atan2(tray_y, tray_x)

    steps = (
        ("open gripper", lambda: moveit.open_gripper()),
        ("move above cube", lambda: moveit.move_to_pose(make_pose(x, y, approach_z, yaw=yaw))),
        ("descend to grasp", lambda: moveit.cartesian_move([make_pose(x, y, grasp_z, yaw=yaw)])),
        ("close gripper", lambda: moveit.close_gripper()),
        ("lift", lambda: moveit.cartesian_move([make_pose(x, y, approach_z, yaw=yaw)])),
        ("move above tray", lambda: moveit.move_to_pose(
            make_pose(tray_x, tray_y, approach_z, yaw=tray_yaw))),
        ("lower into tray", lambda: moveit.cartesian_move(
            [make_pose(tray_x, tray_y, place_z, yaw=tray_yaw)])),
        ("release", lambda: moveit.open_gripper()),
        ("retreat", lambda: moveit.cartesian_move(
            [make_pose(tray_x, tray_y, approach_z, yaw=tray_yaw)])),
        ("home", lambda: moveit.move_to_joints(HOME_JOINTS)),
    )
    for index, (label, action) in enumerate(steps, start=1):
        if on_step:
            on_step(index, len(steps), label)
        action()


HOME_JOINTS = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.5708,
    "elbow_joint": 1.5708,
    "wrist_1_joint": -1.5708,
    "wrist_2_joint": -1.5708,
    "wrist_3_joint": 0.0,
}


class PickPlaceDemo(Node):
    def __init__(self) -> None:
        super().__init__("pick_place_demo")
        self.declare_parameter("cube", "red_cube")
        self.declare_parameter("object_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("approach_height", 0.12)
        # TCP offset ABOVE the cube centre at grasp time. tcp_link sits at the
        # middle of the fingers, which are longer than the cube is tall, so
        # centring exactly would push the fingertips into the table.
        self.declare_parameter("grasp_clearance", 0.012)
        # use_sim_time is declared by rclpy itself; redeclaring it raises.

        cube = str(self.get_parameter("cube").value)
        override = [float(v) for v in self.get_parameter("object_xyz").value]
        if any(override):
            self.object_world = tuple(override)
        elif cube in CUBES:
            self.object_world = CUBES[cube]
        else:
            raise SystemExit(f"unknown cube {cube!r}; known: {sorted(CUBES)}")

        self.approach_height = float(self.get_parameter("approach_height").value)
        self.grasp_clearance = float(self.get_parameter("grasp_clearance").value)
        self.moveit = MoveItHelper(self)

    def run(self) -> int:
        self.get_logger().info("waiting for move_group and controllers...")
        self.moveit.wait_for_services()
        self.get_logger().info("registering table geometry with the planner...")
        self.moveit.add_default_scene()

        def report(index: int, total: int, label: str) -> None:
            self.get_logger().info(f"[{index}/{total}] {label}")

        try:
            pick_place_sequence(
                self.moveit,
                self.object_world,
                approach_height=self.approach_height,
                grasp_clearance=self.grasp_clearance,
                on_step=report,
            )
        except MoveItError as exc:
            self.get_logger().error(f"pick and place failed: {exc}")
            return 1
        self.get_logger().info("pick and place complete")
        return 0


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PickPlaceDemo()
    try:
        code = node.run()
    except KeyboardInterrupt:
        code = 130
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()
