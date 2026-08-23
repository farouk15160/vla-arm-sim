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

        x, y, z = world_to_base(self.object_world)
        grasp_z = z + self.grasp_clearance
        approach_z = grasp_z + self.approach_height
        # Yaw the wrist to align the fingers across the cube's short axis.
        yaw = math.atan2(y, x)

        tray_x, tray_y, tray_z = world_to_base(TRAY_XYZ)
        place_z = tray_z + 0.04
        tray_yaw = math.atan2(tray_y, tray_x)

        steps = (
            ("open gripper", lambda: self.moveit.open_gripper()),
            ("move above cube", lambda: self.moveit.move_to_pose(
                make_pose(x, y, approach_z, yaw=yaw))),
            ("descend to grasp", lambda: self.moveit.cartesian_move(
                [make_pose(x, y, grasp_z, yaw=yaw)])),
            ("close gripper", lambda: self.moveit.close_gripper()),
            ("lift", lambda: self.moveit.cartesian_move(
                [make_pose(x, y, approach_z, yaw=yaw)])),
            ("move above tray", lambda: self.moveit.move_to_pose(
                make_pose(tray_x, tray_y, approach_z, yaw=tray_yaw))),
            ("lower into tray", lambda: self.moveit.cartesian_move(
                [make_pose(tray_x, tray_y, place_z, yaw=tray_yaw)])),
            ("release", lambda: self.moveit.open_gripper()),
            ("retreat", lambda: self.moveit.cartesian_move(
                [make_pose(tray_x, tray_y, approach_z, yaw=tray_yaw)])),
            ("home", lambda: self.moveit.move_to_joints({
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.5708,
                "elbow_joint": 1.5708,
                "wrist_1_joint": -1.5708,
                "wrist_2_joint": -1.5708,
                "wrist_3_joint": 0.0,
            })),
        )

        for index, (label, action) in enumerate(steps, start=1):
            self.get_logger().info(f"[{index}/{len(steps)}] {label}")
            try:
                action()
            except MoveItError as exc:
                self.get_logger().error(f"step {label!r} failed: {exc}")
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
