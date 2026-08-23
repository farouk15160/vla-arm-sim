"""Collect a demonstration dataset by running the scripted pick-and-place.

Zero-shot VLAs flail on this cell because a UR5e with this gripper is an unseen
embodiment. Fine-tuning fixes that, and fine-tuning needs demonstrations. The
scripted pick-and-place already succeeds reliably, so the demonstrations can be
generated rather than teleoperated.

Each episode:
    1. reset the scene, optionally jittering the cube positions
    2. read where the cubes ACTUALLY ended up (randomisation moves them)
    3. start recording
    4. run the same pick_place_sequence the standalone demo uses
    5. stop recording, or discard the episode if any step failed

A failed episode is discarded rather than kept: a dataset containing failures
teaches the policy to fail. Randomisation is what stops the policy simply
memorising one trajectory - without it, every episode is identical and the
model learns a constant.

    # 40 episodes over all three cubes
    ros2 run groot_vla collect_demos --ros-args -p episodes:=40

Then convert and train:
    ros2 run groot_vla export_lerobot --input ~/groot_episodes --output ~/groot_lerobot
    ~/vla_venv/bin/python -m lerobot.scripts.lerobot_train \\
        --policy.path=lerobot/smolvla_base --dataset.root=~/groot_lerobot ...
"""

from __future__ import annotations

import sys
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from groot_vla.moveit_helper import MoveItError, MoveItHelper
from groot_vla.pick_place_demo import HOME_JOINTS, pick_place_sequence
from groot_vla.scene_reset import CUBE_POSES, read_model_poses, set_model_pose

TASK_TEMPLATE = "pick up the {colour} cube and place it in the tray"


class DemoCollector(Node):
    def __init__(self) -> None:
        super().__init__("collect_demos")

        self.declare_parameter("episodes", 30)
        self.declare_parameter("cubes", ["red_cube", "green_cube", "blue_cube"])
        self.declare_parameter("world", "tabletop")
        self.declare_parameter("randomize", True)
        # Keep the jitter inside the arm's comfortable reach; a cube placed
        # where IK fails just wastes an episode.
        self.declare_parameter("jitter", 0.06)
        self.declare_parameter("seed", 0)
        self.declare_parameter("settle_time", 1.5)

        self.episodes = int(self.get_parameter("episodes").value)
        self.cubes = [str(c) for c in self.get_parameter("cubes").value]
        self.world = str(self.get_parameter("world").value)
        self.randomize = bool(self.get_parameter("randomize").value)
        self.jitter = float(self.get_parameter("jitter").value)
        self.settle_time = float(self.get_parameter("settle_time").value)
        self._rng = np.random.default_rng(int(self.get_parameter("seed").value))

        self.moveit_node = rclpy.create_node("collect_demos_moveit")
        self.moveit = MoveItHelper(self.moveit_node)

        self._start = self.create_client(Trigger, "/episode_recorder/start_episode")
        self._stop = self.create_client(Trigger, "/episode_recorder/stop_episode")
        self._discard = self.create_client(Trigger, "/episode_recorder/discard_episode")
        self._task_pub = self.create_publisher(String, "/episode_recorder/task", 10)

    # ------------------------------------------------------------------ #
    def _call(self, client, label: str, timeout: float = 20.0) -> bool:
        if not client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(f"{label}: service unavailable")
            return False
        future = client.call_async(Trigger.Request())
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not future.done():
            self.get_logger().error(f"{label}: timed out")
            return False
        response = future.result()
        if not response.success:
            self.get_logger().warn(f"{label}: {response.message}")
        return bool(response.success)

    def _reset_scene(self) -> dict[str, tuple[float, float, float]]:
        for name, nominal in CUBE_POSES.items():
            target = nominal
            if self.randomize:
                offset = self._rng.uniform(-self.jitter, self.jitter, size=2)
                target = (nominal[0] + offset[0], nominal[1] + offset[1], nominal[2])
            set_model_pose(name, target, self.world)
        # Let the physics settle before reading poses back, or a cube still
        # falling is recorded at the wrong height.
        time.sleep(self.settle_time)
        return read_model_poses(self.world)

    # ------------------------------------------------------------------ #
    def run(self) -> int:
        self.get_logger().info("waiting for move_group ...")
        self.moveit.wait_for_services()
        self.moveit.add_default_scene()

        if not self._start.wait_for_service(timeout_sec=20.0):
            self.get_logger().error(
                "episode_recorder is not running. Start it first:\n"
                "  ros2 run groot_vla episode_recorder --ros-args "
                "-p output_dir:=~/groot_episodes"
            )
            return 1

        recorded = 0
        failed = 0
        for episode in range(self.episodes):
            cube = self.cubes[episode % len(self.cubes)]
            colour = cube.split("_")[0]
            self.get_logger().info(
                f"--- episode {episode + 1}/{self.episodes}: {cube} ---")

            poses = self._reset_scene()
            target = poses.get(cube, CUBE_POSES[cube])
            self.moveit.move_to_joints(HOME_JOINTS)

            # The recorder stamps this on the episode; it becomes the language
            # instruction the policy is conditioned on.
            self._task_pub.publish(String(data=TASK_TEMPLATE.format(colour=colour)))
            time.sleep(0.3)

            if not self._call(self._start, "start_episode"):
                failed += 1
                continue

            try:
                pick_place_sequence(self.moveit, target)
            except MoveItError as exc:
                self.get_logger().warn(f"episode failed ({exc}); discarding")
                self._call(self._discard, "discard_episode")
                failed += 1
                continue

            if self._call(self._stop, "stop_episode"):
                recorded += 1
            else:
                failed += 1

        self.get_logger().info(
            f"done: {recorded} episodes recorded, {failed} discarded")
        return 0 if recorded else 1


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = DemoCollector()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    import threading

    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        code = node.run()
    except KeyboardInterrupt:
        code = 130
    finally:
        executor.shutdown()
        node.moveit_node.destroy_node()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()
