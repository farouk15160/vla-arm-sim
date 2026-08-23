"""Record demonstration episodes for GR00T fine-tuning.

Zero-shot GR00T on a UR5e + custom gripper is an out-of-distribution
embodiment; realistically you post-train on demonstrations. This node captures
them.

Frames are written as PNG + JSONL rather than straight to LeRobot parquet/mp4,
because pyarrow and ffmpeg are not ROS dependencies and should not be forced on
the runtime. Convert afterwards:

    ros2 run groot_vla export_lerobot --input ~/groot_episodes --output ~/groot_lerobot

Usage:
    ros2 run groot_vla episode_recorder --ros-args -p output_dir:=~/groot_episodes
    ros2 service call /episode_recorder/start_episode std_srvs/srv/Trigger
    #  ... drive the arm (RViz, pick_place_demo, teleop) ...
    ros2 service call /episode_recorder/stop_episode std_srvs/srv/Trigger
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

from groot_vla.action_mapper import ARM_JOINTS
from groot_vla.observation_builder import ImageDecodeError, image_to_rgb, resize_rgb

try:
    import cv2

    _HAVE_CV2 = True
except ImportError:  # pragma: no cover
    from PIL import Image as PILImage

    _HAVE_CV2 = False

SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)
GRIPPER_JOINT = "left_finger_joint"


class EpisodeRecorder(Node):
    def __init__(self) -> None:
        super().__init__("episode_recorder")

        self.declare_parameter("output_dir", "~/groot_episodes")
        self.declare_parameter("task", "pick up the red cube and place it in the tray")
        self.declare_parameter("fps", 20.0)
        self.declare_parameter("image_width", 224)
        self.declare_parameter("image_height", 224)
        self.declare_parameter("cameras", ["wrist_view", "ego_view"])
        self.declare_parameter(
            "camera_topics", ["/wrist_camera/image_raw", "/scene_camera/image_raw"]
        )
        self.declare_parameter("gripper_open_position", 0.04)
        self.declare_parameter("auto_start", False)

        self.output_dir = Path(os.path.expanduser(str(self.get_parameter("output_dir").value)))
        self.task = str(self.get_parameter("task").value)
        self.fps = float(self.get_parameter("fps").value)
        self.width = int(self.get_parameter("image_width").value)
        self.height = int(self.get_parameter("image_height").value)
        self.gripper_open = float(self.get_parameter("gripper_open_position").value)

        names = [str(n) for n in self.get_parameter("cameras").value]
        topics = [str(t) for t in self.get_parameter("camera_topics").value]
        if len(names) != len(topics):
            raise ValueError("`cameras` and `camera_topics` must be the same length")
        self.camera_names = names

        self._lock = threading.Lock()
        self._images: dict[str, np.ndarray] = {}
        self._joints: dict[str, float] = {}
        self._recording = False
        self._frames: list[dict] = []
        self._episode_dir: Path | None = None
        self._episode_index = self._next_episode_index()
        self._frame_index = 0

        for name, topic in zip(names, topics):
            self.create_subscription(
                Image, topic, lambda msg, key=name: self._on_image(msg, key), SENSOR_QOS
            )
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)
        self.create_subscription(String, "~/task", self._on_task, 10)

        self.create_service(Trigger, "~/start_episode", self._on_start)
        self.create_service(Trigger, "~/stop_episode", self._on_stop)
        self.create_service(Trigger, "~/discard_episode", self._on_discard)

        self.create_timer(1.0 / self.fps, self._on_tick)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.get_logger().info(
            f"recorder ready. output={self.output_dir} fps={self.fps} "
            f"cameras={names} next_episode={self._episode_index}"
        )
        if bool(self.get_parameter("auto_start").value):
            self._start()

    # ------------------------------------------------------------------ #
    def _next_episode_index(self) -> int:
        """Continue numbering rather than overwriting an existing dataset."""
        if not self.output_dir.exists():
            return 0
        existing = [
            int(p.name.split("_")[-1])
            for p in self.output_dir.glob("episode_*")
            if p.is_dir() and p.name.split("_")[-1].isdigit()
        ]
        return max(existing) + 1 if existing else 0

    def _on_image(self, msg: Image, key: str) -> None:
        try:
            rgb = resize_rgb(image_to_rgb(msg), self.width, self.height)
        except ImageDecodeError as exc:
            self.get_logger().warn(f"{key}: {exc}", throttle_duration_sec=5.0)
            return
        with self._lock:
            self._images[key] = rgb

    def _on_joint_state(self, msg: JointState) -> None:
        with self._lock:
            for name, position in zip(msg.name, msg.position):
                self._joints[name] = float(position)

    def _on_task(self, msg: String) -> None:
        if msg.data.strip():
            self.task = msg.data.strip()
            self.get_logger().info(f"task -> {self.task!r}")

    # ------------------------------------------------------------------ #
    def _on_start(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        if self._recording:
            res.success, res.message = False, "already recording"
            return res
        self._start()
        res.success, res.message = True, f"recording episode {self._episode_index}"
        return res

    def _start(self) -> None:
        self._episode_dir = self.output_dir / f"episode_{self._episode_index:06d}"
        for name in self.camera_names:
            (self._episode_dir / name).mkdir(parents=True, exist_ok=True)
        self._frames = []
        self._frame_index = 0
        self._recording = True
        self.get_logger().info(f"recording -> {self._episode_dir}")

    def _on_stop(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        if not self._recording:
            res.success, res.message = False, "not recording"
            return res
        self._recording = False
        count = self._write_episode()
        res.success = count > 0
        res.message = (
            f"wrote {count} frames to {self._episode_dir}"
            if count
            else "episode had no frames; nothing written"
        )
        self.get_logger().info(res.message)
        self._episode_index += 1
        return res

    def _on_discard(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        import shutil

        self._recording = False
        if self._episode_dir and self._episode_dir.exists():
            shutil.rmtree(self._episode_dir)
            res.message = f"discarded {self._episode_dir}"
        else:
            res.message = "nothing to discard"
        self._frames = []
        res.success = True
        self.get_logger().info(res.message)
        return res

    # ------------------------------------------------------------------ #
    def _on_tick(self) -> None:
        if not self._recording or self._episode_dir is None:
            return
        with self._lock:
            images = dict(self._images)
            joints = dict(self._joints)

        if set(images) != set(self.camera_names):
            self.get_logger().warn(
                f"skipping frame, missing camera(s): "
                f"{sorted(set(self.camera_names) - set(images))}",
                throttle_duration_sec=5.0,
            )
            return
        if not all(j in joints for j in ARM_JOINTS) or GRIPPER_JOINT not in joints:
            self.get_logger().warn("skipping frame, incomplete joint state",
                                   throttle_duration_sec=5.0)
            return

        for name, image in images.items():
            path = self._episode_dir / name / f"frame_{self._frame_index:06d}.png"
            _write_png(path, image)

        arm_state = [joints[j] for j in ARM_JOINTS]
        gripper_state = joints[GRIPPER_JOINT] / self.gripper_open if self.gripper_open else 0.0
        self._frames.append(
            {
                "frame_index": self._frame_index,
                "timestamp": self._frame_index / self.fps,
                "state.single_arm": arm_state,
                "state.gripper": [float(np.clip(gripper_state, 0.0, 1.0))],
            }
        )
        self._frame_index += 1

    def _write_episode(self) -> int:
        """Persist the episode, deriving actions from the next state.

        GR00T's default single-arm data config treats the action at step t as
        the state at t+1 (absolute joint targets). The final frame is dropped
        because it has no successor to label it with.
        """
        if self._episode_dir is None or len(self._frames) < 2:
            return 0

        records = []
        for current, following in zip(self._frames, self._frames[1:]):
            record = dict(current)
            record["action.single_arm"] = following["state.single_arm"]
            record["action.gripper"] = following["state.gripper"]
            records.append(record)

        with (self._episode_dir / "frames.jsonl").open("w") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

        metadata = {
            "episode_index": self._episode_index,
            "task": self.task,
            "fps": self.fps,
            "num_frames": len(records),
            "cameras": self.camera_names,
            "image_size": [self.width, self.height],
            "arm_joints": list(ARM_JOINTS),
            "gripper_joint": GRIPPER_JOINT,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (self._episode_dir / "meta.json").write_text(json.dumps(metadata, indent=2))
        return len(records)


def _write_png(path: Path, rgb: np.ndarray) -> None:
    if _HAVE_CV2:
        cv2.imwrite(str(path), rgb[:, :, ::-1])  # cv2 expects BGR
    else:  # pragma: no cover
        PILImage.fromarray(rgb).save(path)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = EpisodeRecorder()
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
