"""Turn ROS messages into the observation dict a GR00T policy expects.

GR00T's observation layout changed between releases:

  nested (N1.7)   {"video": {"ego_view": arr}, "state": {"single_arm": arr},
                   "language": {"task": [["pick up the cube"]]}}

  flat (N1.5)     {"video.ego_view": arr, "state.single_arm": arr,
                   "annotation.human.task_description": ["pick up the cube"]}

Both are produced here, selected by ``schema``, because which one a given
checkpoint wants depends on the version it was exported with.  Query the
server with ``get_modality_config`` if you are unsure.

Array conventions (from the GR00T policy API):
    video  (B, T, H, W, 3) uint8, RGB
    state  (B, T, D)       float32, physical units
with B = 1 and T = 1 for single-frame closed-loop control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sensor_msgs.msg import Image

# cv2 ships with the ROS desktop install; the numpy path is a fallback only.
try:
    import cv2

    _HAVE_CV2 = True
except ImportError:  # pragma: no cover
    _HAVE_CV2 = False


class ImageDecodeError(ValueError):
    """A sensor_msgs/Image arrived in an encoding we cannot handle."""


def image_to_rgb(msg: Image) -> np.ndarray:
    """Decode sensor_msgs/Image into an (H, W, 3) uint8 RGB array.

    Done by hand rather than with cv_bridge: cv_bridge is a compiled dependency
    that regularly mismatches the numpy in a pip-installed environment, and the
    encodings Gazebo emits are a short list.
    """
    encoding = msg.encoding.lower()
    buffer = np.frombuffer(msg.data, dtype=np.uint8)

    if encoding in ("rgb8", "bgr8"):
        channels = 3
    elif encoding in ("rgba8", "bgra8"):
        channels = 4
    elif encoding == "mono8":
        channels = 1
    else:
        raise ImageDecodeError(
            f"unsupported image encoding {msg.encoding!r}; "
            "expected one of rgb8, bgr8, rgba8, bgra8, mono8"
        )

    # step is the row stride in bytes and may exceed width*channels (padding).
    expected = msg.step * msg.height
    if buffer.size < expected:
        raise ImageDecodeError(
            f"image buffer too small: got {buffer.size} bytes, need {expected}"
        )
    frame = buffer[:expected].reshape(msg.height, msg.step)
    frame = frame[:, : msg.width * channels].reshape(msg.height, msg.width, channels)

    if encoding == "mono8":
        return np.repeat(frame, 3, axis=2)
    if encoding in ("bgr8", "bgra8"):
        frame = frame[:, :, ::-1] if channels == 3 else frame[:, :, [2, 1, 0]]
        return np.ascontiguousarray(frame)
    return np.ascontiguousarray(frame[:, :, :3])


def resize_rgb(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize to (height, width) with area interpolation when downscaling."""
    if image.shape[0] == height and image.shape[1] == width:
        return image
    if _HAVE_CV2:
        shrinking = image.shape[0] > height or image.shape[1] > width
        interp = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
        return cv2.resize(image, (width, height), interpolation=interp)
    # Nearest-neighbour fallback. Adequate to keep the pipeline running, but
    # it is not what the model was trained on - install opencv for real runs.
    rows = (np.arange(height) * image.shape[0] // height).clip(0, image.shape[0] - 1)
    cols = (np.arange(width) * image.shape[1] // width).clip(0, image.shape[1] - 1)
    return np.ascontiguousarray(image[rows][:, cols])


@dataclass
class CameraSpec:
    """One camera in the observation: ROS topic -> GR00T modality key."""

    topic: str
    modality_key: str  # e.g. "ego_view" -> video.ego_view
    width: int = 224
    height: int = 224


@dataclass
class ObservationBuilder:
    """Assembles the observation dict handed to :meth:`GrootClient.get_action`.

    ``state_keys`` maps a GR00T state modality name onto the joints that feed
    it, in order. The default splits the arm from the gripper, which is what
    most single-arm GR00T data configs expect.
    """

    cameras: list[CameraSpec]
    state_keys: dict[str, list[str]] = field(
        default_factory=lambda: {
            "single_arm": [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            "gripper": ["left_finger_joint"],
        }
    )
    schema: str = "nested"  # "nested" (N1.7) | "flat" (N1.5)
    language_key: str = "annotation.human.task_description"  # flat schema only
    # Normalise the finger joint from metres onto [0, 1], the convention most
    # GR00T gripper modalities use. Set to None to pass raw metres.
    gripper_joint_range: tuple[float, float] | None = (0.0, 0.04)

    def build(
        self,
        images: dict[str, np.ndarray],
        joint_positions: dict[str, float],
        instruction: str,
    ) -> dict[str, Any]:
        """
        :param images: modality_key -> (H, W, 3) uint8 RGB, already resized
        :param joint_positions: joint name -> position (rad, or m for fingers)
        :param instruction: the natural-language task
        """
        # Emit cameras in the order declared in `cameras`, NOT dict order.
        # The images dict is populated by subscription callbacks, so its order
        # depends on which camera happened to publish first - and a server that
        # maps our views onto its own image slots positionally would then swap
        # wrist and scene between runs, silently breaking the correspondence
        # with what the policy was trained on.
        video = {
            spec.modality_key: images[spec.modality_key][np.newaxis, np.newaxis, ...]
            for spec in self.cameras
            if spec.modality_key in images
        }

        state: dict[str, np.ndarray] = {}
        for modality, joints in self.state_keys.items():
            values = []
            for joint in joints:
                if joint not in joint_positions:
                    raise KeyError(
                        f"joint {joint!r} needed by state modality {modality!r} "
                        f"is absent from /joint_states"
                    )
                value = joint_positions[joint]
                if modality == "gripper" and self.gripper_joint_range is not None:
                    low, high = self.gripper_joint_range
                    value = (value - low) / (high - low) if high > low else 0.0
                    value = float(np.clip(value, 0.0, 1.0))
                values.append(value)
            state[modality] = np.asarray([[values]], dtype=np.float32)

        if self.schema == "nested":
            return {
                "video": video,
                "state": state,
                "language": {"task": [[instruction]]},
            }
        if self.schema == "flat":
            observation: dict[str, Any] = {}
            observation.update({f"video.{k}": v for k, v in video.items()})
            observation.update({f"state.{k}": v for k, v in state.items()})
            observation[self.language_key] = [instruction]
            return observation
        raise ValueError(f"unknown observation schema {self.schema!r}")

    def state_dimension(self) -> int:
        return sum(len(joints) for joints in self.state_keys.values())
