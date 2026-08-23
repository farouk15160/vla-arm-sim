"""Convert a GR00T action chunk into robot commands, safely.

A GR00T policy returns a *chunk*: several future timesteps per inference, e.g.
``{"action.single_arm": (1, 16, 6), "action.gripper": (1, 16, 1)}``.  We
execute the first ``execution_horizon`` steps as a joint trajectory, then throw
the rest away and re-infer (receding horizon).

Three action spaces are supported, because which one a checkpoint emits depends
entirely on the data config it was fine-tuned with:

  joint_position  absolute joint targets in radians          (most common)
  joint_delta     per-step increments in radians
  eef_delta       6-D Cartesian velocity for moveit_servo

Everything is clamped before it leaves this module: a VLA is a neural network,
not a validated controller, and it will occasionally emit garbage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

# Conservative software limits, tighter than the UR5e's +/- 2*pi. They keep the
# arm out of postures that wrap cables or drive the wrist through the table.
DEFAULT_JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "shoulder_pan_joint": (-3.14159, 3.14159),
    "shoulder_lift_joint": (-3.14159, 0.0),
    "elbow_joint": (-2.7, 2.7),
    "wrist_1_joint": (-3.14159, 3.14159),
    "wrist_2_joint": (-3.14159, 3.14159),
    "wrist_3_joint": (-6.28318, 6.28318),
}


class ActionDecodeError(ValueError):
    """The server's reply did not contain the action keys we asked for."""


def flatten_action(action: Any, prefix: str = "") -> dict[str, np.ndarray]:
    """Flatten a possibly-nested action reply into dot-joined keys.

    Handles both ``{"action": {"single_arm": ...}}`` and
    ``{"action.single_arm": ...}`` so the same code works across GR00T
    releases. Byte keys (a msgpack artefact) are decoded to str.
    """
    flat: dict[str, np.ndarray] = {}
    if not isinstance(action, dict):
        return flat
    for key, value in action.items():
        if isinstance(key, bytes):
            key = key.decode()
        full = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_action(value, prefix=f"{full}."))
        else:
            flat[full] = np.asarray(value)
    return flat


def select_action_array(
    flat: dict[str, np.ndarray], key: str
) -> np.ndarray:
    """Look up ``key`` tolerating a missing/extra ``action.`` prefix."""
    for candidate in (key, f"action.{key}", key.removeprefix("action.")):
        if candidate in flat:
            return flat[candidate]
    raise ActionDecodeError(
        f"action key {key!r} not in server reply; available keys: "
        f"{sorted(flat)}"
    )


def as_chunk(array: np.ndarray) -> np.ndarray:
    """Normalise an action array to (T, D).

    Accepts (B, T, D), (T, D) or (D,) - GR00T servers have shipped all three
    depending on whether the batch axis is squeezed server-side.
    """
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 3:
        return array[0]
    if array.ndim == 2:
        return array
    if array.ndim == 1:
        return array[np.newaxis, :]
    raise ActionDecodeError(f"cannot interpret action array of shape {array.shape}")


@dataclass
class ActionMapper:
    """Decodes action chunks and enforces the safety envelope."""

    action_space: str = "joint_position"
    arm_action_key: str = "single_arm"
    gripper_action_key: str = "gripper"
    arm_joints: tuple[str, ...] = ARM_JOINTS
    joint_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_JOINT_LIMITS)
    )
    # Largest joint move permitted between two consecutive chunk steps [rad].
    # At 10 Hz control this caps effective joint speed at ~1.5 rad/s.
    max_joint_step: float = 0.15
    # Gripper action is normalised [0, 1]; map onto finger travel in metres.
    gripper_range: tuple[float, float] = (0.0, 0.04)
    # Below/above these the gripper command is treated as fully closed/open.
    gripper_closed_below: float = 0.0
    # Cartesian velocity caps for eef_delta [m/s, rad/s].
    max_linear_velocity: float = 0.25
    max_angular_velocity: float = 1.0

    def decode(self, action: Any) -> tuple[np.ndarray, np.ndarray]:
        """Split a server reply into (arm_chunk (T, 6), gripper_chunk (T,)).

        A checkpoint without a separate gripper head yields a zero-length
        gripper chunk, which the caller must treat as "leave the gripper".
        """
        flat = flatten_action(action)
        if not flat:
            raise ActionDecodeError(f"empty action reply: {action!r}")

        arm = as_chunk(select_action_array(flat, self.arm_action_key))
        if arm.shape[1] != len(self.arm_joints):
            raise ActionDecodeError(
                f"arm action has {arm.shape[1]} dims, expected "
                f"{len(self.arm_joints)} for joints {self.arm_joints}"
            )

        try:
            gripper = as_chunk(select_action_array(flat, self.gripper_action_key))
            gripper_chunk = gripper[:, 0]
        except ActionDecodeError:
            gripper_chunk = np.empty((0,), dtype=np.float64)

        return arm, gripper_chunk

    # -- joint-space ------------------------------------------------------- #
    def to_joint_targets(
        self, arm_chunk: np.ndarray, current_positions: np.ndarray
    ) -> np.ndarray:
        """Turn an arm chunk into a (T, 6) array of absolute, clamped targets.

        Rate limiting is applied step-to-step starting from the *measured*
        position, so a chunk that jumps far from where the robot actually is
        gets walked toward rather than commanded directly.
        """
        if self.action_space == "joint_position":
            targets = np.asarray(arm_chunk, dtype=np.float64).copy()
        elif self.action_space == "joint_delta":
            targets = current_positions[np.newaxis, :] + np.cumsum(arm_chunk, axis=0)
        else:
            raise ValueError(
                f"to_joint_targets does not apply to action_space "
                f"{self.action_space!r}"
            )

        if not np.all(np.isfinite(targets)):
            raise ActionDecodeError("policy emitted non-finite joint targets")

        previous = np.asarray(current_positions, dtype=np.float64)
        limited = np.empty_like(targets)
        for index, target in enumerate(targets):
            step = np.clip(target - previous, -self.max_joint_step, self.max_joint_step)
            previous = previous + step
            limited[index] = previous

        for column, joint in enumerate(self.arm_joints):
            low, high = self.joint_limits.get(joint, (-np.inf, np.inf))
            limited[:, column] = np.clip(limited[:, column], low, high)
        return limited

    def to_finger_targets(self, gripper_chunk: np.ndarray) -> np.ndarray:
        """Map normalised gripper commands onto finger joint positions [m].

        GR00T gripper heads are not consistent about polarity: some emit
        1.0 = open, some 1.0 = closed. ``gripper_range`` is (closed, open), so
        flip it in config rather than patching this function.
        """
        if gripper_chunk.size == 0:
            return gripper_chunk
        closed, open_ = self.gripper_range
        normalised = np.clip(np.asarray(gripper_chunk, dtype=np.float64), 0.0, 1.0)
        return closed + normalised * (open_ - closed)

    # -- Cartesian --------------------------------------------------------- #
    def to_twist(self, arm_chunk: np.ndarray) -> np.ndarray:
        """First row of an eef_delta chunk as a clamped 6-vector [vx..wz]."""
        if self.action_space != "eef_delta":
            raise ValueError(
                f"to_twist does not apply to action_space {self.action_space!r}"
            )
        if arm_chunk.shape[1] < 6:
            raise ActionDecodeError(
                f"eef_delta needs 6 dims, got {arm_chunk.shape[1]}"
            )
        twist = np.asarray(arm_chunk[0, :6], dtype=np.float64)
        if not np.all(np.isfinite(twist)):
            raise ActionDecodeError("policy emitted non-finite twist")
        twist[:3] = np.clip(twist[:3], -self.max_linear_velocity, self.max_linear_velocity)
        twist[3:] = np.clip(twist[3:], -self.max_angular_velocity, self.max_angular_velocity)
        return twist


def joint_positions_to_array(
    joint_positions: dict[str, float], joints: Iterable[str]
) -> np.ndarray:
    """Extract ``joints`` from a name->position dict, in order."""
    missing = [j for j in joints if j not in joint_positions]
    if missing:
        raise KeyError(f"joints missing from /joint_states: {missing}")
    return np.asarray([joint_positions[j] for j in joints], dtype=np.float64)
