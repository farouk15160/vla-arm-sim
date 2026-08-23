"""Reset the tabletop between evaluation rollouts.

Sends the arm home and teleports the cubes back to their start poses via
Gazebo's ``/world/<world>/set_pose`` service, so a batch of policy rollouts all
begin from the same state.

    ros2 run groot_vla scene_reset                 # cubes + arm
    ros2 run groot_vla scene_reset --no-arm        # cubes only
    ros2 run groot_vla scene_reset --randomize     # jitter cube positions
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import numpy as np

# Must match worlds/tabletop.sdf.
WORLD = "tabletop"
CUBE_POSES = {
    "red_cube": (0.45, 0.16, 0.62),
    "green_cube": (0.55, -0.05, 0.62),
    "blue_cube": (0.38, -0.18, 0.62),
}
HOME_JOINTS = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.5708,
    "elbow_joint": 1.5708,
    "wrist_1_joint": -1.5708,
    "wrist_2_joint": -1.5708,
    "wrist_3_joint": 0.0,
}


def set_model_pose(name: str, xyz: tuple[float, float, float], world: str = WORLD) -> bool:
    """Teleport a model. Uses the gz CLI so this stays a plain script.

    Note that set_pose does not zero the body's velocity, so a cube that was
    mid-fall keeps its momentum; pausing the sim first gives a cleaner reset.
    """
    request = (
        f'name: "{name}", position: {{x: {xyz[0]}, y: {xyz[1]}, z: {xyz[2]}}}, '
        f"orientation: {{x: 0, y: 0, z: 0, w: 1}}"
    )
    command = [
        "gz", "service", "-s", f"/world/{world}/set_pose",
        "--reqtype", "gz.msgs.Pose",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "2000",
        "--req", request,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  {name}: FAILED {result.stderr.strip() or result.stdout.strip()}",
              file=sys.stderr)
        return False
    print(f"  {name} -> ({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})")
    return True


def read_model_poses(world: str = WORLD, timeout: float = 3.0) -> dict[str, tuple[float, float, float]]:
    """Read every model's live pose out of Gazebo.

    Needed after --randomize: the cubes are no longer where CUBE_POSES says,
    and a demonstration collected against the nominal position would grasp thin
    air. Parsed from the CLI rather than a ROS subscription because
    /world/<name>/dynamic_pose/info is not bridged by default.
    """
    try:
        result = subprocess.run(
            ["gz", "topic", "-e", "-t", f"/world/{world}/dynamic_pose/info", "-n", "1"],
            capture_output=True, text=True, timeout=timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {}
    if result.returncode != 0:
        return {}

    poses: dict[str, tuple[float, float, float]] = {}
    name = None
    pending: dict[str, list[float]] = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith('name: "'):
            name = line.split('"')[1]
        elif name and line[:3] in ("x: ", "y: ", "z: "):
            pending.setdefault(name, []).append(float(line[3:]))
    for key, values in pending.items():
        if len(values) >= 3:
            poses[key] = (values[0], values[1], values[2])
    return poses


def send_arm_home(duration: float = 4.0) -> bool:
    """Publish a single trajectory point straight at the arm controller."""
    positions = ", ".join(str(HOME_JOINTS[j]) for j in HOME_JOINTS)
    names = ", ".join(f'"{j}"' for j in HOME_JOINTS)
    message = (
        f"{{joint_names: [{names}], "
        f"points: [{{positions: [{positions}], "
        f"time_from_start: {{sec: {int(duration)}, nanosec: 0}}}}]}}"
    )
    command = [
        "ros2", "topic", "pub", "--once",
        "/arm_controller/joint_trajectory",
        "trajectory_msgs/msg/JointTrajectory",
        message,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  arm: FAILED {result.stderr.strip()}", file=sys.stderr)
        return False
    print("  arm -> home")
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--world", default=WORLD)
    parser.add_argument("--no-arm", action="store_true", help="leave the arm where it is")
    parser.add_argument("--no-cubes", action="store_true", help="leave the cubes where they are")
    parser.add_argument("--randomize", action="store_true",
                        help="jitter cube x/y, for evaluating generalisation")
    parser.add_argument("--jitter", type=float, default=0.05, help="max jitter [m]")
    parser.add_argument("--seed", type=int, default=None)
    args, _unknown = parser.parse_known_args(argv)

    rng = np.random.default_rng(args.seed)
    failures = 0

    if not args.no_cubes:
        print("resetting objects:")
        for name, xyz in CUBE_POSES.items():
            target = xyz
            if args.randomize:
                offset = rng.uniform(-args.jitter, args.jitter, size=2)
                target = (xyz[0] + offset[0], xyz[1] + offset[1], xyz[2])
            failures += not set_model_pose(name, target, args.world)

    if not args.no_arm:
        print("resetting arm:")
        failures += not send_arm_home()

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
