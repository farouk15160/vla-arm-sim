"""Convert recorded episodes into a LeRobot dataset for fine-tuning.

Reads the PNG + JSONL episodes written by `episode_recorder` and builds a
dataset with LeRobot's own `LeRobotDataset` API rather than writing the layout
by hand.

That choice is deliberate. LeRobot 0.4.x expects dataset format **v3.0**, which
consolidates many episodes per parquet/mp4 file and keeps episode metadata as
parquet with per-episode statistics. Hand-writing it is easy to get subtly
wrong, and the failure mode is a training run that starts and then reads
garbage. Their builder also computes the normalisation statistics, which the
policy needs and which cannot be guessed.

(LeRobot ships a v2.1 -> v3.0 converter, but it resolves the dataset through the
HuggingFace hub and cannot convert a purely local one.)

Run it with the policy venv, which has torch, pandas and lerobot:

    ~/vla_venv/bin/python export_lerobot.py \\
        --input ~/groot_episodes --output ~/groot_lerobot
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

# State is the SIX arm joints only; action is those six plus the gripper.
#
# The asymmetry is deliberate. lerobot/smolvla_base declares a 6-dimensional
# observation.state, and fine-tuning from it keeps that declaration while taking
# the action width from the dataset. Writing a 7-dim state produces a checkpoint
# whose config says state [6] and action [7], which loads fine and then dies at
# inference with "The size of tensor a (6) must match the size of tensor b (7)".
# Six in, seven out matches the base model and still gives the policy gripper
# control, which the base checkpoint does not have.
STATE_DIM = 6
ACTION_DIM = 7


def find_ffmpeg() -> str | None:
    """Locate ffmpeg, preferring the system copy then imageio-ffmpeg's.

    LeRobot encodes the videos itself but needs a binary available. The bundled
    fallback matters because installing ffmpeg system-wide needs root, while the
    venv can pip-install imageio-ffmpeg without it.
    """
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


def read_episode(directory: Path) -> tuple[dict, list[dict]]:
    metadata = json.loads((directory / "meta.json").read_text())
    rows = [
        json.loads(line)
        for line in (directory / "frames.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return metadata, rows


def load_frame(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def convert(input_dir: Path, output_dir: Path, repo_id: str, robot_type: str) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episodes = sorted(p for p in input_dir.glob("episode_*") if p.is_dir())
    if not episodes:
        raise SystemExit(f"no episode_* directories under {input_dir}")

    first_meta, _ = read_episode(episodes[0])
    fps = int(round(float(first_meta["fps"])))
    cameras = list(first_meta["cameras"])
    height, width = int(first_meta["image_size"][1]), int(first_meta["image_size"][0])

    arm_joints = list(first_meta["arm_joints"])
    action_names = [*arm_joints, first_meta["gripper_joint"]]
    features = {
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": arm_joints},
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": action_names},
    }
    for camera in cameras:
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        }

    if output_dir.exists():
        raise SystemExit(
            f"{output_dir} already exists. Remove it or choose another --output; "
            "LeRobot refuses to create a dataset over an existing directory."
        )

    dataset = LeRobotDataset.create(
        repo_id=repo_id, fps=fps, features=features,
        root=output_dir, robot_type=robot_type, use_videos=True,
    )

    total = 0
    for episode in episodes:
        metadata, rows = read_episode(episode)
        if len(rows) < 2:
            print(f"skipping {episode.name}: too few frames", file=sys.stderr)
            continue
        task = metadata["task"]

        for row in rows:
            frame = {
                "observation.state": np.asarray(row["state.single_arm"], dtype=np.float32),
                "action": np.asarray(
                    row["action.single_arm"] + row["action.gripper"], dtype=np.float32),
                "task": task,
            }
            for camera in cameras:
                frame[f"observation.images.{camera}"] = load_frame(
                    episode / camera / f"frame_{row['frame_index']:06d}.png")
            dataset.add_frame(frame)

        dataset.save_episode()
        total += len(rows)
        print(f"converted {episode.name} ({len(rows)} frames)")

    print(f"\ndone: {len(episodes)} episodes, {total} frames -> {output_dir}")
    print(
        "\nfine-tune with:\n"
        "  ~/vla_venv/bin/lerobot-train \\\n"
        "      --policy.path=lerobot/smolvla_base \\\n"
        "      --policy.repo_id=local/smolvla_ur5e --policy.push_to_hub=false \\\n"
        f"      --dataset.repo_id={repo_id} --dataset.root={output_dir} \\\n"
        "      --batch_size=4 --steps=20000 --output_dir=~/smolvla_ur5e"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-id", default="local/groot_ur5e",
                        help="dataset identifier; only a name for a local dataset")
    parser.add_argument("--robot-type", default="ur5e_parallel_gripper")
    args, _unknown = parser.parse_known_args(argv)

    try:
        import lerobot  # noqa: F401
    except ImportError:
        raise SystemExit(
            "lerobot is not importable. Run this with the policy venv:\n"
            "    ~/vla_venv/bin/python export_lerobot.py --input ... --output ...")
    if find_ffmpeg() is None:
        raise SystemExit(
            "No ffmpeg available. Install it system-wide (sudo apt install ffmpeg) "
            "or, without root:\n    ~/vla_venv/bin/pip install imageio-ffmpeg")

    convert(args.input.expanduser(), args.output.expanduser(), args.repo_id, args.robot_type)


if __name__ == "__main__":
    main()
