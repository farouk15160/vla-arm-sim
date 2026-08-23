"""Convert recorded episodes into a LeRobot dataset that GR00T can fine-tune on.

Reads the PNG + JSONL episodes written by ``episode_recorder`` and emits the
LeRobot v2.1 layout Isaac-GR00T expects:

    <out>/meta/info.json
    <out>/meta/tasks.jsonl
    <out>/meta/episodes.jsonl
    <out>/meta/modality.json          <- GR00T-specific, maps columns to modalities
    <out>/data/chunk-000/episode_000000.parquet
    <out>/videos/chunk-000/observation.images.<cam>/episode_000000.mp4

Then:
    python scripts/gr00t_finetune.py \
        --dataset-path <out> --embodiment-tag NEW_EMBODIMENT \
        --data-config <your config>

Extra dependencies, needed only here and not by the ROS runtime:
    pip install pandas pyarrow
    sudo apt install ffmpeg
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

CHUNK = "chunk-000"


def require(module: str, hint: str) -> None:
    try:
        __import__(module)
    except ImportError:
        raise SystemExit(f"missing dependency {module!r}. Install it with: {hint}")


def encode_video(frames_dir: Path, output: Path, fps: float) -> None:
    """PNG sequence -> H.264 mp4. GR00T's video backend decodes yuv420p."""
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found. Install it with: sudo apt install ffmpeg")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%06d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        # yuv420p needs even dimensions; 224x224 is fine but a custom size
        # might not be.
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-crf", "20",
        str(output),
    ]
    subprocess.run(command, check=True)


def convert(input_dir: Path, output_dir: Path, robot_type: str = "ur5e_parallel_gripper") -> None:
    import pandas as pd

    episodes = sorted(p for p in input_dir.glob("episode_*") if p.is_dir())
    if not episodes:
        raise SystemExit(f"no episode_* directories under {input_dir}")

    (output_dir / "meta").mkdir(parents=True, exist_ok=True)
    (output_dir / "data" / CHUNK).mkdir(parents=True, exist_ok=True)

    tasks: dict[str, int] = {}
    episode_records: list[dict] = []
    total_frames = 0
    cameras: list[str] = []
    fps = 20.0

    for new_index, episode in enumerate(episodes):
        metadata = json.loads((episode / "meta.json").read_text())
        fps = float(metadata["fps"])
        cameras = list(metadata["cameras"])
        task = metadata["task"]
        task_index = tasks.setdefault(task, len(tasks))

        rows = [json.loads(line) for line in (episode / "frames.jsonl").read_text().splitlines() if line]
        if not rows:
            print(f"skipping empty episode {episode.name}", file=sys.stderr)
            continue

        frame = pd.DataFrame(
            {
                "observation.state": [
                    r["state.single_arm"] + r["state.gripper"] for r in rows
                ],
                "action": [r["action.single_arm"] + r["action.gripper"] for r in rows],
                "timestamp": [r["timestamp"] for r in rows],
                "frame_index": list(range(len(rows))),
                "episode_index": [new_index] * len(rows),
                "index": list(range(total_frames, total_frames + len(rows))),
                "task_index": [task_index] * len(rows),
            }
        )
        frame.to_parquet(
            output_dir / "data" / CHUNK / f"episode_{new_index:06d}.parquet", index=False
        )

        for camera in cameras:
            encode_video(
                episode / camera,
                output_dir / "videos" / CHUNK / f"observation.images.{camera}"
                / f"episode_{new_index:06d}.mp4",
                fps,
            )

        episode_records.append(
            {"episode_index": new_index, "tasks": [task], "length": len(rows)}
        )
        total_frames += len(rows)
        print(f"converted {episode.name} -> episode_{new_index:06d} ({len(rows)} frames)")

    state_dim = 7  # 6 arm joints + 1 normalised gripper

    info = {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "total_episodes": len(episode_records),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": len(episode_records) * len(cameras),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{len(episode_records)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [state_dim],
                "names": {"motors": [f"motor_{i}" for i in range(state_dim)]},
            },
            "action": {
                "dtype": "float32",
                "shape": [state_dim],
                "names": {"motors": [f"motor_{i}" for i in range(state_dim)]},
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            **{
                f"observation.images.{camera}": {
                    "dtype": "video",
                    "shape": [224, 224, 3],
                    "names": ["height", "width", "channel"],
                    "info": {
                        "video.fps": fps,
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "has_audio": False,
                    },
                }
                for camera in cameras
            },
        },
    }
    (output_dir / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    with (output_dir / "meta" / "tasks.jsonl").open("w") as handle:
        for task, index in sorted(tasks.items(), key=lambda item: item[1]):
            handle.write(json.dumps({"task_index": index, "task": task}) + "\n")

    with (output_dir / "meta" / "episodes.jsonl").open("w") as handle:
        for record in episode_records:
            handle.write(json.dumps(record) + "\n")

    # GR00T reads modality.json to slice the flat state/action vectors into
    # named modalities; the index ranges must match the column layout above.
    modality = {
        "state": {
            "single_arm": {"start": 0, "end": 6},
            "gripper": {"start": 6, "end": 7},
        },
        "action": {
            "single_arm": {"start": 0, "end": 6},
            "gripper": {"start": 6, "end": 7},
        },
        "video": {camera: {"original_key": f"observation.images.{camera}"} for camera in cameras},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    (output_dir / "meta" / "modality.json").write_text(json.dumps(modality, indent=2))

    print(
        f"\ndone: {len(episode_records)} episodes, {total_frames} frames -> {output_dir}\n"
        f"fine-tune with:\n"
        f"  python scripts/gr00t_finetune.py --dataset-path {output_dir} "
        f"--embodiment-tag NEW_EMBODIMENT"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="episode_recorder output_dir")
    parser.add_argument("--output", required=True, type=Path, help="LeRobot dataset destination")
    parser.add_argument("--robot-type", default="ur5e_parallel_gripper")
    args, _unknown = parser.parse_known_args(argv)

    require("pandas", "pip install pandas")
    require("pyarrow", "pip install pyarrow")
    convert(args.input.expanduser(), args.output.expanduser(), args.robot_type)


if __name__ == "__main__":
    main()
