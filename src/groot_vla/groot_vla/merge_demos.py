"""Merge demonstration directories from parallel collection into one dataset.

`collect_parallel.sh` gives every worker its own output directory, and each
numbers its episodes from zero. Concatenating them naively would collide on
`episode_000000`, so this renumbers as it copies.

    python3 merge_demos.py --inputs ~/ws/data/demos/run1/worker_* \\
                          --output ~/ws/data/demos/run1/merged

Then export the merged directory as usual. Copies rather than moves by default,
so a mistake cannot destroy a collection run that took an hour.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def episode_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("episode_*") if p.is_dir())


def merge(inputs: list[Path], output: Path, move: bool) -> int:
    output.mkdir(parents=True, exist_ok=True)
    existing = episode_dirs(output)
    if existing:
        # Continue numbering rather than overwriting, so merging twice into the
        # same target appends instead of clobbering.
        next_index = max(int(p.name.split("_")[-1]) for p in existing) + 1
        print(f"{output} already holds {len(existing)} episodes; appending from {next_index}")
    else:
        next_index = 0

    total = 0
    for source_root in inputs:
        episodes = episode_dirs(source_root)
        if not episodes:
            print(f"  {source_root}: no episodes, skipping", file=sys.stderr)
            continue
        for episode in episodes:
            meta_path = episode / "meta.json"
            frames_path = episode / "frames.jsonl"
            if not meta_path.exists() or not frames_path.exists():
                # A worker killed mid-episode leaves a directory with images but
                # no metadata; it cannot be trained on.
                print(f"  {episode}: incomplete, skipping", file=sys.stderr)
                continue

            destination = output / f"episode_{next_index:06d}"
            if move:
                shutil.move(str(episode), destination)
            else:
                shutil.copytree(episode, destination)

            # The index inside meta.json has to match the new directory name;
            # the exporter reads both and they must agree.
            metadata = json.loads((destination / "meta.json").read_text())
            metadata["episode_index"] = next_index
            metadata["merged_from"] = str(source_root.name)
            (destination / "meta.json").write_text(json.dumps(metadata, indent=2))

            next_index += 1
            total += 1
        print(f"  {source_root.name}: {len(episodes)} episodes")

    print(f"\nmerged {total} episodes -> {output}")
    return total


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--move", action="store_true",
                        help="move instead of copy (faster, but destructive)")
    args, _unknown = parser.parse_known_args(argv)

    inputs = [p.expanduser() for p in args.inputs if p.expanduser().is_dir()]
    if not inputs:
        raise SystemExit("no input directories found")
    if merge(inputs, args.output.expanduser(), args.move) == 0:
        raise SystemExit("nothing merged")


if __name__ == "__main__":
    main()
