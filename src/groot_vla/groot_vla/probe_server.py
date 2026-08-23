"""Diagnose a GR00T policy server before wiring the robot to it.

Answers the questions that otherwise cost an hour of confused debugging:
is it reachable, which observation schema does it accept, what modality keys
does it expect, what shape are the actions, and how slow is a forward pass?

    ros2 run groot_vla probe_server --host 10.0.0.42 --port 5555
    ros2 run groot_vla probe_server --show-config
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from groot_vla.action_mapper import flatten_action
from groot_vla.groot_client import USING_REAL_MSGPACK_NUMPY, GrootClient, PolicyServerError


def synthetic_observation(schema: str, width: int, height: int, instruction: str) -> dict:
    """A plausible single-arm observation, used purely to shake out the server."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (1, 1, height, width, 3), dtype=np.uint8)
    arm = np.zeros((1, 1, 6), dtype=np.float32)
    gripper = np.ones((1, 1, 1), dtype=np.float32)

    if schema == "nested":
        return {
            "video": {"ego_view": image, "wrist_view": image.copy()},
            "state": {"single_arm": arm, "gripper": gripper},
            "language": {"task": [[instruction]]},
        }
    return {
        "video.ego_view": image,
        "video.wrist_view": image.copy(),
        "state.single_arm": arm,
        "state.gripper": gripper,
        "annotation.human.task_description": [instruction],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--timeout-ms", type=int, default=20000)
    parser.add_argument("--schema", choices=["nested", "flat", "both"], default="both")
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--instruction", default="pick up the red cube")
    parser.add_argument("--show-config", action="store_true",
                        help="print the server's modality config and exit")
    parser.add_argument("--repeat", type=int, default=3, help="timed get_action calls")
    args, _unknown = parser.parse_known_args(argv)

    client = GrootClient(args.host, args.port, timeout_ms=args.timeout_ms)
    print(f"endpoint: {client.endpoint}")
    print(f"codec:    {'msgpack_numpy' if USING_REAL_MSGPACK_NUMPY else 'builtin compatible'}")

    print("\n[1] ping ...", end=" ", flush=True)
    if not client.ping():
        print("NO REPLY")
        raise SystemExit(
            "Server unreachable. Check that it is running, the port is open, "
            "and no firewall sits in between."
        )
    print("ok")

    print("\n[2] modality config")
    try:
        config = client.get_modality_config()
        _print_tree(config, indent=4)
    except PolicyServerError as exc:
        print(f"    unavailable: {exc}")
        config = None
    if args.show_config:
        return

    schemas = ["nested", "flat"] if args.schema == "both" else [args.schema]
    working: list[str] = []
    for schema in schemas:
        print(f"\n[3] get_action with {schema!r} schema")
        observation = synthetic_observation(schema, args.width, args.height, args.instruction)
        try:
            latencies = []
            action = None
            for _ in range(max(args.repeat, 1)):
                started = time.monotonic()
                action = client.get_action(observation)
                latencies.append((time.monotonic() - started) * 1000.0)
            flat = flatten_action(action)
            print(f"    accepted. latency: min {min(latencies):.0f} ms / "
                  f"mean {sum(latencies)/len(latencies):.0f} ms")
            for key, value in sorted(flat.items()):
                array = np.asarray(value)
                print(f"    {key}: shape={array.shape} dtype={array.dtype}")
            working.append(schema)
        except PolicyServerError as exc:
            print(f"    rejected: {exc}")
        except Exception as exc:  # noqa: BLE001 - report, do not crash the probe
            print(f"    failed: {type(exc).__name__}: {exc}")

    print("\nsummary")
    if working:
        print(f"    set observation_schema: \"{working[0]}\" in groot_policy.yaml")
    else:
        print("    no schema accepted. Compare the modality config above with "
              "ObservationBuilder's keys and adjust *_camera_key / state_keys.")
    client.close()


def _print_tree(value: object, indent: int = 0) -> None:
    pad = " " * indent
    if isinstance(value, dict):
        for key, item in value.items():
            name = key.decode() if isinstance(key, bytes) else key
            if isinstance(item, (dict, list)):
                print(f"{pad}{name}:")
                _print_tree(item, indent + 2)
            else:
                print(f"{pad}{name}: {item}")
    elif isinstance(value, list):
        print(f"{pad}{value}")
    else:
        print(f"{pad}{value}")


if __name__ == "__main__":
    main()
