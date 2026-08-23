"""OpenVLA-7B policy server, 4-bit quantised to fit a 6 GB card.

The third policy option, and the largest that fits locally:

    mock        -    no model, protocol only
    SmolVLA     - 450M, ~0.9 GB   fast (~1 Hz on a 2060), no gripper head
    OpenVLA-7B  -   7B, ~3.9 GB   in 4-bit NF4, slower, 7-DoF EEF deltas

OpenVLA is a Llama-2-7B backbone with DINOv2 + SigLIP vision, trained on Open
X-Embodiment. Loaded in 4-bit it needs about 3.9 GB of VRAM - it does not fit
otherwise, since bf16 would want ~14 GB. Quantisation is officially supported
by the checkpoint, so this is not a hack; it does cost some accuracy.

It emits a **single** 7-vector per call, not a chunk:

    [dx, dy, dz, droll, dpitch, dyaw, gripper]

which are end-effector deltas. That maps onto this stack's `eef_delta` action
space, so run it with moveit_servo:

    ros2 launch groot_arm_moveit_config servo.launch.py
    ros2 launch groot_vla openvla_policy.launch.py

Same ZeroMQ + msgpack protocol as every other server here, so nothing in the
ROS stack changes.

Run it in the policy venv, which needs bitsandbytes:

    ~/vla_venv/bin/pip install bitsandbytes timm
    ~/vla_venv/bin/python openvla_server.py --port 5555

The `unnorm_key` selects which training dataset's action statistics are used to
un-normalise the output. It MUST match a key the checkpoint carries;
`--list-unnorm-keys` prints the options.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from groot_vla.groot_client import packb, unpackb
    from groot_vla.smolvla_server import decode_key, flatten_observation, squeeze_frame, squeeze_state
except ImportError:  # running directly in the policy venv
    from groot_client import packb, unpackb  # type: ignore[no-redef]
    from smolvla_server import (  # type: ignore[no-redef]
        decode_key,
        flatten_observation,
        squeeze_frame,
        squeeze_state,
    )

import zmq

ARM_DOF = 6


class OpenVLAServer:
    def __init__(
        self,
        model_path: str = "openvla/openvla-7b",
        host: str = "0.0.0.0",
        port: int = 5555,
        unnorm_key: str = "bridge_orig",
        load_in_4bit: bool = True,
        action_horizon: int = 1,
        verbose: bool = True,
    ) -> None:
        self.unnorm_key = unnorm_key
        self.action_horizon = max(action_horizon, 1)
        self.verbose = verbose
        self._calls = 0
        self._running = True

        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.torch = torch
        if not torch.cuda.is_available():
            raise SystemExit(
                "OpenVLA needs a CUDA GPU. Use smolvla_server (which runs on CPU) instead."
            )

        print(f"[openvla] loading {model_path} (4bit={load_in_4bit}) ...", flush=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.bfloat16,
        }
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            # NF4 with double quantisation: the smallest footprint the
            # checkpoint supports without a noticeable accuracy cliff.
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["device_map"] = {"": 0}

        self.model = AutoModelForVision2Seq.from_pretrained(model_path, **kwargs)
        if not load_in_4bit:
            self.model = self.model.to("cuda:0")
        self.model.eval()

        available = self._available_unnorm_keys()
        if available and self.unnorm_key not in available:
            fallback = available[0]
            print(
                f"[openvla] unnorm_key {self.unnorm_key!r} not in checkpoint; "
                f"using {fallback!r}. Available: {available}",
                flush=True,
            )
            self.unnorm_key = fallback

        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[openvla] VRAM {used:.2f} / {total:.1f} GB", flush=True)
        print(f"[openvla] ready. unnorm_key={self.unnorm_key}", flush=True)

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.bind(f"tcp://{host}:{port}")
        self._socket.setsockopt(zmq.RCVTIMEO, 500)
        print(f"[openvla] listening on tcp://{host}:{port}", flush=True)

    def _available_unnorm_keys(self) -> list[str]:
        stats = getattr(self.model, "norm_stats", None)
        return sorted(stats.keys()) if isinstance(stats, dict) else []

    def stop(self, *_args: object) -> None:
        self._running = False

    # ------------------------------------------------------------------ #
    def _predict(self, observation: dict[str, Any]) -> dict[str, Any]:
        from PIL import Image

        videos, states, task = flatten_observation(observation)
        if not videos:
            raise ValueError("observation contained no video modality")

        # OpenVLA is single-image. Prefer the third-person view: it was trained
        # on external cameras, not wrist-mounted ones.
        frame = None
        for preferred in ("ego_view", "scene", "third_person"):
            if preferred in videos:
                frame = squeeze_frame(videos[preferred])
                break
        if frame is None:
            frame = squeeze_frame(next(iter(videos.values())))
        image = Image.fromarray(np.ascontiguousarray(frame.astype(np.uint8)))

        instruction = (task or "pick up the object").lower().strip()
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"

        inputs = self.processor(prompt, image).to("cuda:0", dtype=self.torch.bfloat16)
        with self.torch.inference_mode():
            action = self.model.predict_action(
                **inputs, unnorm_key=self.unnorm_key, do_sample=False
            )
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape[0] < 7:
            action = np.pad(action, (0, 7 - action.shape[0]))

        # [dx dy dz droll dpitch dyaw gripper]; the ROS side reads the first
        # six as a twist and the seventh as the gripper command.
        twist = action[:6][np.newaxis, np.newaxis, :].repeat(self.action_horizon, axis=1)
        gripper = action[6:7][np.newaxis, np.newaxis, :].repeat(self.action_horizon, axis=1)
        return {
            "action.single_arm": twist.astype(np.float32),
            "action.gripper": gripper.astype(np.float32),
        }

    def _modality_config(self) -> dict[str, Any]:
        return {
            "video": {"modality_keys": ["ego_view"], "delta_indices": [0]},
            "state": {"modality_keys": ["single_arm", "gripper"], "delta_indices": [0]},
            "action": {
                "modality_keys": ["single_arm", "gripper"],
                "delta_indices": list(range(self.action_horizon)),
            },
            "language": {"modality_keys": ["task"], "delta_indices": [0]},
        }

    def _handle(self, request: Any) -> Any:
        if not isinstance(request, dict):
            return {"error": f"malformed request of type {type(request).__name__}"}
        endpoint = decode_key(request.get("endpoint") or request.get(b"endpoint") or "")
        data = request.get("data", request.get(b"data"))

        if endpoint == "ping":
            return {"status": "ok", "server": "openvla", "unnorm_key": self.unnorm_key}
        if endpoint == "get_modality_config":
            return self._modality_config()
        if endpoint == "reset":
            return {"status": "reset"}  # OpenVLA is stateless between calls
        if endpoint == "kill":
            self._running = False
            return {"status": "shutting down"}
        if endpoint == "get_action":
            if not isinstance(data, dict):
                return {"error": "get_action needs an observation dict"}
            started = time.monotonic()
            action = self._predict(data)
            elapsed = (time.monotonic() - started) * 1000.0
            self._calls += 1
            if self.verbose and self._calls % 10 == 1:
                print(f"[openvla] call {self._calls}: {elapsed:.0f} ms", flush=True)
            return action
        return {"error": f"unknown endpoint {endpoint!r}"}

    def serve(self) -> None:
        while self._running:
            try:
                raw = self._socket.recv()
            except zmq.Again:
                continue
            except zmq.ZMQError as exc:
                print(f"[openvla] socket error: {exc}", file=sys.stderr, flush=True)
                break
            try:
                reply = self._handle(unpackb(raw))
            except Exception as exc:  # noqa: BLE001 - a REP socket MUST reply
                import traceback

                traceback.print_exc()
                reply = {"error": f"{type(exc).__name__}: {exc}"}
            self._socket.send(packb(reply))

        self._socket.close(linger=0)
        self._context.term()
        print(f"[openvla] stopped after {self._calls} inferences", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-path", default="openvla/openvla-7b")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--unnorm-key", default="bridge_orig",
                        help="which dataset's action statistics to un-normalise with")
    parser.add_argument("--no-4bit", action="store_true",
                        help="load in bf16 (needs ~14 GB VRAM)")
    parser.add_argument("--action-horizon", type=int, default=1)
    parser.add_argument("--list-unnorm-keys", action="store_true",
                        help="print the checkpoint's dataset keys and exit")
    parser.add_argument("--quiet", action="store_true")
    args, _unknown = parser.parse_known_args(argv)

    server = OpenVLAServer(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        unnorm_key=args.unnorm_key,
        load_in_4bit=not args.no_4bit,
        action_horizon=args.action_horizon,
        verbose=not args.quiet,
    )
    if args.list_unnorm_keys:
        print("available unnorm keys:", server._available_unnorm_keys())
        return
    signal.signal(signal.SIGINT, server.stop)
    signal.signal(signal.SIGTERM, server.stop)
    server.serve()


if __name__ == "__main__":
    main()
