"""SmolVLA policy server - a drop-in replacement for the GR00T server.

Why SmolVLA: GR00T N1.7-3B needs 16 GB+ of VRAM. SmolVLA is ~450M parameters
(about 0.9 GB in bf16) and was built for consumer hardware, so it fits a 6 GB
card with room for the sim. It is a real vision-language-action policy: RGB
frames plus a natural-language instruction in, action chunks out.

This speaks the SAME ZeroMQ + msgpack protocol as the GR00T inference server
and the bundled mock, so nothing in the ROS stack changes - point
`policy_host`/`policy_port` at this process instead.

It must run in an environment that has torch + lerobot, which is NOT the ROS
environment:

    python3 -m venv ~/vla_venv
    ~/vla_venv/bin/pip install "lerobot[smolvla] @ file:///path/to/lerobot" \
                               pyzmq msgpack msgpack-numpy
    ~/vla_venv/bin/python smolvla_server.py --port 5555

Zero-shot performance on a UR5e with this gripper will be poor: it is an
unseen embodiment with unfamiliar joint ranges, and the checkpoint's
normalisation statistics come from its own training data. Record
demonstrations (`episode_recorder`), convert them (`export_lerobot`) and
fine-tune. The point of this server is that the whole loop runs and is
measurable on hardware you actually have.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from typing import Any

import numpy as np

# groot_client is ROS-free (numpy + zmq + msgpack only), so it imports cleanly
# in the policy venv when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from groot_vla.groot_client import packb, unpackb
except ImportError:  # running outside the ROS package
    from groot_client import packb, unpackb  # type: ignore[no-redef]

import zmq

ARM_DOF = 6
# Weights are ~0.9 GiB; the rest is activations and the CUDA context.
MIN_FREE_VRAM_GIB = 1.5


def free_vram_gib() -> float | None:
    """Free VRAM in GiB, read WITHOUT importing torch.

    This has to work before torch is imported. Once torch initialises it binds
    the CUDA device, and libraries downstream (transformers, LeRobot) then
    place tensors there regardless of any device we ask for - which is exactly
    how a 'fall back to CPU' path still dies with a CUDA OOM.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        return int(result.stdout.strip().splitlines()[0]) / 1024.0
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def describe_gpu_hogs() -> str:
    """Name the processes holding VRAM, so the fallback is explicable."""
    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "    (could not read nvidia-smi)"
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return "\n".join(f"    {line}" for line in lines) or "    (none reported)"


# --------------------------------------------------------------------------- #
# observation translation
# --------------------------------------------------------------------------- #
def decode_key(key: Any) -> str:
    return key.decode() if isinstance(key, bytes) else str(key)


def flatten_observation(observation: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], str]:
    """Split either observation schema into (videos, states, instruction).

    Accepts the nested N1.7 layout and the flat N1.5 layout, because the ROS
    client can be configured for either and this server should not care.
    """
    videos: dict[str, np.ndarray] = {}
    states: dict[str, np.ndarray] = {}
    instruction = ""

    for raw_key, value in observation.items():
        key = decode_key(raw_key)
        if key == "video" and isinstance(value, dict):
            videos.update({decode_key(k): np.asarray(v) for k, v in value.items()})
        elif key.startswith("video."):
            videos[key.removeprefix("video.")] = np.asarray(value)
        elif key == "state" and isinstance(value, dict):
            states.update({decode_key(k): np.asarray(v) for k, v in value.items()})
        elif key.startswith("state."):
            states[key.removeprefix("state.")] = np.asarray(value)
        elif key == "language" and isinstance(value, dict):
            task = next(iter(value.values()), None)
            instruction = _unwrap_text(task)
        elif key.startswith("annotation."):
            instruction = _unwrap_text(value)

    return videos, states, instruction


def _unwrap_text(value: Any) -> str:
    while isinstance(value, (list, tuple)) and value:
        value = value[0]
    if isinstance(value, bytes):
        return value.decode()
    return value if isinstance(value, str) else ""


def squeeze_frame(array: np.ndarray) -> np.ndarray:
    """(B, T, H, W, C) -> (H, W, C), taking the most recent frame."""
    while array.ndim > 3:
        array = array[-1]
    return array


def squeeze_state(array: np.ndarray) -> np.ndarray:
    """(B, T, D) -> (D,), taking the most recent step."""
    array = np.asarray(array, dtype=np.float32)
    while array.ndim > 1:
        array = array[-1]
    return array


# --------------------------------------------------------------------------- #
# server
# --------------------------------------------------------------------------- #
class SmolVLAServer:
    def __init__(
        self,
        model_path: str = "lerobot/smolvla_base",
        host: str = "0.0.0.0",
        port: int = 5555,
        device: str = "cuda",
        action_horizon: int = 16,
        state_dim: int = 7,
        verbose: bool = True,
    ) -> None:
        self.action_horizon = action_horizon
        self.state_dim = state_dim
        self.verbose = verbose
        self._calls = 0
        self._running = True
        self._latencies: list[float] = []

        # Decide CPU-vs-GPU BEFORE torch is imported. A full GPU is as fatal as
        # a missing one, and the failure is far more confusing: loading dies
        # part way through with a CUDA OOM. SmolVLA is only 450M, so CPU is a
        # genuinely usable fallback - a few seconds per inference, not a crash.
        if device == "cuda":
            free_gib = free_vram_gib()
            if free_gib is not None and free_gib < MIN_FREE_VRAM_GIB:
                print(
                    f"[smolvla] only {free_gib:.2f} GiB VRAM free, need about "
                    f"{MIN_FREE_VRAM_GIB:.1f} GiB. Falling back to CPU (slower).\n"
                    f"[smolvla] GPU is currently used by:\n{describe_gpu_hogs()}",
                    flush=True,
                )
                device = "cpu"
        if device == "cpu":
            # Hiding the device is the only reliable way to keep downstream
            # libraries off the GPU; asking them politely is not enough.
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        import torch

        self.torch = torch
        if device == "cuda" and not torch.cuda.is_available():
            print("[smolvla] CUDA not available, falling back to CPU", flush=True)
            device = "cpu"
        self.device = torch.device(device)

        print(f"[smolvla] loading {model_path} ...", flush=True)
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.policy = SmolVLAPolicy.from_pretrained(model_path)
        self.policy.to(self.device)
        self.policy.eval()
        # The saved processor config hard-codes the device it was exported with
        # (cuda). Instantiating it on a CPU-only run fails outright, so the
        # device step is overridden to whatever we actually resolved to.
        device_override = {"device_processor": {"device": str(self.device)}}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=model_path,
            preprocessor_overrides=device_override,
            postprocessor_overrides=device_override,
        )

        # The checkpoint declares which camera keys and state width it was
        # trained with; our two cameras are mapped onto those in order.
        self.image_keys = [
            key for key in self.policy.config.input_features if key.startswith("observation.image")
        ]
        state_feature = self.policy.config.input_features.get("observation.state")
        self.expected_state_dim = (
            int(np.prod(state_feature.shape)) if state_feature is not None else state_dim
        )
        action_feature = self.policy.config.output_features.get("action")
        self.action_dim = int(np.prod(action_feature.shape)) if action_feature is not None else 7

        if device == "cuda" and torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[smolvla] VRAM {used:.2f} / {total:.1f} GB", flush=True)

        print(
            f"[smolvla] ready. device={self.device} image_keys={self.image_keys} "
            f"state_dim={self.expected_state_dim} action_dim={self.action_dim} "
            f"chunk={self.policy.config.chunk_size}",
            flush=True,
        )

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.bind(f"tcp://{host}:{port}")
        self._socket.setsockopt(zmq.RCVTIMEO, 500)  # keeps Ctrl-C responsive
        print(f"[smolvla] listening on tcp://{host}:{port}", flush=True)

    def stop(self, *_args: object) -> None:
        self._running = False

    # ------------------------------------------------------------------ #
    def _build_batch(self, videos: dict[str, np.ndarray], states: dict[str, np.ndarray], task: str):
        from lerobot.policies.utils import prepare_observation_for_inference

        # Concatenate the state modalities in a fixed order, then pad or trim
        # to whatever width the checkpoint expects.
        parts = []
        for name in ("single_arm", "arm", "joints"):
            if name in states:
                parts.append(squeeze_state(states[name]))
                break
        for name in ("gripper", "hand"):
            if name in states:
                parts.append(squeeze_state(states[name]))
                break
        state = np.concatenate(parts) if parts else np.zeros(self.state_dim, dtype=np.float32)

        if state.shape[0] < self.expected_state_dim:
            state = np.pad(state, (0, self.expected_state_dim - state.shape[0]))
        elif state.shape[0] > self.expected_state_dim:
            state = state[: self.expected_state_dim]

        observation: dict[str, np.ndarray] = {"observation.state": state.astype(np.float32)}

        # Map our cameras onto the checkpoint's expected image keys, in order.
        frames = [squeeze_frame(v) for v in videos.values()]
        if not frames:
            raise ValueError("observation contained no video modality")
        for index, key in enumerate(self.image_keys):
            frame = frames[index] if index < len(frames) else frames[-1]
            observation[key] = np.ascontiguousarray(frame.astype(np.uint8))

        batch = prepare_observation_for_inference(observation, self.device, task=task)
        return self.preprocessor(batch)

    def _get_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        videos, states, task = flatten_observation(observation)
        batch = self._build_batch(videos, states, task)

        with self.torch.inference_mode():
            chunk = self.policy.predict_action_chunk(batch)
            chunk = self.postprocessor(chunk)

        actions = chunk.detach().to("cpu", dtype=self.torch.float32).numpy()
        if actions.ndim == 2:  # (T, D) -> (1, T, D)
            actions = actions[np.newaxis, ...]
        actions = actions[:, : self.action_horizon, :]

        arm = actions[:, :, :ARM_DOF]
        if actions.shape[2] > ARM_DOF:
            gripper = actions[:, :, ARM_DOF : ARM_DOF + 1]
        else:
            # No gripper head: hold whatever the observation reported.
            held = float(squeeze_state(states.get("gripper", np.ones(1, dtype=np.float32)))[0])
            gripper = np.full((actions.shape[0], actions.shape[1], 1), held, dtype=np.float32)

        return {
            "action.single_arm": arm.astype(np.float32),
            "action.gripper": gripper.astype(np.float32),
        }

    def _modality_config(self) -> dict[str, Any]:
        return {
            "video": {"modality_keys": ["ego_view", "wrist_view"], "delta_indices": [0]},
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
            return {"status": "ok", "server": "smolvla", "device": str(self.device)}
        if endpoint == "get_modality_config":
            return self._modality_config()
        if endpoint == "reset":
            self.policy.reset()
            self._latencies.clear()
            return {"status": "reset"}
        if endpoint == "kill":
            self._running = False
            return {"status": "shutting down"}
        if endpoint == "get_action":
            if not isinstance(data, dict):
                return {"error": "get_action needs an observation dict"}
            started = time.monotonic()
            action = self._get_action(data)
            elapsed = (time.monotonic() - started) * 1000.0
            self._latencies.append(elapsed)
            self._calls += 1
            if self.verbose and self._calls % 20 == 1:
                recent = self._latencies[-20:]
                print(
                    f"[smolvla] call {self._calls}: {elapsed:.0f} ms "
                    f"(mean {sum(recent)/len(recent):.0f} ms)",
                    flush=True,
                )
            return action
        return {"error": f"unknown endpoint {endpoint!r}"}

    def serve(self) -> None:
        while self._running:
            try:
                raw = self._socket.recv()
            except zmq.Again:
                continue
            except zmq.ZMQError as exc:
                print(f"[smolvla] socket error: {exc}", file=sys.stderr, flush=True)
                break
            try:
                reply = self._handle(unpackb(raw))
            except Exception as exc:  # noqa: BLE001 - a REP socket MUST reply or
                # the client's REQ socket wedges forever
                import traceback

                traceback.print_exc()
                reply = {"error": f"{type(exc).__name__}: {exc}"}
            self._socket.send(packb(reply))

        self._socket.close(linger=0)
        self._context.term()
        print(f"[smolvla] stopped after {self._calls} inferences", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-path", default="lerobot/smolvla_base",
                        help="HF id or local path to a fine-tuned checkpoint")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--action-horizon", type=int, default=16,
                        help="chunk steps returned per inference")
    parser.add_argument("--quiet", action="store_true")
    args, _unknown = parser.parse_known_args(argv)

    server = SmolVLAServer(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        device=args.device,
        action_horizon=args.action_horizon,
        verbose=not args.quiet,
    )
    signal.signal(signal.SIGINT, server.stop)
    signal.signal(signal.SIGTERM, server.stop)
    server.serve()


if __name__ == "__main__":
    main()
