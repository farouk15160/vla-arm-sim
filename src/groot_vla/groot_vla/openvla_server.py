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
import subprocess
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

# Must be set before torch is imported. Expandable segments let the allocator
# grow blocks instead of reserving fixed-size ones, which materially reduces
# fragmentation when loading 7B of weights into a nearly-full card.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ARM_DOF = 6
# Weights need ~4.4 GB in NF4. Below this much free VRAM the model cannot be
# resident and layers are offloaded to CPU instead.
NF4_WEIGHTS_GIB = 4.4
# Room for activations, the vision tower and the CUDA context, on top of the
# weights. Calibrated against a measured successful run: the process peaked at
# 4.28 GiB resident with ~0.5 GiB spare. Larger values are safer but on a 6 GB
# card quickly exceed what the device can ever offer - keep the total below the
# card's usable capacity or the server can never start. Override with
# --gpu-headroom-gib when you want to attempt a marginal fit.
ACTIVATION_HEADROOM_GIB = 0.4


def gpu_memory_gib() -> tuple[float, float]:
    """(free, total) VRAM in GiB, read from the driver rather than torch.

    torch.cuda.mem_get_info reports what the driver sees, so memory held by
    OTHER processes - a game, a browser, another model - is accounted for.
    """
    import torch

    free, total = torch.cuda.mem_get_info()
    return free / 1024 ** 3, total / 1024 ** 3


def describe_gpu_hogs() -> str:
    """Name every process holding VRAM.

    Deliberately parses the full nvidia-smi process table rather than
    --query-compute-apps. That query lists CUDA contexts ONLY, so the desktop
    compositor, the browser and Gazebo's renderer - all graphics contexts, and
    together often more than a gigabyte - are invisible to it. Using it alone
    produces the actively misleading report that the only process on the GPU is
    yourself, while a third of the card is already gone.
    """
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return "    (could not read nvidia-smi)"

    lines = result.stdout.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if "Processes:" in line)
    except StopIteration:
        return "    (no process table)"

    rows = []
    for line in lines[start:]:
        # Data rows look like: |  0  N/A  N/A  1234  G  /usr/lib/xorg/Xorg  602MiB |
        if line.startswith("|") and "MiB" in line and "GPU Memory" not in line:
            rows.append("    " + line.strip("| ").rstrip("| ").strip())
    return "\n".join(rows) or "    (none reported)"


class OpenVLAServer:
    def __init__(
        self,
        model_path: str = "openvla/openvla-7b",
        host: str = "0.0.0.0",
        port: int = 5555,
        unnorm_key: str = "bridge_orig",
        load_in_4bit: bool = True,
        device_map: str = "auto",
        gpu_headroom_gib: float = ACTIVATION_HEADROOM_GIB,
        action_horizon: int = 1,
        verbose: bool = True,
    ) -> None:
        self.unnorm_key = unnorm_key
        self.gpu_headroom_gib = gpu_headroom_gib
        self.action_horizon = max(action_horizon, 1)
        self.verbose = verbose
        self._calls = 0
        self._running = True

        # Decide before importing torch: once it binds the CUDA device,
        # downstream libraries place tensors there whatever we ask for.
        if device_map == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.torch = torch
        self.cpu_only = device_map == "cpu"
        if not self.cpu_only and not torch.cuda.is_available():
            print("[openvla] no CUDA device; falling back to CPU", flush=True)
            self.cpu_only = True
        if self.cpu_only:
            # Quantisation is a GPU technique; on CPU we load bf16 weights,
            # which needs ~14 GB of RAM and is very slow, but it does run.
            print("[openvla] CPU mode: expect tens of seconds per inference", flush=True)
            load_in_4bit = False

        if self.cpu_only:
            free_gib = total_gib = 0.0
        else:
            free_gib, total_gib = gpu_memory_gib()
        if not self.cpu_only:
            print(
                f"[openvla] GPU: {free_gib:.2f} GiB free of {total_gib:.2f} GiB",
                flush=True,
            )

        # A 4-bit model must be FULLY resident on the GPU. Letting accelerate
        # offload some layers to CPU loads without complaint and then fails on
        # the first inference with "Blockwise 4bit quantization only supports
        # 16/32-bit floats, but got torch.uint8" - bitsandbytes cannot
        # dequantize blocks living in host memory. Refusing up front with a
        # precise shortfall beats a server that starts and then breaks.
        needed = NF4_WEIGHTS_GIB + self.gpu_headroom_gib if load_in_4bit else 14.0
        if not self.cpu_only and free_gib < needed:
            shortfall = int((needed - free_gib) * 1024)
            raise SystemExit(
                f"\n[openvla] NOT ENOUGH VRAM.\n"
                f"  free:     {free_gib:.2f} GiB\n"
                f"  required: {needed:.2f} GiB  ({NF4_WEIGHTS_GIB:.1f} weights + "
                f"{self.gpu_headroom_gib:.1f} activations)\n"
                f"  short by: {shortfall} MiB\n\n"
                f"A 4-bit model cannot be partially offloaded: it would load and then\n"
                f"fail on the first inference. Free that much VRAM, or pick another route:\n\n"
                f"  * close GPU-heavy desktop apps (a browser is often 150-400 MiB)\n"
                f"  * run the simulator headless:  gazebo_gui:=false   (~200 MiB)\n"
                f"  * use SmolVLA instead, which needs 0.9 GiB:\n"
                f"        ros2 launch groot_arm_bringup system.launch.py policy:=smolvla\n"
                f"  * run OpenVLA on another machine and point the stack at it:\n"
                f"        system.launch.py policy:=openvla policy_host:=<that machine>\n"
                f"  * last resort, entirely on CPU (~14 GB RAM, very slow):\n"
                f"        openvla_server.py --device-map cpu\n\n"
                f"Current GPU users (graphics contexts included):\n{describe_gpu_hogs()}\n"
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
            # fp32_cpu_offload is what permits accelerate to place layers that
            # do not fit on the CPU rather than raising OutOfMemoryError.
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            # Everything on device 0. Room has already been established above;
            # "auto" with a budget would silently offload the overflow, which
            # 4-bit cannot survive at inference time.
            kwargs["device_map"] = {"": 0}
        elif self.cpu_only:
            kwargs["device_map"] = {"": "cpu"}
        else:
            kwargs["device_map"] = {"": 0}

        try:
            self.model = AutoModelForVision2Seq.from_pretrained(model_path, **kwargs)
        except torch.OutOfMemoryError as exc:
            raise SystemExit(
                f"[openvla] out of VRAM while loading: {exc}\n"
                f"Free the GPU, or use smolvla_server (0.9 GiB).\n"
                f"Current GPU users:\n{describe_gpu_hogs()}"
            ) from exc

        if not load_in_4bit and not self.cpu_only:
            self.model = self.model.to("cuda:0")
        self.model.eval()

        # Report where the model actually ended up: silently running half the
        # layers on the CPU explains an otherwise baffling 10x slowdown.
        placement = getattr(self.model, "hf_device_map", None)
        if isinstance(placement, dict):
            on_cpu = sum(1 for d in placement.values() if d in ("cpu", "disk"))
            if on_cpu:
                print(
                    f"[openvla] {on_cpu}/{len(placement)} module groups offloaded to CPU "
                    f"- expect much slower inference",
                    flush=True,
                )

        available = self._available_unnorm_keys()
        if available and self.unnorm_key not in available:
            fallback = available[0]
            print(
                f"[openvla] unnorm_key {self.unnorm_key!r} not in checkpoint; "
                f"using {fallback!r}. Available: {available}",
                flush=True,
            )
            self.unnorm_key = fallback

        if not self.cpu_only:
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

        device = "cpu" if self.cpu_only else "cuda:0"
        inputs = self.processor(prompt, image).to(device, dtype=self.torch.bfloat16)
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
    parser.add_argument("--device-map", default="auto", choices=["auto", "cpu"],
                        help="'cpu' runs entirely on CPU when the GPU is busy")
    parser.add_argument("--gpu-headroom-gib", type=float, default=ACTIVATION_HEADROOM_GIB,
                        help="VRAM held back from the weight budget for activations")
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
        device_map=args.device_map,
        gpu_headroom_gib=args.gpu_headroom_gib,
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
