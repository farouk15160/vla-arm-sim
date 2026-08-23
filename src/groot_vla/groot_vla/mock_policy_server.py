"""A stand-in GR00T inference server.

Speaks the real wire protocol (ZMQ REQ/REP + msgpack-numpy) and returns
well-formed action chunks, so the entire ROS pipeline - observation assembly,
serialisation, chunk decoding, clamping, trajectory execution - can be
exercised without a 16 GB GPU or a model download.

It is NOT a policy. Two scripted behaviours are available:

  hold    every action repeats the observed joint state (the arm should not
          move at all; the sharpest test that your plumbing adds no drift)
  wave    a slow sinusoid on the wrist joints plus a periodic gripper cycle
          (proves trajectories actually reach the controller)

Run:
    ros2 run groot_vla mock_policy_server --behaviour wave --port 5555
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from typing import Any

import numpy as np
import zmq

from groot_vla.groot_client import packb, unpackb

ARM_DOF = 6


def extract_state(observation: dict[str, Any]) -> tuple[np.ndarray, float]:
    """Pull (arm joint vector, gripper scalar) out of either schema."""
    arm = np.zeros(ARM_DOF, dtype=np.float64)
    gripper = 1.0

    def _decode(key: Any) -> str:
        return key.decode() if isinstance(key, bytes) else str(key)

    state: dict[str, Any] = {}
    for key, value in observation.items():
        name = _decode(key)
        if name == "state" and isinstance(value, dict):
            state.update({_decode(k): v for k, v in value.items()})
        elif name.startswith("state."):
            state[name.removeprefix("state.")] = value

    if "single_arm" in state:
        arm = np.asarray(state["single_arm"], dtype=np.float64).reshape(-1)[:ARM_DOF]
    if "gripper" in state:
        gripper = float(np.asarray(state["gripper"], dtype=np.float64).reshape(-1)[0])
    return arm, gripper


def extract_instruction(observation: dict[str, Any]) -> str:
    for key, value in observation.items():
        name = key.decode() if isinstance(key, bytes) else str(key)
        if name == "language" and isinstance(value, dict):
            task = next(iter(value.values()), None)
            while isinstance(task, (list, tuple)) and task:
                task = task[0]
            if isinstance(task, bytes):
                return task.decode()
            if isinstance(task, str):
                return task
        if name.startswith("annotation."):
            task = value
            while isinstance(task, (list, tuple)) and task:
                task = task[0]
            if isinstance(task, bytes):
                return task.decode()
            if isinstance(task, str):
                return task
    return ""


class MockPolicyServer:
    def __init__(
        self,
        port: int = 5555,
        host: str = "0.0.0.0",
        behaviour: str = "wave",
        horizon: int = 16,
        amplitude: float = 0.25,
        period: float = 12.0,
        verbose: bool = True,
    ) -> None:
        self.behaviour = behaviour
        self.horizon = horizon
        self.amplitude = amplitude
        self.period = period
        self.verbose = verbose
        self._calls = 0
        self._started = time.monotonic()
        self._running = True
        # Reference pose for 'wave'. Captured from the first observation and
        # held: waving around the *observed* state instead would make each
        # cycle add to the last, integrating into runaway motion.
        self._reference: np.ndarray | None = None

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.bind(f"tcp://{host}:{port}")
        self._socket.setsockopt(zmq.RCVTIMEO, 500)  # so Ctrl-C is responsive
        print(f"[mock] listening on tcp://{host}:{port} behaviour={behaviour}", flush=True)

    def stop(self, *_args: object) -> None:
        self._running = False

    # ------------------------------------------------------------------ #
    def _action_chunk(self, observation: dict[str, Any]) -> dict[str, Any]:
        arm, gripper = extract_state(observation)
        steps = np.arange(self.horizon)

        if self._reference is None:
            self._reference = arm.copy()

        if self.behaviour == "hold":
            arm_chunk = np.repeat(arm[np.newaxis, :], self.horizon, axis=0)
            gripper_chunk = np.full((self.horizon, 1), gripper)
        elif self.behaviour == "wave":
            elapsed = time.monotonic() - self._started
            phase = 2.0 * np.pi * (elapsed + steps * 0.1) / self.period
            arm_chunk = np.repeat(self._reference[np.newaxis, :], self.horizon, axis=0)
            # Move only wrist_1 and shoulder_pan: large enough to see, small
            # enough that the arm cannot reach the table or itself.
            arm_chunk[:, 0] += self.amplitude * np.sin(phase)
            arm_chunk[:, 3] += 0.5 * self.amplitude * np.sin(2.0 * phase)
            # Gripper opens and closes on a slower cycle.
            gripper_chunk = (
                0.5 + 0.5 * np.sin(2.0 * np.pi * elapsed / (self.period * 1.5))
            ) * np.ones((self.horizon, 1))
        else:
            raise ValueError(f"unknown behaviour {self.behaviour!r}")

        # Shape (B, T, D) with B=1, matching the documented GR00T action format.
        return {
            "action.single_arm": arm_chunk[np.newaxis, ...].astype(np.float32),
            "action.gripper": gripper_chunk[np.newaxis, ...].astype(np.float32),
        }

    def _modality_config(self) -> dict[str, Any]:
        return {
            "video": {"modality_keys": ["ego_view", "wrist_view"], "delta_indices": [0]},
            "state": {"modality_keys": ["single_arm", "gripper"], "delta_indices": [0]},
            "action": {
                "modality_keys": ["single_arm", "gripper"],
                "delta_indices": list(range(self.horizon)),
            },
            "language": {"modality_keys": ["task"], "delta_indices": [0]},
        }

    def _handle(self, request: Any) -> Any:
        if not isinstance(request, dict):
            return {"error": f"malformed request of type {type(request).__name__}"}
        endpoint = request.get("endpoint") or request.get(b"endpoint")
        if isinstance(endpoint, bytes):
            endpoint = endpoint.decode()
        data = request.get("data", request.get(b"data"))

        if endpoint == "ping":
            return {"status": "ok", "server": "groot_vla mock", "behaviour": self.behaviour}
        if endpoint == "get_modality_config":
            return self._modality_config()
        if endpoint == "reset":
            self._started = time.monotonic()
            self._reference = None
            return {"status": "reset"}
        if endpoint == "kill":
            self._running = False
            return {"status": "shutting down"}
        if endpoint == "get_action":
            self._calls += 1
            if not isinstance(data, dict):
                return {"error": "get_action needs an observation dict"}
            if self.verbose and self._calls % 20 == 1:
                arm, gripper = extract_state(data)
                print(
                    f"[mock] call {self._calls}: task={extract_instruction(data)!r} "
                    f"arm={np.round(arm, 3)} gripper={gripper:.2f}",
                    flush=True,
                )
            return self._action_chunk(data)
        return {"error": f"unknown endpoint {endpoint!r}"}

    def serve(self) -> None:
        while self._running:
            try:
                raw = self._socket.recv()
            except zmq.Again:
                continue
            except zmq.ZMQError as exc:
                print(f"[mock] socket error: {exc}", file=sys.stderr, flush=True)
                break
            try:
                reply = self._handle(unpackb(raw))
            except Exception as exc:  # noqa: BLE001 - a REP socket MUST reply,
                # otherwise the client's REQ socket wedges forever
                reply = {"error": f"{type(exc).__name__}: {exc}"}
            self._socket.send(packb(reply))

        self._socket.close(linger=0)
        self._context.term()
        print(f"[mock] stopped after {self._calls} get_action calls", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--behaviour", "--behavior", dest="behaviour",
                        choices=["hold", "wave"], default="wave")
    parser.add_argument("--horizon", type=int, default=16, help="action chunk length")
    parser.add_argument("--amplitude", type=float, default=0.25, help="wave size [rad]")
    parser.add_argument("--period", type=float, default=12.0, help="wave period [s]")
    parser.add_argument("--quiet", action="store_true")
    # ros2 run injects --ros-args; ignore anything we do not recognise.
    args, _unknown = parser.parse_known_args(argv)

    server = MockPolicyServer(
        port=args.port,
        host=args.host,
        behaviour=args.behaviour,
        horizon=args.horizon,
        amplitude=args.amplitude,
        period=args.period,
        verbose=not args.quiet,
    )
    signal.signal(signal.SIGINT, server.stop)
    signal.signal(signal.SIGTERM, server.stop)
    server.serve()


if __name__ == "__main__":
    main()
