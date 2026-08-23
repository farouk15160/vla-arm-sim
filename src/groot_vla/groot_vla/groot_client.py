"""Torch-free client for an Isaac GR00T policy server.

Why not just import ``gr00t.policy.server_client.PolicyClient``?  Because
Isaac-GR00T pins Python 3.12 + torch 2.7 + transformers in its own uv
environment, and dragging that into the rclpy process is both heavy and prone
to ABI clashes with the system numpy that ROS uses.  The client half of the
protocol is only ZeroMQ + msgpack, so it is reimplemented here.

Wire protocol (matches ``gr00t/policy/server_client.py``):
    * ZeroMQ REQ (client) <-> REP (server)
    * msgpack payloads with the msgpack-numpy ndarray extension
    * request: ``{"endpoint": str, "data": Any, "api_token": str | None}``
    * endpoints: ping, get_action, get_modality_config, reset, kill

If the real ``msgpack_numpy`` package is installed it is used verbatim;
otherwise the encoder below reproduces its format exactly.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    import msgpack
except ImportError as exc:  # pragma: no cover - dependency check
    raise ImportError(
        "groot_vla needs msgpack. Install it with:\n"
        "  sudo apt install python3-msgpack\n"
        "  # or: pip install --break-system-packages msgpack"
    ) from exc

try:
    import zmq
except ImportError as exc:  # pragma: no cover - dependency check
    raise ImportError(
        "groot_vla needs pyzmq. Install it with:\n"
        "  sudo apt install python3-zmq"
    ) from exc

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# msgpack-numpy compatible codec
# --------------------------------------------------------------------------- #
try:
    import msgpack_numpy as _mnp

    _encode_hook = _mnp.encode
    _decode_hook = _mnp.decode
    USING_REAL_MSGPACK_NUMPY = True
except ImportError:
    USING_REAL_MSGPACK_NUMPY = False

    def _encode_hook(obj: Any) -> Any:
        """Serialize numpy scalars/arrays the way msgpack_numpy.encode does."""
        if isinstance(obj, np.ndarray):
            kind = b""
            descr = obj.dtype.str
            if obj.dtype.kind == "V":  # structured array
                kind = b"V"
                descr = obj.dtype.descr
            return {
                b"nd": True,
                b"type": descr,
                b"kind": kind,
                b"shape": obj.shape,
                # tobytes() forces a C-contiguous copy, which the format requires
                b"data": obj.tobytes(),
            }
        if isinstance(obj, (np.bool_, np.number)):
            return {b"nd": False, b"type": obj.dtype.str, b"data": obj.tobytes()}
        if isinstance(obj, complex):
            return {b"complex": True, b"data": repr(obj)}
        return obj

    def _decode_hook(obj: Any) -> Any:
        """Inverse of :func:`_encode_hook`."""
        try:
            if b"nd" in obj:
                if obj[b"nd"] is True:
                    descr = obj[b"type"]
                    if obj.get(b"kind") == b"V":
                        descr = [
                            tuple(
                                t.decode() if isinstance(t, bytes) else t for t in field
                            )
                            for field in descr
                        ]
                    elif isinstance(descr, bytes):
                        descr = descr.decode()
                    return np.frombuffer(obj[b"data"], dtype=np.dtype(descr)).reshape(
                        obj[b"shape"]
                    )
                descr = obj[b"type"]
                if isinstance(descr, bytes):
                    descr = descr.decode()
                return np.frombuffer(obj[b"data"], dtype=np.dtype(descr))[0]
            if b"complex" in obj:
                data = obj[b"data"]
                return complex(data.decode() if isinstance(data, bytes) else data)
        except (KeyError, TypeError):
            pass
        return obj


def packb(obj: Any) -> bytes:
    return msgpack.packb(obj, default=_encode_hook, use_bin_type=True)


def unpackb(payload: bytes) -> Any:
    return msgpack.unpackb(payload, object_hook=_decode_hook, raw=False)


class PolicyServerError(RuntimeError):
    """The server was reachable but returned an error, or did not answer."""


# --------------------------------------------------------------------------- #
# client
# --------------------------------------------------------------------------- #
class GrootClient:
    """Blocking request/reply client for a GR00T inference server.

    A REQ socket enforces strict send/recv alternation, and a timed-out recv
    leaves it permanently wedged.  Every timeout therefore tears the socket
    down and rebuilds it, which is what :meth:`_reset_socket` is for.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self._context = zmq.Context.instance()
        self._socket: zmq.Socket | None = None
        self._reset_socket()

    # -- lifecycle ---------------------------------------------------------- #
    @property
    def endpoint(self) -> str:
        return f"tcp://{self.host}:{self.port}"

    def _reset_socket(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
        socket = self._context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        # Drop queued messages instantly on close; without this a dead server
        # keeps the process alive at shutdown.
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.endpoint)
        self._socket = socket

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None

    def __enter__(self) -> "GrootClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- protocol ----------------------------------------------------------- #
    def call_endpoint(
        self, endpoint: str, data: Any = None, requires_input: bool = True
    ) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token is not None:
            request["api_token"] = self.api_token

        assert self._socket is not None
        try:
            self._socket.send(packb(request))
            reply = unpackb(self._socket.recv())
        except zmq.Again as exc:
            self._reset_socket()
            raise PolicyServerError(
                f"no reply from GR00T server at {self.endpoint} "
                f"within {self.timeout_ms} ms"
            ) from exc
        except zmq.ZMQError as exc:
            self._reset_socket()
            raise PolicyServerError(f"ZMQ error talking to {self.endpoint}: {exc}") from exc

        if isinstance(reply, dict) and "error" in reply:
            raise PolicyServerError(str(reply["error"]))
        return reply

    # -- endpoints ---------------------------------------------------------- #
    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except PolicyServerError:
            return False

    def get_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Run one inference step. Returns the raw action dict from the server."""
        reply = self.call_endpoint("get_action", observation)
        # Newer servers answer (action, info); older ones answer just action.
        if isinstance(reply, (tuple, list)) and len(reply) == 2:
            action, _info = reply
            return action
        return reply

    def get_modality_config(self) -> Any:
        return self.call_endpoint("get_modality_config", requires_input=False)

    def reset(self, options: dict[str, Any] | None = None) -> Any:
        return self.call_endpoint("reset", options or {})

    def kill_server(self) -> None:
        try:
            self.call_endpoint("kill", requires_input=False)
        except PolicyServerError:
            pass  # a server that dies before replying is the expected case
