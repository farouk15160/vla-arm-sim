"""Qt control panel for the whole VLA cell.

Everything you would otherwise type into three terminals, in one window:
policy arm/halt, the task instruction, scene reset, the scripted baseline,
gripper jogging, named poses, and a live status readout.

    ros2 run groot_vla control_gui

Design notes
------------
* Every ROS call that can block (planning, execution, a slow inference) runs on
  a worker thread. Qt's event loop must never wait on MoveIt or the window
  freezes mid-motion and looks crashed.
* Buttons that command motion are disabled while a motion is in flight, so a
  double-click cannot queue two trajectories.
* The panel is a pure client: it owns no policy state. Everything it shows is
  read from /groot_policy/status, so it stays correct even if you also drive
  the system from the command line.
"""

from __future__ import annotations

import json
import sys
import threading
from functools import partial

import rclpy
from python_qt_binding.QtCore import Qt, QTimer, Signal
from python_qt_binding.QtGui import QFont
from python_qt_binding.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger

from groot_vla.moveit_helper import MoveItError, MoveItHelper

NAMED_POSES = {
    "home": {
        "shoulder_pan_joint": 0.0, "shoulder_lift_joint": -1.5708, "elbow_joint": 1.5708,
        "wrist_1_joint": -1.5708, "wrist_2_joint": -1.5708, "wrist_3_joint": 0.0,
    },
    "observe": {
        "shoulder_pan_joint": 0.0, "shoulder_lift_joint": -1.2217, "elbow_joint": 1.2217,
        "wrist_1_joint": -1.5708, "wrist_2_joint": -1.5708, "wrist_3_joint": 0.0,
    },
    "up": {
        "shoulder_pan_joint": 0.0, "shoulder_lift_joint": -1.5708, "elbow_joint": 0.0,
        "wrist_1_joint": -1.5708, "wrist_2_joint": 0.0, "wrist_3_joint": 0.0,
    },
}

TASKS = [
    "pick up the red cube and place it in the tray",
    "pick up the green cube and place it in the tray",
    "pick up the blue cube and place it in the tray",
    "move the red cube to the left",
    "put all the cubes in the tray",
]


class GuiNode(Node):
    """ROS side of the panel. Holds no Qt references."""

    def __init__(self) -> None:
        super().__init__("control_gui")
        latching = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.status: dict = {}
        self.create_subscription(String, "/groot_policy/status", self._on_status, latching)

        self.enable_client = self.create_client(SetBool, "/groot_policy/enable")
        self.halt_client = self.create_client(Trigger, "/groot_policy/halt")
        self.reset_policy_client = self.create_client(Trigger, "/groot_policy/reset_policy")
        self.instruction_pub = self.create_publisher(String, "/groot_policy/instruction", 10)

        # MoveItHelper owns a SingleThreadedExecutor and spins it inside its
        # blocking calls. That node must therefore NOT also belong to this
        # panel's MultiThreadedExecutor: a node can only be added to one
        # executor, and spinning the same one from two threads raises
        # "Executor is already spinning". A dedicated node keeps them apart.
        self.moveit_node = rclpy.create_node("control_gui_moveit")
        self.moveit = MoveItHelper(self.moveit_node)
        self._moveit_ready = False

    def _on_status(self, message: String) -> None:
        try:
            self.status = json.loads(message.data)
        except json.JSONDecodeError:
            pass

    def ensure_moveit(self) -> None:
        if not self._moveit_ready:
            self.moveit.wait_for_services(timeout=30.0)
            self._moveit_ready = True


class ControlPanel(QWidget):
    log_signal = Signal(str)
    done_signal = Signal()

    def __init__(self, node: GuiNode) -> None:
        super().__init__()
        self.node = node
        self._busy = threading.Lock()
        self.setWindowTitle("VLA Arm Control")
        self.resize(560, 720)

        self.log_signal.connect(self._append_log)
        self.done_signal.connect(self._refresh_enabled)

        layout = QVBoxLayout(self)
        layout.addWidget(self._policy_group())
        layout.addWidget(self._task_group())
        layout.addWidget(self._motion_group())
        layout.addWidget(self._status_group())

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(400)  # bounded, so a long session cannot grow forever
        self._log.setFont(QFont("monospace", 9))
        layout.addWidget(self._log, stretch=1)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_status)
        self._timer.start(500)
        self.log("panel ready")

    # ------------------------------------------------------------------ #
    def _policy_group(self) -> QGroupBox:
        box = QGroupBox("Policy")
        grid = QGridLayout(box)

        self.btn_enable = QPushButton("Arm policy")
        self.btn_enable.setStyleSheet("background:#2d7d2d; color:white; font-weight:bold;")
        self.btn_enable.clicked.connect(lambda: self._set_policy(True))
        grid.addWidget(self.btn_enable, 0, 0)

        self.btn_disable = QPushButton("Disable")
        self.btn_disable.clicked.connect(lambda: self._set_policy(False))
        grid.addWidget(self.btn_disable, 0, 1)

        btn_halt = QPushButton("HALT")
        btn_halt.setStyleSheet("background:#b02020; color:white; font-weight:bold;")
        btn_halt.clicked.connect(self._halt)
        grid.addWidget(btn_halt, 0, 2)

        btn_reset_policy = QPushButton("Reset policy state")
        btn_reset_policy.clicked.connect(self._reset_policy)
        grid.addWidget(btn_reset_policy, 1, 0, 1, 3)
        return box

    def _task_group(self) -> QGroupBox:
        box = QGroupBox("Task instruction")
        outer = QVBoxLayout(box)

        self.task_combo = QComboBox()
        self.task_combo.addItems(TASKS)
        self.task_combo.setEditable(True)
        outer.addWidget(self.task_combo)

        row = QHBoxLayout()
        btn_send = QPushButton("Send instruction")
        btn_send.clicked.connect(self._send_instruction)
        row.addWidget(btn_send)
        outer.addLayout(row)
        return box

    def _motion_group(self) -> QGroupBox:
        box = QGroupBox("Manual control (MoveIt)")
        grid = QGridLayout(box)

        for column, name in enumerate(NAMED_POSES):
            button = QPushButton(f"Go {name}")
            button.clicked.connect(partial(self._go_named, name))
            grid.addWidget(button, 0, column)

        btn_open = QPushButton("Open gripper")
        btn_open.clicked.connect(lambda: self._run("open gripper", self.node.moveit.open_gripper))
        grid.addWidget(btn_open, 1, 0)

        btn_close = QPushButton("Close gripper")
        btn_close.clicked.connect(lambda: self._run("close gripper", self.node.moveit.close_gripper))
        grid.addWidget(btn_close, 1, 1)

        btn_scene = QPushButton("Reset scene")
        btn_scene.clicked.connect(self._reset_scene)
        grid.addWidget(btn_scene, 1, 2)

        self.cube_combo = QComboBox()
        self.cube_combo.addItems(["red_cube", "green_cube", "blue_cube"])
        grid.addWidget(self.cube_combo, 2, 0)

        btn_demo = QPushButton("Run scripted pick && place")
        btn_demo.clicked.connect(self._run_demo)
        grid.addWidget(btn_demo, 2, 1, 1, 2)

        self.motion_buttons = [
            btn_open, btn_close, btn_demo,
            *[grid.itemAtPosition(0, c).widget() for c in range(len(NAMED_POSES))],
        ]
        return box

    def _status_group(self) -> QGroupBox:
        box = QGroupBox("Status")
        grid = QGridLayout(box)
        self.lbl_state = QLabel("policy: unknown")
        self.lbl_state.setFont(QFont("monospace", 10, QFont.Bold))
        grid.addWidget(self.lbl_state, 0, 0, 1, 2)
        self.lbl_server = QLabel("server: -")
        grid.addWidget(self.lbl_server, 1, 0, 1, 2)
        self.lbl_counts = QLabel("inferences: -   failures: -   latency: -")
        grid.addWidget(self.lbl_counts, 2, 0, 1, 2)
        self.lbl_error = QLabel("")
        self.lbl_error.setWordWrap(True)
        self.lbl_error.setStyleSheet("color:#c04040;")
        grid.addWidget(self.lbl_error, 3, 0, 1, 2)
        return box

    # ------------------------------------------------------------------ #
    def log(self, text: str) -> None:
        self.log_signal.emit(text)

    def _append_log(self, text: str) -> None:
        self._log.appendPlainText(text)

    def _refresh_status(self) -> None:
        status = self.node.status
        if not status:
            self.lbl_state.setText("policy: no status yet (is policy_node running?)")
            return
        enabled = status.get("enabled", False)
        self.lbl_state.setText(f"policy: {'ARMED' if enabled else 'disabled'}")
        self.lbl_state.setStyleSheet(
            "color:#2d7d2d;" if enabled else "color:#808080;"
        )
        self.lbl_server.setText(
            f"server: {status.get('server','-')}   space: {status.get('action_space','-')}"
        )
        self.lbl_counts.setText(
            f"inferences: {status.get('inferences','-')}   "
            f"failures: {status.get('failures','-')}   "
            f"latency: {status.get('last_latency_ms','-')} ms"
        )
        self.lbl_error.setText(status.get("last_error", "") or "")

    def _refresh_enabled(self) -> None:
        idle = not self._busy.locked()
        for button in self.motion_buttons:
            button.setEnabled(idle)

    # ------------------------------------------------------------------ #
    def _run(self, label: str, action) -> None:
        """Run a blocking call on a worker thread; refuse if one is running."""
        if not self._busy.acquire(blocking=False):
            self.log(f"busy, ignoring: {label}")
            return
        self._refresh_enabled()

        def worker() -> None:
            try:
                self.log(f"{label} ...")
                self.node.ensure_moveit()
                action()
                self.log(f"{label}: done")
            except MoveItError as exc:
                self.log(f"{label}: FAILED - {exc}")
            except Exception as exc:  # noqa: BLE001 - never kill the worker
                self.log(f"{label}: error - {type(exc).__name__}: {exc}")
            finally:
                self._busy.release()
                self.done_signal.emit()

        threading.Thread(target=worker, daemon=True, name="gui_worker").start()

    def _go_named(self, name: str) -> None:
        self._run(f"go {name}", lambda: self.node.moveit.move_to_joints(NAMED_POSES[name]))

    def _set_policy(self, enable: bool) -> None:
        client = self.node.enable_client
        if not client.service_is_ready():
            self.log("/groot_policy/enable unavailable - is policy_node running?")
            return
        request = SetBool.Request()
        request.data = enable
        future = client.call_async(request)

        def on_done(fut) -> None:
            try:
                response = fut.result()
                self.log(f"enable({enable}): {response.message}")
            except Exception as exc:  # noqa: BLE001
                self.log(f"enable failed: {exc}")

        future.add_done_callback(on_done)

    def _halt(self) -> None:
        if not self.node.halt_client.service_is_ready():
            self.log("/groot_policy/halt unavailable")
            return
        self.node.halt_client.call_async(Trigger.Request())
        self.log("HALT sent")

    def _reset_policy(self) -> None:
        if not self.node.reset_policy_client.service_is_ready():
            self.log("/groot_policy/reset_policy unavailable")
            return
        self.node.reset_policy_client.call_async(Trigger.Request())
        self.log("policy state reset requested")

    def _send_instruction(self) -> None:
        text = self.task_combo.currentText().strip()
        if not text:
            return
        self.node.instruction_pub.publish(String(data=text))
        self.log(f"instruction -> {text!r}")

    def _reset_scene(self) -> None:
        import subprocess

        def action() -> None:
            subprocess.run(["ros2", "run", "groot_vla", "scene_reset"],
                           capture_output=True, timeout=60)

        self._run("reset scene", action)

    def _run_demo(self) -> None:
        import subprocess

        cube = self.cube_combo.currentText()

        def action() -> None:
            result = subprocess.run(
                ["ros2", "run", "groot_vla", "pick_place_demo",
                 "--ros-args", "-p", f"cube:={cube}", "-p", "use_sim_time:=true"],
                capture_output=True, text=True, timeout=300,
            )
            tail = [line for line in result.stdout.splitlines() if line.strip()][-1:]
            if tail:
                self.log(tail[0])

        self._run(f"pick and place {cube}", action)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = GuiNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True, name="ros_spin")
    spin_thread.start()

    app = QApplication(sys.argv[:1])
    panel = ControlPanel(node)
    panel.show()
    try:
        code = app.exec_()
    finally:
        executor.shutdown()
        node.moveit_node.destroy_node()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()
