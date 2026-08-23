"""Qt control panel for the whole VLA cell.

Everything you would otherwise type into three terminals, in one window:
policy arm/halt, the task instruction, the RViz goal marker, scene reset, the
scripted baseline, gripper jogging, named poses, a live view of what the policy
is emitting, and MoveIt's status.

    ros2 run groot_vla control_gui

Design notes
------------
* Every ROS call that can block (planning, execution, a slow inference) runs on
  a worker thread. Qt's event loop must never wait on MoveIt or the window
  freezes mid-motion and looks crashed.
* Buttons that command motion are disabled while a motion is in flight, so a
  double-click cannot queue two trajectories.
* The panel is a pure client: it owns no policy state. Everything it shows is
  read from topics, so it stays correct even if you also drive the system from
  the command line or from RViz.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from functools import partial

import rclpy
from geometry_msgs.msg import PoseStamped
from python_qt_binding.QtCore import Qt, QTimer, Signal
from python_qt_binding.QtGui import QFont
from python_qt_binding.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from rclpy.callback_groups import ReentrantCallbackGroup
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

STYLE = """
QGroupBox {
    font-weight: bold;
    border: 1px solid #c0c4c8;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px 0 5px;
    color: #33383d;
}
QPushButton {
    padding: 7px 10px;
    border: 1px solid #b6bbc0;
    border-radius: 4px;
    background: #fbfbfc;
}
QPushButton:hover  { background: #eef2f6; }
QPushButton:pressed{ background: #e0e6ec; }
QPushButton:disabled { color: #a0a4a8; background: #f2f2f3; }
"""


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
        self.action: dict = {}
        self.action_stamp = 0.0
        self.goal_pose: PoseStamped | None = None
        self.marker_status = ""

        self.create_subscription(String, "/groot_policy/status", self._on_status, latching)
        self.create_subscription(String, "/groot_policy/action", self._on_action, 10)
        self.create_subscription(String, "/goal_marker/status", self._on_marker_status, 10)
        self.create_subscription(PoseStamped, "/goal_marker/goal_pose", self._on_goal_pose, 10)

        group = ReentrantCallbackGroup()
        self.enable_client = self.create_client(SetBool, "/groot_policy/enable", callback_group=group)
        self.halt_client = self.create_client(Trigger, "/groot_policy/halt", callback_group=group)
        self.reset_policy_client = self.create_client(
            Trigger, "/groot_policy/reset_policy", callback_group=group)
        self.go_marker_client = self.create_client(
            Trigger, "/goal_marker/go_to_marker", callback_group=group)
        self.instruction_pub = self.create_publisher(String, "/groot_policy/instruction", 10)

        # MoveItHelper spins its own executor inside blocking calls, so its node
        # must not also belong to this panel's MultiThreadedExecutor.
        self.moveit_node = rclpy.create_node("control_gui_moveit")
        self.moveit = MoveItHelper(self.moveit_node)
        self._moveit_ready = False

    def _on_status(self, message: String) -> None:
        try:
            self.status = json.loads(message.data)
        except json.JSONDecodeError:
            pass

    def _on_action(self, message: String) -> None:
        try:
            self.action = json.loads(message.data)
            self.action_stamp = time.monotonic()
        except json.JSONDecodeError:
            pass

    def _on_marker_status(self, message: String) -> None:
        self.marker_status = message.data

    def _on_goal_pose(self, message: PoseStamped) -> None:
        self.goal_pose = message

    def moveit_available(self) -> bool:
        """True when move_group is advertising the action we plan through."""
        return self.moveit._move_group.server_is_ready()

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
        self.setStyleSheet(STYLE)
        self.resize(640, 900)

        self.log_signal.connect(self._append_log)
        self.done_signal.connect(self._refresh_enabled)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.addWidget(self._status_group())
        layout.addWidget(self._policy_group())
        layout.addWidget(self._vla_group())
        layout.addWidget(self._marker_group())
        layout.addWidget(self._motion_group())

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(400)  # bounded; a long session cannot grow forever
        self._log.setFont(QFont("monospace", 9))
        self._log.setMinimumHeight(110)
        layout.addWidget(self._log, stretch=1)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(400)
        self.log("panel ready")

    # ------------------------------------------------------------------ #
    # groups
    # ------------------------------------------------------------------ #
    def _status_group(self) -> QGroupBox:
        box = QGroupBox("System")
        grid = QGridLayout(box)

        self.lbl_policy_state = QLabel("policy: waiting…")
        self.lbl_policy_state.setFont(QFont("sans", 12, QFont.Bold))
        grid.addWidget(self.lbl_policy_state, 0, 0, 1, 2)

        self.lbl_moveit = QLabel("MoveIt: checking…")
        self.lbl_moveit.setFont(QFont("monospace", 9))
        grid.addWidget(self.lbl_moveit, 1, 0, 1, 2)

        self.lbl_server = QLabel("server: -")
        self.lbl_server.setFont(QFont("monospace", 9))
        grid.addWidget(self.lbl_server, 2, 0, 1, 2)

        self.lbl_counts = QLabel("inferences: -   failures: -   latency: -")
        self.lbl_counts.setFont(QFont("monospace", 9))
        grid.addWidget(self.lbl_counts, 3, 0, 1, 2)

        self.lbl_error = QLabel("")
        self.lbl_error.setWordWrap(True)
        self.lbl_error.setStyleSheet("color:#b02020;")
        grid.addWidget(self.lbl_error, 4, 0, 1, 2)
        return box

    def _policy_group(self) -> QGroupBox:
        box = QGroupBox("Policy")
        outer = QVBoxLayout(box)

        row = QHBoxLayout()
        self.btn_enable = QPushButton("▶  Arm policy")
        self.btn_enable.setStyleSheet(
            "background:#2d7d2d; color:white; font-weight:bold; border:1px solid #216021;")
        self.btn_enable.clicked.connect(lambda: self._set_policy(True))
        row.addWidget(self.btn_enable, 2)

        self.btn_disable = QPushButton("Disable")
        self.btn_disable.clicked.connect(lambda: self._set_policy(False))
        row.addWidget(self.btn_disable, 1)

        btn_halt = QPushButton("■  HALT")
        btn_halt.setStyleSheet(
            "background:#b02020; color:white; font-weight:bold; border:1px solid #8a1919;")
        btn_halt.clicked.connect(self._halt)
        row.addWidget(btn_halt, 1)
        outer.addLayout(row)

        self.task_combo = QComboBox()
        self.task_combo.addItems(TASKS)
        self.task_combo.setEditable(True)
        outer.addWidget(self.task_combo)

        row2 = QHBoxLayout()
        btn_send = QPushButton("Send instruction")
        btn_send.clicked.connect(self._send_instruction)
        row2.addWidget(btn_send, 2)
        btn_reset_policy = QPushButton("Reset policy state")
        btn_reset_policy.clicked.connect(self._reset_policy)
        row2.addWidget(btn_reset_policy, 1)
        outer.addLayout(row2)
        return box

    def _vla_group(self) -> QGroupBox:
        """Live view of what the policy is emitting."""
        box = QGroupBox("VLA output")
        grid = QGridLayout(box)

        self.lbl_vla_head = QLabel("idle — arm the policy to see its output")
        self.lbl_vla_head.setFont(QFont("monospace", 9))
        grid.addWidget(self.lbl_vla_head, 0, 0, 1, 7)

        # One bar per arm DoF plus the gripper. Bars beat raw numbers here:
        # you can see at a glance whether the policy is saturating a joint.
        self.vla_bars: list[QProgressBar] = []
        self.vla_labels: list[QLabel] = []
        names = ["J1", "J2", "J3", "J4", "J5", "J6", "grip"]
        for column, name in enumerate(names):
            caption = QLabel(name)
            caption.setAlignment(Qt.AlignCenter)
            caption.setFont(QFont("monospace", 8))
            grid.addWidget(caption, 1, column)

            bar = QProgressBar()
            bar.setOrientation(Qt.Vertical)
            bar.setRange(-1000, 1000)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(70)
            grid.addWidget(bar, 2, column, Qt.AlignHCenter)
            self.vla_bars.append(bar)

            value = QLabel("–")
            value.setAlignment(Qt.AlignCenter)
            value.setFont(QFont("monospace", 8))
            grid.addWidget(value, 3, column)
            self.vla_labels.append(value)

        self.lbl_vla_detail = QLabel("")
        self.lbl_vla_detail.setFont(QFont("monospace", 8))
        self.lbl_vla_detail.setStyleSheet("color:#556;")
        self.lbl_vla_detail.setWordWrap(True)
        grid.addWidget(self.lbl_vla_detail, 4, 0, 1, 7)
        return box

    def _marker_group(self) -> QGroupBox:
        box = QGroupBox("RViz goal marker")
        outer = QVBoxLayout(box)

        self.lbl_marker = QLabel("marker: waiting for /goal_marker …")
        self.lbl_marker.setFont(QFont("monospace", 9))
        outer.addWidget(self.lbl_marker)

        row = QHBoxLayout()
        self.btn_go = QPushButton("⇒  GO TO MARKER")
        self.btn_go.setStyleSheet(
            "background:#1f6fb2; color:white; font-weight:bold; border:1px solid #17558a;")
        self.btn_go.clicked.connect(self._go_to_marker)
        row.addWidget(self.btn_go, 2)

        btn_hint = QPushButton("How?")
        btn_hint.clicked.connect(lambda: self.log(
            "Drag the cyan sphere in RViz (arrows move, rings rotate), then press "
            "GO TO MARKER — or right-click the marker and choose 'Move here'."))
        row.addWidget(btn_hint, 1)
        outer.addLayout(row)

        self.lbl_marker_status = QLabel("")
        self.lbl_marker_status.setFont(QFont("monospace", 8))
        self.lbl_marker_status.setWordWrap(True)
        outer.addWidget(self.lbl_marker_status)
        return box

    def _motion_group(self) -> QGroupBox:
        box = QGroupBox("Manual control (MoveIt)")
        grid = QGridLayout(box)

        pose_buttons = []
        for column, name in enumerate(NAMED_POSES):
            button = QPushButton(f"Go {name}")
            button.clicked.connect(partial(self._go_named, name))
            grid.addWidget(button, 0, column)
            pose_buttons.append(button)

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

        self.motion_buttons = [btn_open, btn_close, btn_demo, *pose_buttons]
        return box

    # ------------------------------------------------------------------ #
    # refresh
    # ------------------------------------------------------------------ #
    def log(self, text: str) -> None:
        self.log_signal.emit(text)

    def _append_log(self, text: str) -> None:
        self._log.appendPlainText(text)

    def _refresh(self) -> None:
        self._refresh_status()
        self._refresh_vla()
        self._refresh_marker()

    def _refresh_status(self) -> None:
        moveit_up = self.node.moveit_available()
        self.lbl_moveit.setText(
            f"MoveIt: {'connected (/move_action)' if moveit_up else 'NOT reachable'}"
        )
        self.lbl_moveit.setStyleSheet("color:#2d7d2d;" if moveit_up else "color:#b02020;")
        for button in self.motion_buttons:
            button.setToolTip("" if moveit_up else "move_group is not running")

        status = self.node.status
        if not status:
            self.lbl_policy_state.setText("policy: no status (policy_node not running)")
            self.lbl_policy_state.setStyleSheet("color:#808080;")
            return
        enabled = status.get("enabled", False)
        self.lbl_policy_state.setText(f"policy: {'ARMED' if enabled else 'disabled'}")
        self.lbl_policy_state.setStyleSheet(
            "color:#2d7d2d;" if enabled else "color:#808080;")
        self.lbl_server.setText(
            f"server: {status.get('server','-')}   space: {status.get('action_space','-')}")
        self.lbl_counts.setText(
            f"inferences: {status.get('inferences','-')}   "
            f"failures: {status.get('failures','-')}   "
            f"latency: {status.get('last_latency_ms','-')} ms")
        self.lbl_error.setText(status.get("last_error", "") or "")

    def _refresh_vla(self) -> None:
        action = self.node.action
        if not action:
            return
        age = time.monotonic() - self.node.action_stamp
        # Grey out once the policy stops producing, so a frozen readout is not
        # mistaken for a live one.
        stale = age > 5.0
        space = action.get("action_space", "?")
        self.lbl_vla_head.setText(
            f"{'STALE' if stale else 'LIVE'}  space={space}  "
            f"chunk={action.get('chunk_len','?')}  "
            f"{action.get('latency_ms','?')} ms  ({age:.1f}s ago)")
        self.lbl_vla_head.setStyleSheet("color:#808080;" if stale else "color:#1f6fb2;")

        arm = action.get("arm", [])
        # Scale for the bars: joint targets are radians (±3.2), Cartesian
        # deltas are m/s and rad/s (±0.3). Different ranges, same widget.
        span = 3.2 if space == "joint_position" else 0.5
        values = list(arm) + [action.get("gripper")]
        for index, bar in enumerate(self.vla_bars):
            value = values[index] if index < len(values) else None
            if value is None:
                bar.setValue(0)
                self.vla_labels[index].setText("–")
                continue
            scale = 1.0 if index == 6 else span
            bar.setValue(int(max(-1.0, min(1.0, float(value) / scale)) * 1000))
            self.vla_labels[index].setText(f"{float(value):+.2f}")

        detail = ""
        if action.get("delta_from_current"):
            deltas = action["delta_from_current"]
            biggest = max(abs(v) for v in deltas) if deltas else 0.0
            detail = "Δ from current: " + " ".join(f"{v:+.3f}" for v in deltas)
            detail += f"   (max {biggest:.3f} rad)"
        self.lbl_vla_detail.setText(detail)

    def _refresh_marker(self) -> None:
        pose = self.node.goal_pose
        available = self.node.go_marker_client.service_is_ready()
        self.btn_go.setEnabled(available and not self._busy.locked())
        if pose is None:
            self.lbl_marker.setText("marker: waiting for /goal_marker …")
            return
        p = pose.pose.position
        self.lbl_marker.setText(
            f"marker: x={p.x:+.3f}  y={p.y:+.3f}  z={p.z:+.3f}   (frame {pose.header.frame_id})")
        self.lbl_marker_status.setText(self.node.marker_status)

    def _refresh_enabled(self) -> None:
        idle = not self._busy.locked()
        for button in self.motion_buttons:
            button.setEnabled(idle)

    # ------------------------------------------------------------------ #
    # actions
    # ------------------------------------------------------------------ #
    def _run(self, label: str, action) -> None:
        """Run a blocking call on a worker thread; refuse if one is running."""
        if not self._busy.acquire(blocking=False):
            self.log(f"busy, ignoring: {label}")
            return
        self._refresh_enabled()

        def worker() -> None:
            try:
                self.log(f"{label} …")
                self.node.ensure_moveit()
                action()
                self.log(f"{label}: done")
            except MoveItError as exc:
                self.log(f"{label}: FAILED — {exc}")
            except Exception as exc:  # noqa: BLE001 - never kill the worker
                self.log(f"{label}: error — {type(exc).__name__}: {exc}")
            finally:
                self._busy.release()
                self.done_signal.emit()

        threading.Thread(target=worker, daemon=True, name="gui_worker").start()

    def _go_named(self, name: str) -> None:
        self._run(f"go {name}", lambda: self.node.moveit.move_to_joints(NAMED_POSES[name]))

    def _go_to_marker(self) -> None:
        """Ask goal_marker to move to wherever the marker sits."""
        client = self.node.go_marker_client
        if not client.service_is_ready():
            self.log("/goal_marker/go_to_marker unavailable — is goal_marker running?")
            return
        future = client.call_async(Trigger.Request())

        def on_done(fut) -> None:
            try:
                response = fut.result()
                self.log(f"go to marker: {response.message}")
            except Exception as exc:  # noqa: BLE001
                self.log(f"go to marker failed: {exc}")

        future.add_done_callback(on_done)

    def _set_policy(self, enable: bool) -> None:
        client = self.node.enable_client
        if not client.service_is_ready():
            self.log("/groot_policy/enable unavailable — is policy_node running?")
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
        def action() -> None:
            subprocess.run(["ros2", "run", "groot_vla", "scene_reset"],
                           capture_output=True, timeout=60)

        self._run("reset scene", action)

    def _run_demo(self) -> None:
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
    threading.Thread(target=executor.spin, daemon=True, name="ros_spin").start()

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
