# System Documentation

A simulated UR5e manipulation cell for developing and evaluating
vision-language-action (VLA) policies under ROS 2 Jazzy.

This document describes what the system *is* and *why it is built this way*.
For commands, see `README.md`.

**Contents**

1. [Scope and design thesis](#1-scope-and-design-thesis)
2. [Architecture](#2-architecture)
3. [Node and topic graph](#3-node-and-topic-graph)
4. [Control paths: with and without the VLA](#4-control-paths-with-and-without-the-vla)
5. [SmolVLA: the policy](#5-smolvla-the-policy)
6. [The dataset and training procedure](#6-the-dataset-and-training-procedure)
7. [Unity Robotics and the ROS-TCP data flow](#7-unity-robotics-and-the-ros-tcp-data-flow)
8. [Constraints and safety envelope](#8-constraints-and-safety-envelope)
9. [Inspecting a running system](#9-inspecting-a-running-system)
10. [Measured performance](#10-measured-performance)
11. [Known limitations](#11-known-limitations)

---

## 1. Scope and design thesis

The system exists to answer one question empirically: *can a general-purpose VLA
policy be made to perform a manipulation task on a specific robot, and how would
we know?*

Three commitments follow from that, and they explain most of the structure.

**The ROS side is a policy client, not a model host.** Policies run as separate
processes behind a ZeroMQ socket. Swapping SmolVLA for OpenVLA, or for a
scripted stand-in, changes which server is running and nothing else. This is not
incidental: it is what makes the comparison meaningful, because every policy sees
byte-identical observations through the same code path.

**The measurement apparatus is part of the system.** A scripted MoveIt
pick-and-place is maintained alongside the policy, not as scaffolding but as the
control condition. Without a competent baseline on the same hardware, a policy
failure cannot be attributed between the policy, the controller, the grasp
geometry, or the simulator.

**Nothing is claimed that was not observed.** Every performance number in this
document was measured on the target machine (RTX 2060, 6 GB; 16 cores; 14 GB
RAM). Several sections document *negative* results, because a failure with a
known cause is more useful than a silent one.

---

## 2. Architecture

```
                        ┌──────────────────────────────────────┐
                        │           POLICY PROCESS             │
                        │  (its own venv: torch, transformers) │
                        │                                      │
                        │   mock  │  SmolVLA  │  OpenVLA       │
                        └──────────────┬───────────────────────┘
                                       │ ZeroMQ REQ/REP
                                       │ msgpack + ndarray ext
                                       │ observation ──▶ ◀── action chunk
┌──────────────────────────────────────┴───────────────────────────────────┐
│                              ROS 2 GRAPH                                  │
│                                                                           │
│   cameras ──▶ policy_node ──▶ arm_controller ──▶ ┌────────────────────┐  │
│   joint_states                gripper_controller │  SIMULATOR         │  │
│                                                  │  Gazebo  or  Unity │  │
│   move_group ──────────────▶ (same controllers)  └────────────────────┘  │
│      ▲                                                                    │
│      │                                                                    │
│   goal_marker, control_gui, pick_place_demo                              │
└───────────────────────────────────────────────────────────────────────────┘
```

### Why the policy is out of process

Three independent reasons, each sufficient on its own:

1. **Dependency isolation.** Isaac-GR00T pins Python 3.12 + torch 2.7;
   OpenVLA pins `transformers==4.40`; LeRobot requires `transformers>=4.57`.
   These cannot coexist in one interpreter. They are separate venvs precisely
   because the constraint is unresolvable, not merely inconvenient.
2. **Locality is a deployment parameter.** A 3B model needing 16 GB runs on
   another machine by changing `policy_host`. No code path differs.
3. **Failure isolation.** A policy that crashes, hangs, or emits NaNs cannot
   take the robot control stack with it.

### Packages

| Package | Responsibility |
|---|---|
| `groot_arm_description` | URDF/xacro (UR5e + gripper + cameras), Gazebo world, `ros2_control` config, materials, meshes |
| `groot_arm_moveit_config` | SRDF, kinematics, OMPL/Pilz pipelines, `move_group`, MoveIt Servo |
| `groot_arm_bringup` | Launch composition; `system.launch.py` is the single entry point |
| `groot_vla` | Policy client and node, three policy servers, observation/action translation, MoveIt helper, GUI, RViz integration, data collection and export |
| `ros_tcp_endpoint` | Unity bridge (fetched, not vendored) |

---

## 3. Node and topic graph

Captured from a running system (`policy:=mock`, headless), with transient and
introspection nodes elided.

### Nodes

| Node | Package | Role |
|---|---|---|
| `/gz_ros_control` | `gz_ros2_control` | `controller_manager` inside Gazebo; owns the hardware interfaces |
| `/controller_manager` | `controller_manager` | Loads, configures and switches controllers |
| `/joint_state_broadcaster` | `ros2_controllers` | Publishes `/joint_states` at 500 Hz |
| `/arm_controller` | `joint_trajectory_controller` | 6 arm joints, position interface |
| `/gripper_controller` | `joint_trajectory_controller` | 2 finger joints |
| `/forward_position_controller` | `position_controllers` | Loaded **inactive**; alternative streaming path |
| `/robot_state_publisher` | `robot_state_publisher` | URDF → TF |
| `/move_group` | `moveit_ros_move_group` | Planning, IK, trajectory execution |
| `/moveit_simple_controller_manager` | `moveit_plugins` | Maps planning groups → controllers |
| `/gz_bridge` | `ros_gz_bridge` | `/clock`, `CameraInfo` |
| `/camera_image_bridge` | `ros_gz_image` | Both RGB streams |
| `/groot_policy` | `groot_vla` | The VLA control loop |
| `/goal_marker` | `groot_vla` | Interactive 6-DOF RViz goal |
| `/world_publisher` | `groot_vla` | World → RViz markers + planning-scene collision objects |
| `/control_gui` | `groot_vla` | Qt operator panel (when `gui:=true`) |
| `/unity_endpoint` | `ros_tcp_endpoint` | TCP bridge to Unity (when used) |

### Principal topics

| Topic | Type | Publisher → Subscriber |
|---|---|---|
| `/clock` | `rosgraph_msgs/Clock` | Gazebo → all (`use_sim_time`) |
| `/joint_states` | `sensor_msgs/JointState` | `joint_state_broadcaster` → policy, MoveIt, TF |
| `/wrist_camera/image_raw` | `sensor_msgs/Image` | Gazebo → policy, recorder |
| `/scene_camera/image_raw` | `sensor_msgs/Image` | Gazebo → policy, recorder |
| `/arm_controller/joint_trajectory` | `trajectory_msgs/JointTrajectory` | **policy or Servo** → controller |
| `/gripper_controller/joint_trajectory` | `trajectory_msgs/JointTrajectory` | policy → controller |
| `/servo_node/delta_twist_cmds` | `geometry_msgs/TwistStamped` | policy → Servo (`eef_delta` only) |
| `/groot_policy/status` | `std_msgs/String` (JSON) | policy → GUI, marker |
| `/groot_policy/action` | `std_msgs/String` (JSON) | policy → GUI |
| `/groot_policy/instruction` | `std_msgs/String` | operator → policy |
| `/world_markers` | `visualization_msgs/MarkerArray` | `world_publisher` → RViz |
| `/goal_marker/*` | interactive marker protocol | `goal_marker` ↔ RViz |
| `/monitored_planning_scene` | `moveit_msgs/PlanningScene` | `move_group` → RViz, Servo |

### Services and actions

| Interface | Type | Purpose |
|---|---|---|
| `/groot_policy/enable` | `std_srvs/SetBool` | Arm/disarm. Pings the server first and refuses if unreachable |
| `/groot_policy/halt` | `std_srvs/Trigger` | Disarm and freeze at the measured position |
| `/groot_policy/reset_policy` | `std_srvs/Trigger` | Clear the server's action-chunk history between episodes |
| `/goal_marker/go_to_marker` | `std_srvs/Trigger` | Plan and execute to the marker |
| `/episode_recorder/{start,stop,discard}_episode` | `std_srvs/Trigger` | Recording control |
| `/servo_node/switch_command_type` | `moveit_msgs/ServoCommandType` | Select TWIST mode |
| `/move_action` | `moveit_msgs/MoveGroup` | Plan + execute |
| `/execute_trajectory` | `moveit_msgs/ExecuteTrajectory` | Execute precomputed |
| `/compute_ik`, `/compute_cartesian_path` | services | IK, straight-line segments |

---

## 4. Control paths: with and without the VLA

Both paths terminate at the **same** `arm_controller`. This is the single most
important structural fact about the system, and the source of its one true
concurrency hazard.

### Without the VLA — classical pipeline

```
goal (pose or joints)
   └─▶ move_group
         ├─ /compute_ik ......... pose → joint solution (KDL, tip = tcp_link)
         ├─ OMPL RRTConnect ..... collision-free path in joint space
         ├─ time parameterisation (TOTG) using joint_limits.yaml
         └─ FollowJointTrajectory ─▶ arm_controller ─▶ hardware interface
```

Entered by: the RViz MotionPlanning panel, the goal marker, the GUI's manual
controls, or `pick_place_demo`.

Properties: **collision-aware** (the planning scene contains the table),
**kinematically validated** (IK failure is reported, not approximated),
**reproducible**. This is the control condition against which policy behaviour
is judged.

### With the VLA — learned pipeline

```
cameras + /joint_states
   └─▶ policy_node
         ├─ ObservationBuilder ... decode, resize 224×224, assemble modality dict
         ├─ GrootClient .......... ZeroMQ REQ, msgpack, → policy server
         │                          ◀── action chunk (B, T, D)
         ├─ ActionMapper ......... decode, clamp, rate-limit
         └─ JointTrajectory ─▶ arm_controller           (joint_position/joint_delta)
            or TwistStamped ─▶ moveit_servo ─▶ arm_controller   (eef_delta)
```

Properties: **no collision checking** — the policy is expected to have learned
avoidance, and nothing verifies it. **No kinematic validation** — joint targets
are commanded directly. Safety therefore comes from the envelope in §8, not from
the planner.

### The two cannot run concurrently

Both publish `JointTrajectory` to `/arm_controller/joint_trajectory`. When the
policy is armed and MoveIt also executes, the controller receives interleaved
trajectories, MoveIt's execution is preempted, and it reports
`CONTROL_FAILED (-4)`. That reads like a defect but is contention.

The system therefore **interlocks** rather than arbitrates: while
`/groot_policy/status` reports `enabled: true`, `goal_marker` refuses motion
requests with an explanation and the GUI disables its motion controls with a
tooltip. Refusing is preferred to queueing because the operator should know
which agent owns the arm.

### Why `eef_delta` needs a different path

OpenVLA emits end-effector deltas `[dx, dy, dz, dR, dP, dY, gripper]`, not joint
targets. Converting these requires IK, singularity handling and joint-limit
avoidance at streaming rate — which is exactly `moveit_servo`. Two consequences:

- Servo halts if commands stop for `incoming_command_timeout` (0.25 s). A policy
  at 0.2 Hz would therefore move the arm in stutters, so `policy_node`
  **republishes the last twist at 20 Hz**, expiring it after `twist_hold_time`
  so a dead policy cannot leave the arm drifting.
- Servo rejects all input until a command type is selected. `policy_node` calls
  `/servo_node/switch_command_type` on arm.

---

## 5. SmolVLA: the policy

### What it is

SmolVLA (HuggingFace/LeRobot) is a 450M-parameter vision-language-action model:
a SmolVLM2-500M vision-language backbone, truncated to 16 layers, paired with a
flow-matching **action expert**. It consumes RGB frames, a proprioceptive state
vector and a natural-language instruction, and emits a *chunk* of future actions
— 50 timesteps by default — rather than a single step.

Action chunking matters architecturally: it means inference frequency and
control frequency are decoupled. A policy running at 1 Hz can still command a
smoothly-sampled trajectory, because each inference covers
`execution_horizon × action_dt` seconds of motion.

### Why this model

It is the only capable VLA that fits the hardware with room to work.

| Model | Params | VRAM measured | Fits 6 GB? |
|---|---|---|---|
| mock | — | 0 | trivially |
| **SmolVLA** | 450M | **0.91 GB** | comfortably |
| OpenVLA-7B (NF4) | 7B | **4.32 GB** weights, ~4.7 GB with activations | marginal |
| GR00T N1.7-3B | 3B | 16 GB+ required | no |

### Observation contract

```python
{
  "video":    {"wrist_view": uint8 (1, 1, 224, 224, 3),
               "ego_view":   uint8 (1, 1, 224, 224, 3)},
  "state":    {"single_arm": float32 (1, 1, 6),
               "gripper":    float32 (1, 1, 1)},
  "language": {"task": [["pick up the red cube and place it in the tray"]]},
}
```

Shapes are `(batch, time, …)` with both dimensions 1 for single-frame
closed-loop control. Two schemas exist — nested (GR00T N1.7) and flat (N1.5) —
and `probe_server` determines empirically which a given server accepts.

Cameras are emitted **in declaration order, not dictionary order**. The images
dictionary is populated by subscription callbacks, so its ordering depends on
which camera published first; a server mapping views onto image slots
positionally would otherwise swap wrist and scene between runs, silently
destroying correspondence with training data.

### Action contract and the dimensional asymmetry

State is **6-dimensional**, actions are **7-dimensional**. This is deliberate
and non-obvious.

`lerobot/smolvla_base` declares a 6-dim `observation.state`. Fine-tuning from a
checkpoint retains that declaration while taking action width from the dataset.
Writing a 7-dim state therefore produces a checkpoint whose config records
state `[6]` and action `[7]` — which loads without complaint and fails on the
first inference:

```
RuntimeError: The size of tensor a (6) must match the size of tensor b (7)
```

The exporter consequently writes the six arm joints as state, and the six joints
plus normalised gripper as action. A fine-tuned checkpoint gains gripper control
the base model structurally lacks (`action_dim` 6 → 7).

### Wire protocol

ZeroMQ REQ/REP carrying msgpack with the msgpack-numpy ndarray extension.
Request: `{"endpoint": str, "data": Any, "api_token": str?}`; endpoints `ping`,
`get_action`, `get_modality_config`, `reset`, `kill`.

`groot_client.py` implements this without importing torch, and contains a
fallback codec used when `msgpack_numpy` is absent. That fallback was verified
**byte-identical** to the reference implementation across ndarray, scalar,
non-contiguous and nested payloads — the correctness of every observation
depends on it.

A REQ socket enforces strict send/receive alternation and a timed-out receive
wedges it permanently, so every timeout tears down and rebuilds the socket.

---

## 6. The dataset and training procedure

### Demonstrations are generated, not teleoperated

The scripted pick-and-place succeeds reliably, so it serves as the expert.
`collect_demos` runs it repeatedly, and records observations and actions at
10 Hz.

The **action label at step *t* is the state at *t+1*** — the expert's next
configuration is the correct answer for the current observation. The final frame
of each episode is discarded because it has no successor.

Episodes that fail are **discarded**, not kept. A dataset containing failures
teaches failure.

### Domain randomisation

Each episode varies:

| Varied | Range |
|---|---|
| Key light | colour, direction (azimuth full circle, elevation 0.55–1.25 rad), intensity 0.55–1.25 |
| Table surface | six colour families, jittered ±0.06 |
| Walls | five colours, jittered |
| Objects | box / cylinder / sphere; 35–55 mm; seven colours; randomised position |
| Distractors | *n* additional objects that must be ignored |

Deliberately **not** varied: robot, camera poses, tray position, physics.
Randomising the observer as well as the observed makes failures difficult to
attribute, and camera extrinsics are something a real deployment calibrates.

Object size is bounded to 35–55 mm because the gripper spans 24–104 mm. A wider
object cannot be grasped and would only contribute failure.

The instruction names what was actually spawned — *"pick up the purple cylinder
and place it in the tray"* — so language grounds on the real object rather than
a fixed token.

Two Gazebo behaviours constrain the implementation, both discovered empirically:

- **A static model re-created at runtime returns without collision geometry.**
  Replacing the table to recolour it dropped every object through it to the
  floor (objects settled at z = 0.02 rather than 0.62). The table is therefore
  never replaced; its colour comes from a thin visual-only overlay while the
  world's collision surface is untouched.
- **`create`/`remove` are asynchronous**, and even the `/blocking` variants
  occasionally return before the entity has left the world. Removal is
  confirmed by polling the model list, and spawning retries.

### Training

Only **100M of 450M parameters** are trainable: SmolVLA freezes the vision
encoder and trains only the action expert (`freeze_vision_encoder`,
`train_expert_only`). This is the sole reason training fits in 6 GB — training
all 450M would require optimiser states of ~3.6 GB on top of weights,
gradients and activations.

```
dataset.num_episodes = 41
dataset.num_frames   = 6378
batch size           = 4
learnable params     = 99,880,992 (100M) of 450,046,176
VRAM                 = 4.1 GB
throughput           = ~1.7 s/step
```

Observed loss: 0.372 → 0.191 → 0.126 → 0.105 over 500 steps, gradient norm
falling monotonically (3.8 → 1.6). A falling loss demonstrates that the
demonstrations are being fitted; it is **not** evidence of task competence.

`--rename_map` is mandatory: the checkpoint declares `camera1/2/3` while the
dataset uses semantic names.

### Dataset format

LeRobot **v3.0**. The exporter builds datasets through LeRobot's own
`LeRobotDataset` API rather than writing the layout directly, because v3.0
consolidates many episodes per parquet/mp4 file, keeps episode metadata as
parquet, and requires per-episode normalisation statistics that cannot be
guessed. LeRobot ships a v2.1→v3.0 converter, but it resolves datasets through
the HuggingFace hub and cannot convert a local one.

---

## 7. Unity Robotics and the ROS-TCP data flow

### Purpose

Unity is an **alternative simulator**, not a replacement runtime. Its HDRP
renderer offers global illumination, area lights and reflections that Ogre2 in
this Gazebo build does not. The value is visual variety in the dataset.

### Data flow

```
┌────────────────── UNITY ──────────────────┐        ┌──────── ROS 2 ────────┐
│                                            │        │                       │
│  Camera (wrist) ─┐                         │        │                       │
│  Camera (scene) ─┼─▶ GrootArmBridge ──┐    │  TCP   │  ros_tcp_endpoint     │
│  ArticulationBody ┘   • RGB24 readback │    │ :10000 │        │              │
│         ▲             • row flip       ├────┼───────▶│        ▼              │
│         │             • JointState     │    │        │  /wrist_camera/…     │
│         │                              │    │        │  /scene_camera/…     │
│         │                              │    │        │  /joint_states       │
│         │                              │    │        │        │              │
│         └────── xDrive.target ◀────────┤◀───┼────────┤  /arm_controller/    │
│                  (deg = rad × 57.3)    │    │        │   joint_trajectory   │
└────────────────────────────────────────┘    │        └───────────────────────┘
```

The bridge publishes on **exactly the topic names, message types and image
dimensions Gazebo uses**. The policy stack cannot determine which simulator
produced an observation. That equivalence is the design goal: it permits
comparing renderers, or training on Unity output and evaluating in Gazebo,
without modifying any ROS code.

### Coordinate and unit conversions

Three conversions are required, and each fails silently if omitted:

| Quantity | Unity | ROS | Handling |
|---|---|---|---|
| Image origin | bottom-left | top-left | Rows flipped on publish. Omitting this yields a vertically mirrored image that appears plausible and poisons training |
| Joint angle | degrees (`xDrive.target`) | radians | `Mathf.Rad2Deg` on command |
| Up axis | Y-up | Z-up | Handled by URDF-Importer at import; camera placement must account for it manually |

### Trajectory handling

The bridge applies the **last** point of an incoming chunk as the drive target,
not each intermediate point. Unity's articulation drives interpolate internally;
replaying every waypoint fights the drive rather than assisting it. This is an
approximation of `ros2_control`'s trajectory execution, not an implementation of
it — see §11.

### Clock

Gazebo publishes `/clock`; Unity does not. Nodes must therefore run with
`use_sim_time:=false` when Unity is the simulator, or they will block
indefinitely awaiting a clock that never ticks.

### Endpoint patch

Upstream `ros_tcp_endpoint` calls `rclpy.shutdown()` twice. Earlier ROS
distributions tolerated this; Jazzy raises
`rcl_shutdown already called on the given context`. The setup script applies a
guard. The package is fetched rather than vendored.

---

## 8. Constraints and safety envelope

A VLA is a neural network, not a validated controller. It will occasionally emit
out-of-distribution actions. Because the learned path bypasses collision
checking and kinematic validation entirely (§4), the following constraints are
load-bearing rather than defensive programming.

### Policy-level

| Constraint | Value | Rationale |
|---|---|---|
| Starts disarmed | — | Arming is an explicit operator decision |
| Server reachability check | on arm | Refuses to arm against an unreachable server |
| Per-step joint clamp | 0.08 rad (SmolVLA) | Bounds commanded joint velocity |
| Absolute joint limits | tighter than UR5e's ±2π | Excludes cable-wrapping and through-table postures |
| Workspace box | `[-0.2, 0.95] × [-0.75, 0.75] × [-0.10, 0.85]` on `tcp_link` | Leaving it disarms and halts |
| Observation watchdog | 1–5 s | Halts if cameras or joint states go stale |
| Twist expiry (`eef_delta`) | `twist_hold_time` | A dead policy cannot leave the arm drifting |
| Exception containment | — | Any inference fault disarms rather than killing the control thread |
| `dry_run` | optional | Logs commands instead of sending them |

The workspace guard is not theoretical: during development it detected a runaway
positive-feedback loop and halted the arm at the boundary.

### Trajectory-level

Rate limiting is applied from the **live** joint position read immediately
before publishing, not from the observation captured before inference. On a slow
policy that snapshot is a full inference period old (~850 ms for SmolVLA), the
arm has moved, and every chunk begins with a step discontinuity the controller
cannot track — producing
`Holding position due to state tolerance violation` (measured: position error
0.500069 against a 0.5 rad tolerance). Reading fresh state eliminated this: six
violations over a 75 s armed run became zero.

### Controller-level

| Parameter | Value | Why this value |
|---|---|---|
| `arm_controller` goal tolerance | 0.05 rad | Comfortably inside joint range |
| `gripper_controller` goal tolerance | 0.005 m | **Must** be far below the 0.04 m stroke. A tolerance wider than the joint's own range makes the controller report "already at goal" immediately, and the fingers never move |
| `gripper_controller` `goal_time` | 0.5 s | **Must** be non-zero. `0.0` means *wait indefinitely*, so a finger stalled on a grasped object never terminates the action |
| Finger joint limits | −0.005 to 0.045 m | Deliberately **wider** than the commanded 0–0.04 range. Repeatedly driving a force-controlled joint into a mechanical stop makes the physics engine's joint-limit constraint lock it |
| Finger joint friction | 0.0 | `gz_ros2_control` drives position interfaces with a *force*; 1 N of Coulomb friction stalls a 50 g finger outright while every layer above reports success |

### Grasp geometry

`tcp_link` is at the **centre of the finger span**, not the fingertips. Placing
it at the tips makes "move the TCP to the object centre" drive the fingers half
a finger-length into the supporting surface.

---

## 9. Inspecting a running system

### The graph

```bash
ros2 node list
ros2 topic list -t
ros2 node info /groot_policy
rqt_graph
```

**`ros2 node list` is not authoritative.** It queries a caching daemon, and
because the launch file staggers node startup, a query issued while nodes are
still appearing caches an incomplete picture and continues serving it. If the
list looks wrong:

```bash
ros2 daemon stop && sleep 5 && ros2 node list
```

It returns empty for ~10 s afterwards while rediscovering — that is expected,
not a failure. `ros2 topic list` uses a different mechanism and is more
trustworthy.

### The policy

```bash
ros2 topic echo /groot_policy/status    # armed, server, inferences, failures, latency
ros2 topic echo /groot_policy/action    # the emitted chunk, and Δ from current position
ros2 run groot_vla probe_server --host <host> --port <port>
```

`probe_server` reports reachability, the server's modality configuration, which
observation schema it accepts, action keys and shapes, and measured latency. It
is the correct first diagnostic before connecting a robot to an unfamiliar
server.

### Controllers

```bash
ros2 control list_controllers
ros2 control list_hardware_interfaces
ros2 topic echo /gripper_controller/controller_state --once
```

The controller-state topic separates two failure classes that appear identical
from outside: if `output` tracks the command but `feedback` does not, the fault
is in physics or the hardware interface, not in ROS.

### GPU

```bash
nvidia-smi        # the full Processes table, including 'G' rows
```

`--query-compute-apps` lists **CUDA contexts only**. The desktop compositor,
browser and Gazebo's renderer are graphics contexts and invisible to it — over a
gigabyte on this machine. Using that query alone produces the actively
misleading report that the only process on the GPU is the policy server itself.

### Simulator

```bash
gz model --list
gz topic -e -t /world/tabletop/dynamic_pose/info -n 1
```

Object ground truth from the simulator is how task success is verified. "The
demo reported success" and "the cube is in the tray" are different claims; only
the second is checked here.

---

## 10. Measured performance

All figures from the target machine (RTX 2060 6 GB, 16 cores, 14 GB RAM).

### Policy inference

| Server | VRAM | Latency | Action space |
|---|---|---|---|
| mock | 0 | ~1 ms | joint |
| SmolVLA (GPU) | 0.91 GB | 850 ms | joint |
| SmolVLA (CPU fallback) | 0 | ~25 s | joint |
| OpenVLA-7B NF4 | 4.32 GB | 2.8–4.8 s | eef_delta |

### Simulation

| Configuration | Camera rate |
|---|---|
| flat colours, 640×480 | 30 Hz |
| + PBR normal/roughness/AO maps | 27 Hz |
| + procedural sky | 19.8 Hz *(rejected)* |
| + AmbientCG textures, 320×240 | **39 Hz** |

### Data collection

| Metric | Before | After |
|---|---|---|
| Episode duration | ~45 s | **18 s** |
| Contributing: `real_time_factor` | 1.0 | 0 (unthrottled) |
| Contributing: MoveIt velocity scaling | 0.25 | 0.6 |
| Contributing: camera resolution | 640×480 | 320×240 |

Parallel collection: ~1.6 GB RAM and 0.25 GB VRAM per worker. **RAM-bound, not
core-bound** on this machine — two to three workers. At 3 workers × 18 s, 150
episodes takes ~15 minutes rather than two hours.

### Functional verification

| Test | Result |
|---|---|
| Scripted pick-and-place | 3/3 cubes within ~2 mm of tray centre (Gazebo ground truth) |
| Gripper cycling | 24/24 open/close, 0 failures |
| Finger stall on 40 mm cube | 0.0080 m — the exact theoretical contact point |
| Goal marker | requested poses reached within 2.4 mm and 4.7 mm |
| MoveIt Servo | commanded axis tracked, other two held < 1 mm |
| Mock closed loop | 392 inferences, 0 failures |
| SmolVLA closed loop | 58 inferences, 0 failures |
| OpenVLA closed loop | 24 inferences, 0 failures, 0.14 m Cartesian travel |
| Domain-randomised collection | 41 episodes, 0 discarded |
| Parallel collection | 2 workers × 2 episodes, merged correctly |
| msgpack fallback codec | byte-identical to reference across 6 payload classes |

---

## 11. Known limitations

Stated plainly, because a documented limitation is more useful than an implied
capability.

**Zero-shot VLA policies do not perform the task.** SmolVLA and OpenVLA were
trained on other robots. A UR5e with this gripper is an unseen embodiment with
unfamiliar joint ranges and foreign normalisation statistics. Both move the arm
plausibly and neither completes the task. The safety envelope catches the
resulting excursions. This is expected, not a defect.

**The current dataset is small.** 41 episodes / 6,378 frames. For a single task
under this much randomisation, 100–200 episodes is the realistic target.
Randomisation *increases* the data requirement, since invariance must be learned.

**The gripper is fictional.** It is primitive geometry with 0–40 mm travel,
designed for this simulation. No commercial gripper matches it, so the seventh
action dimension is calibrated against something that does not exist. Real
deployment requires modelling the actual gripper and retraining.

**Sim-to-real transfer is not addressed.** Beyond the gripper: Ogre2 output is
not photographic, the real UR5e has different joint dynamics, compliance and
controller latency, and camera extrinsics will differ. Domain randomisation
narrows the gap but does not close it from simulation-only data. This is an open
research problem, not a configuration step.

**OpenVLA is marginal on this hardware.** It requires ~4.8 GB free; a typical
desktop leaves 4.5–4.6 GB. A 4-bit model **cannot be partially offloaded** —
CPU offload loads without complaint and then fails on the first inference with
`Blockwise 4bit quantization only supports 16/32-bit floats, but got
torch.uint8`, because bitsandbytes cannot dequantise blocks in host memory. The
server therefore refuses up front with the precise shortfall.

**Ogre2 has no global illumination in this build.** No bounced light, no
ray-traced reflections. Photorealism is not reachable; §10 quantifies what
remains available.

**Unity is not control-faithful.** There is no `ros2_control` in Unity. Its
articulation drives are not the controller a real UR5e uses, and MoveIt
trajectory execution is approximated by the bridge rather than executed. Use
Unity for visual variety in datasets; use Gazebo for anything concerning control
fidelity.

**Gazebo occasionally jams the finger joints.** Under repeated grasping the
prismatic joints can stop responding — the controller's output tracks perfectly
while the measured position does not. Widening the joint limits beyond the
commanded range largely resolved it (24/24 cycles clean), but the underlying
contact-solver behaviour remains. `MoveItHelper.set_gripper` verifies motion
occurred and raises rather than reporting false success, because a stalled
finger and a dead finger both abort with `GOAL_TOLERANCE_VIOLATED`.

---

## Appendix A: captured graph

Verbatim from a running system (`policy:=mock`, headless, Gazebo).
Transient, introspection and TF-listener nodes elided.

### Nodes

```
/arm_controller
/camera_image_bridge
/controller_manager
/forward_position_controller
/goal_marker
/goal_marker
/gripper_controller
/groot_policy
/gz_bridge
/gz_ros_control
/joint_state_broadcaster
/move_group
/move_group/moveit
/moveit_3087812598
/moveit_simple_controller_manager
/robot_state_publisher
/unity_endpoint
/world_publisher
```

### Topics

```
/arm_controller/controller_state [control_msgs/msg/JointTrajectoryControllerState]
/arm_controller/joint_trajectory [trajectory_msgs/msg/JointTrajectory]
/arm_controller/speed_scaling_input [control_msgs/msg/SpeedScalingFactor]
/arm_controller/transition_event [lifecycle_msgs/msg/TransitionEvent]
/attached_collision_object [moveit_msgs/msg/AttachedCollisionObject]
/clock [rosgraph_msgs/msg/Clock]
/collision_object [moveit_msgs/msg/CollisionObject]
/controller_manager/activity [controller_manager_msgs/msg/ControllerManagerActivity]
/controller_manager/introspection_data/full [pal_statistics_msgs/msg/Statistics]
/controller_manager/introspection_data/names [pal_statistics_msgs/msg/StatisticsNames]
/controller_manager/introspection_data/values [pal_statistics_msgs/msg/StatisticsValues]
/controller_manager/statistics/full [pal_statistics_msgs/msg/Statistics]
/controller_manager/statistics/names [pal_statistics_msgs/msg/StatisticsNames]
/controller_manager/statistics/values [pal_statistics_msgs/msg/StatisticsValues]
/diagnostics [diagnostic_msgs/msg/DiagnosticArray]
/display_contacts [visualization_msgs/msg/MarkerArray]
/display_planned_path [moveit_msgs/msg/DisplayTrajectory]
/dynamic_joint_states [control_msgs/msg/DynamicJointState]
/forward_position_controller/commands [std_msgs/msg/Float64MultiArray]
/forward_position_controller/transition_event [lifecycle_msgs/msg/TransitionEvent]
/goal_marker/feedback [visualization_msgs/msg/InteractiveMarkerFeedback]
/goal_marker/goal_pose [geometry_msgs/msg/PoseStamped]
/goal_marker/status [std_msgs/msg/String]
/goal_marker/update [visualization_msgs/msg/InteractiveMarkerUpdate]
/gripper_controller/controller_state [control_msgs/msg/JointTrajectoryControllerState]
/gripper_controller/joint_trajectory [trajectory_msgs/msg/JointTrajectory]
/gripper_controller/speed_scaling_input [control_msgs/msg/SpeedScalingFactor]
/gripper_controller/transition_event [lifecycle_msgs/msg/TransitionEvent]
/groot_policy/action [std_msgs/msg/String]
/groot_policy/instruction [std_msgs/msg/String]
/groot_policy/status [std_msgs/msg/String]
/joint_state_broadcaster/transition_event [lifecycle_msgs/msg/TransitionEvent]
/joint_states [sensor_msgs/msg/JointState]
/monitored_planning_scene [moveit_msgs/msg/PlanningScene]
/pipeline_state [moveit_msgs/msg/PipelineState]
/planning_scene [moveit_msgs/msg/PlanningScene]
/planning_scene_world [moveit_msgs/msg/PlanningSceneWorld]
/robot_description_semantic [std_msgs/msg/String]
/robot_description [std_msgs/msg/String]
/scene_camera/camera_info [sensor_msgs/msg/CameraInfo]
/scene_camera/image_raw/compressedDepth [sensor_msgs/msg/CompressedImage]
/scene_camera/image_raw/compressed [sensor_msgs/msg/CompressedImage]
/scene_camera/image_raw [sensor_msgs/msg/Image]
/scene_camera/image_raw/theora [theora_image_transport/msg/Packet]
/scene_camera/image_raw/zstd [sensor_msgs/msg/CompressedImage]
/servo_node/delta_twist_cmds [geometry_msgs/msg/TwistStamped]
/tf_static [tf2_msgs/msg/TFMessage]
/tf [tf2_msgs/msg/TFMessage]
/trajectory_execution_event [std_msgs/msg/String]
/world_markers [visualization_msgs/msg/MarkerArray]
/wrist_camera/camera_info [sensor_msgs/msg/CameraInfo]
/wrist_camera/image_raw/compressedDepth [sensor_msgs/msg/CompressedImage]
/wrist_camera/image_raw/compressed [sensor_msgs/msg/CompressedImage]
/wrist_camera/image_raw [sensor_msgs/msg/Image]
/wrist_camera/image_raw/theora [theora_image_transport/msg/Packet]
/wrist_camera/image_raw/zstd [sensor_msgs/msg/CompressedImage]
```

### `/groot_policy` interfaces

The VLA node's complete wiring. Note it subscribes only to sensors and
publishes only commands and telemetry: it holds no planning state, which is
why it can be swapped or disabled without disturbing the rest of the graph.

```
/groot_policy
  Subscribers:
    /clock: rosgraph_msgs/msg/Clock
    /groot_policy/instruction: std_msgs/msg/String
    /joint_states: sensor_msgs/msg/JointState
    /scene_camera/image_raw: sensor_msgs/msg/Image
    /tf: tf2_msgs/msg/TFMessage
    /tf_static: tf2_msgs/msg/TFMessage
    /wrist_camera/image_raw: sensor_msgs/msg/Image
  Publishers:
    /arm_controller/joint_trajectory: trajectory_msgs/msg/JointTrajectory
    /gripper_controller/joint_trajectory: trajectory_msgs/msg/JointTrajectory
    /groot_policy/action: std_msgs/msg/String
    /groot_policy/status: std_msgs/msg/String
    /servo_node/delta_twist_cmds: geometry_msgs/msg/TwistStamped
  Service Servers:
    /groot_policy/enable: std_srvs/srv/SetBool
    /groot_policy/halt: std_srvs/srv/Trigger
    /groot_policy/reset_policy: std_srvs/srv/Trigger
  Service Clients:
    /servo_node/switch_command_type: moveit_msgs/srv/ServoCommandType
  Action Servers:

  Action Clients:


```
