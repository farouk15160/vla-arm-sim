# VLA arm test cell — ROS 2 Jazzy + MoveIt 2 + Gazebo Harmonic

A simulated UR5e with a parallel gripper and two cameras, driven by a
vision-language-action policy. Simulation only.

```
 cameras + /joint_states ──► observation ──► policy server (ZMQ) ──► action chunk
                                                                        │
                              JointTrajectory / TwistStamped ──► ros2_control ──► Gazebo
```

The ROS side is a **protocol client**, not a model host. Swapping policies means
starting a different server — the robot, MoveIt config and safety layer are
untouched.

## Which policy

| Server | Params | VRAM measured | Latency | Action space | Fits your 6 GB? |
|---|---|---|---|---|---|
| `mock_policy_server` | — | none | ~1 ms | joint | yes — pipeline testing |
| `smolvla_server` | 450M | **0.91 GB** | ~950 ms | joint | yes |
| `openvla_server` | 7B @ 4-bit | **4.38 GB** | ~2.8-4.8 s | **eef_delta** | yes |
| GR00T N1.7-3B | 3B | 16 GB+ | — | joint | no — remote GPU only |

All four speak the same ZeroMQ + msgpack protocol, so the ROS side is
identical for each. Pick one with `policy:=` on the launch file.

**SmolVLA** is the fast default. **OpenVLA-7B** is the big one: 15 GB on disk,
quantised to NF4 at load time so it runs in 4.38 GB. It emits 7-DoF
end-effector deltas rather than joint targets, so it drives the arm through
`moveit_servo` — `system.launch.py policy:=openvla` starts Servo for you.

OpenVLA needs **its own venv**: it pins `transformers==4.40`, while LeRobot
requires `>=4.57`. The two cannot coexist, and since each server is a separate
process that costs nothing but disk.

### Expect poor zero-shot behaviour

`lerobot/smolvla_base` has never seen a UR5e with this gripper. Its declared
state/action width is 6 (no gripper dimension) and its normalisation statistics
come from other robots. Zero-shot it produces plausible-looking but
task-incompetent motion — the arm moves, it does not solve the task. This is
expected for any VLA on an unseen embodiment.

To get real behaviour you must fine-tune: record demonstrations with
`episode_recorder`, convert with `export_lerobot`, then train. See
**Collecting data for fine-tuning** below. The value of this stack is that the
full loop is measurable end to end on hardware you own.

## Build

```bash
cd ~/Desktop/code/robotics_and_ros/groot_arm_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

One non-ROS dependency for the ROS side:

```bash
sudo apt install python3-msgpack        # or: pip install --break-system-packages msgpack
```

`msgpack-numpy` is optional — `groot_vla/groot_client.py` contains a
byte-identical fallback codec used automatically when it is absent.

### Policy environment (separate from ROS)

torch and lerobot must NOT go into the ROS environment. Build a venv once:

```bash
python3 -m venv ~/vla_venv
~/vla_venv/bin/pip install \
    "lerobot[smolvla] @ file://$HOME/path/to/lerobot" \
    pyzmq msgpack msgpack-numpy
~/vla_venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Point the `file://` URL at your own LeRobot checkout, or drop the path and use
`"lerobot[smolvla]"` to install from PyPI.

Verified working: torch 2.7.1+cu126, CUDA available, lerobot 0.4.3, ~6.5 GB on disk.

For OpenVLA, a second venv with its pinned stack:

```bash
python3 -m venv ~/openvla_venv
~/openvla_venv/bin/pip install torch torchvision "transformers==4.40.1" \
    "tokenizers==0.19.1" "timm==0.9.10" "accelerate==0.30.1" bitsandbytes \
    pyzmq msgpack msgpack-numpy pillow numpy
```

`accelerate` must be pinned: newer versions call `.to()` on a 4-bit model,
which bitsandbytes rejects. `bitsandbytes` must NOT be pinned to 0.43 — that
version needs a `triton.ops` module recent torch no longer ships.

---

## Quick start — one command

Everything (sim, MoveIt, RViz, world, goal marker, GUI, policy server, policy
bridge) comes up from a single launch file:

```bash
cd ~/Desktop/code/robotics_and_ros/groot_arm_ws
source /opt/ros/jazzy/setup.bash && source install/setup.bash

ros2 launch groot_arm_bringup system.launch.py                    # mock, no GPU
ros2 launch groot_arm_bringup system.launch.py policy:=smolvla    # 0.9 GB, ~1 Hz
ros2 launch groot_arm_bringup system.launch.py policy:=openvla    # 4.4 GB, ~0.2 Hz
ros2 launch groot_arm_bringup system.launch.py policy:=none       # robot only
```

The policy server is started for you in the right venv. `policy:=openvla` also
starts `moveit_servo`, because OpenVLA speaks Cartesian deltas.

### Arguments

| Argument | Default | Meaning |
|---|---|---|
| `policy` | `mock` | `none` / `mock` / `smolvla` / `openvla` |
| `instruction` | pick up the red cube… | the task string |
| `gui` | `true` | Qt control panel |
| `rviz` | `true` | RViz |
| `gazebo_gui` | `true` | Gazebo window (false = headless) |
| `goal_marker` | `true` | draggable 3D goal in RViz |
| `policy_host` / `policy_port` | `127.0.0.1` / `5555` | non-local host = connect, don't start |
| `venv_python` | `~/vla_venv/bin/python` | interpreter for SmolVLA |
| `openvla_python` | `~/openvla_venv/bin/python` | interpreter for OpenVLA |
| `model_path` | *(empty)* | override the checkpoint (e.g. your fine-tune) |

Example — headless, your own fine-tuned SmolVLA, no GUI:

```bash
ros2 launch groot_arm_bringup system.launch.py \
    policy:=smolvla model_path:=$PWD/data/checkpoints/pickplace/checkpoints/last/pretrained_model \
    gazebo_gui:=false gui:=false \
    instruction:="put the blue cube in the tray"
```

## The control panel

`gui:=true` opens a Qt window with everything you would otherwise type:

* **Arm policy / Disable / HALT** — the policy enable service, plus a big red stop
* **Task instruction** — a dropdown of preset tasks, editable; publishes to
  `/groot_policy/instruction`, so the task changes without restarting anything
* **Manual control** — named poses (home / observe / up), gripper open and
  close, scene reset, and the scripted pick-and-place for a chosen cube
* **System** — armed state, **MoveIt connection status**, server, inference
  count, failures, latency, last error
* **VLA output** — live bar per arm joint plus the gripper, showing exactly what
  the policy is emitting while it is armed, with the chunk length, inference
  latency, staleness, and the delta from the arm's current position. This is the
  fastest way to tell a policy that is thinking from one that is stuck.
* **RViz goal marker** — the marker's live pose and a **GO TO MARKER** button

move_group is **always** started by the launch files — it is never optional,
because the goal marker, the manual controls and the scripted demo all plan
through it. The panel shows its connection state so you can see at a glance
whether it is reachable.

Blocking work runs on worker threads, so the window never freezes mid-motion,
and motion buttons disable while a motion is in flight so a double-click cannot
queue two trajectories.

## The RViz scene and the 3D goal marker

`world_publisher` republishes the Gazebo world (table, tray, three cubes) as
markers on `/world_markers`, and registers the static parts as MoveIt collision
objects. RViz and Gazebo now show the same thing, and the planner knows the
table is there. Cube poses are polled live from Gazebo, so cubes the robot
moves are followed rather than drawn where they started.

`goal_marker` adds a draggable 6-DOF handle in RViz. Three ways to make the
robot go there, easiest first:

1. Drag the cyan sphere, then press **⇒ GO TO MARKER** in the control panel.
2. Drag it, then right-click the marker → **Move here**.
3. Right-click → tick **Auto-go on release**; from then on, simply letting go
   of the marker moves the robot.

IK runs on the grasp frame (`tcp_link`), then it plans and executes through
move_group. The panel shows the marker's live coordinates and reports success
or the MoveIt error in its log.

Scriptable too:

```bash
ros2 service call /goal_marker/go_to_marker std_srvs/srv/Trigger
ros2 topic echo /goal_marker/goal_pose      # where the marker is
```

Other menu entries: **Reset marker to TCP**, **Open/Close gripper**.

Both displays are already in the shipped RViz config. If you built your own,
add *MarkerArray* on `/world_markers` and *InteractiveMarkers* on `/goal_marker`.

### Test it without a GPU

```bash
ros2 launch groot_arm_bringup system.launch.py policy:=mock
```

`mock_policy_server --behaviour hold` echoes the observed joint state back, so
the arm must not move at all — the sharpest test that the pipeline adds no
drift. `--behaviour wave` gives bounded motion proving trajectories reach the
controllers.

### Check a server before connecting the robot

```bash
ros2 run groot_vla probe_server --host 127.0.0.1 --port 5555
```

Reports reachability, modality config, which observation schema it accepts,
action keys and shapes, and measured latency.

## Classical baseline

A scripted MoveIt pick-and-place, for comparison against whatever the policy does:

```bash
ros2 launch groot_arm_bringup demo.launch.py
ros2 run groot_vla pick_place_demo --ros-args -p cube:=red_cube
ros2 run groot_vla scene_reset --randomize      # reset between rollouts
```

---

## Pointing at a real GR00T server

On the GPU machine, from an Isaac-GR00T checkout:

```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.7-3B \
    --embodiment-tag NEW_EMBODIMENT \
    --device cuda:0 --host 0.0.0.0 --port 5555
```

Check what it expects **before** connecting the robot:

```bash
ros2 run groot_vla probe_server --host <GPU_HOST> --port 5555
```

`probe_server` reports reachability, the server's modality config, which
observation schema it accepts, the action keys and shapes, and inference
latency. It tells you exactly what to put in `groot_policy.yaml`. Then:

```bash
ros2 launch groot_arm_bringup vla.launch.py \
    policy_host:=<GPU_HOST> \
    instruction:="pick up the red cube and place it in the tray"
```

Change the task at runtime without restarting:

```bash
ros2 topic pub --once /groot_policy/instruction std_msgs/String "{data: 'put the blue cube in the tray'}"
```

### Observation schema

GR00T's layout changed between releases and `groot_vla` supports both, selected
by `observation_schema` in `config/groot_policy.yaml`:

- `nested` (N1.7) — `{"video": {...}, "state": {...}, "language": {"task": [[...]]}}`
- `flat` (N1.5) — `{"video.ego_view": ..., "state.single_arm": ..., "annotation.human.task_description": [...]}`

Let `probe_server` decide; it tries both.

### Action spaces

Set `action_space` to match what your checkpoint emits:

| Value | Meaning | Path to the robot |
|---|---|---|
| `joint_position` | absolute joint targets (rad) | JointTrajectory → `arm_controller` |
| `joint_delta` | per-step increments (rad) | integrated, then as above |
| `eef_delta` | 6-D Cartesian velocity | TwistStamped → `moveit_servo` |

`eef_delta` additionally needs `ros2 launch groot_arm_moveit_config servo.launch.py`.
The policy node selects Servo's TWIST command type for you on enable.

---

## Safety

A VLA is a neural network, not a validated controller, and it will occasionally
emit nonsense. Everything below is load-bearing — do not loosen it before you
have watched the policy behave:

- **starts disabled**, and `~/enable` refuses to arm if the server is not answering
- per-step joint clamp (`max_joint_step`, default 0.15 rad)
- absolute joint limits, tighter than the UR5e's own
- **workspace box** on `tcp_link`; leaving it disables the policy and halts the arm
- **stale-observation watchdog** — halts if cameras or joint states go quiet
- any exception in the inference loop disables the policy rather than killing the thread
- `dry_run: true` logs commands instead of sending them

The workspace guard is not theoretical — during bring-up it caught a runaway
positive-feedback loop and stopped the arm at the box edge.

---

## Fine-tuning SmolVLA on this cell

Zero-shot VLAs flail here because a UR5e with this gripper is an unseen
embodiment. Fine-tuning is the fix, and the demonstrations can be **generated**
rather than teleoperated, because the scripted pick-and-place already succeeds.

### 1. Collect demonstrations

```bash
# terminal 1: simulator
ros2 launch groot_arm_bringup system.launch.py policy:=none gazebo_gui:=false

# terminal 2: recorder
ros2 run groot_vla episode_recorder --ros-args \
    -p output_dir:=$PWD/data/demos/pickplace -p fps:=10.0 -p use_sim_time:=true

# terminal 3: drive it
ros2 run groot_vla collect_demos --ros-args \
    -p episodes:=40 -p distractors:=2 -p use_sim_time:=true
```

`collect_demos` rebuilds the scene each episode, records, runs the same
`pick_place_sequence` the standalone demo uses, and **discards any episode that
fails** - a dataset containing failures teaches the policy to fail.

About 8 MB and 30 s per episode at 10 fps, so 40 episodes is ~20 minutes and
300 MB.

#### Domain randomisation

Every episode varies, so the dataset teaches the task rather than the decor. A
policy trained on one fixed scene learns *"the graspable thing is the small red
square at pixel (x, y) on a beige table"* and fails the moment anything changes.

| Varied | How |
|---|---|
| lighting | key light colour, direction, intensity |
| table | surface colour (six families, jittered) |
| walls | room colour |
| objects | shape (box / cylinder / sphere), size 35-55 mm, colour, position |
| distractors | extra objects that must be ignored |

Deliberately **not** varied: the robot, camera poses, tray position, physics.
Randomising the observer as well as the observed makes it much harder to tell
whether a failure is the policy or the setup, and camera extrinsics are
something a real deployment calibrates rather than guesses.

The instruction names what was actually spawned — *"pick up the purple cylinder
and place it in the tray"* — so the language grounds on the real object.

```bash
# preview one randomised scene without collecting
ros2 run groot_vla domain_randomizer --distractors 3 --seed 7

# fixed three-cube scene instead
ros2 run groot_vla collect_demos --ros-args -p domain_randomize:=false
# spheres too (they roll out of the gripper, so more episodes are discarded)
ros2 run groot_vla collect_demos --ros-args -p shapes:='[box,cylinder,sphere]'
```

Two Gazebo behaviours worth knowing, both found the hard way:

* A **static model re-created at runtime comes back without collision**. So the
  table is never replaced — its colour comes from a thin, visual-only overlay
  laid on top, and the world's collision surface is untouched. Replacing the
  table drops every object through it to the floor (verified: objects end at
  z = 0.02 instead of 0.62).
* The `create`/`remove` services are **asynchronous** and return before the
  entity exists; the `/blocking` variants are used throughout. Object removal
  also sweeps the whole `obj_*` namespace rather than a list this process
  built, because each run is a fresh process — otherwise leftovers block the
  new spawns (which do not allow renaming) and the scene silently never
  changes.

### 2. Convert to a LeRobot dataset

```bash
~/vla_venv/bin/python src/groot_vla/groot_vla/export_lerobot.py \
    --input ~/groot_episodes --output ~/groot_lerobot
```

Built through LeRobot's own `LeRobotDataset` API, which computes the
normalisation statistics and writes format v3.0 correctly. (Their v2.1 to v3.0
converter resolves datasets through the HuggingFace hub and cannot convert a
local one.)

### 3. Train

```bash
~/vla_venv/bin/lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --policy.repo_id=local/smolvla_ur5e --policy.push_to_hub=false \
    --dataset.repo_id=local/groot_ur5e --dataset.root=$PWD/data/datasets/pickplace \
    --rename_map='{"observation.images.wrist_view": "observation.images.camera1", "observation.images.ego_view": "observation.images.camera2"}' \
    --batch_size=4 --steps=20000 --output_dir=$PWD/data/checkpoints/pickplace \
    --policy.device=cuda --wandb.enable=false
```

`--rename_map` is required: the checkpoint declares `camera1/2/3` while the
dataset uses readable names.

Measured on the 6 GB RTX 2060 at batch size 2: **3.7 GB VRAM**, ~0.8 s/step,
**100M of 450M parameters trainable** - SmolVLA freezes the vision encoder and
trains only the action expert by default, which is what makes this fit.

### 4. Serve the result

```bash
ros2 launch groot_arm_bringup system.launch.py policy:=smolvla \
    model_path:=$PWD/data/checkpoints/pickplace/checkpoints/last/pretrained_model
```

A fine-tuned checkpoint has **7-dim actions** (6 joints + gripper) where the
base model has 6 and cannot grasp at all.

### State is 6-dim, actions are 7-dim

Not a typo. `lerobot/smolvla_base` declares a 6-dimensional `observation.state`,
and fine-tuning from it keeps that declaration while taking the action width
from the dataset. Writing a 7-dim state yields a checkpoint whose config says
state `[6]` and action `[7]`; it loads happily and then dies on the first
inference with *"The size of tensor a (6) must match the size of tensor b (7)"*.
So the exporter writes the six arm joints as state, and the six joints plus
gripper as action.

## Where things live

Generated data stays inside the workspace under `data/`, and is git-ignored:

| Path | What |
|---|---|
| `data/demos/` | raw recordings (PNG + JSONL) from `episode_recorder` |
| `data/datasets/` | LeRobot v3.0 datasets from `export_lerobot` |
| `data/checkpoints/` | `lerobot-train` output |

See `data/README.md` for what is currently there.

The two policy virtualenvs stay in `$HOME` deliberately. A venv hardcodes
absolute paths in `pyvenv.cfg` and every `bin/` shebang, so **moving one breaks
every command inside it**. To relocate, delete and recreate at the new path:

```bash
rm -rf ~/vla_venv && python3 -m venv /new/path/vla_venv
/new/path/vla_venv/bin/pip install \
    "lerobot[smolvla] @ file://$HOME/path/to/lerobot" pyzmq msgpack msgpack-numpy
```

## Packages

| Package | Contents |
|---|---|
| `groot_arm_description` | UR5e (from `ur_description`) + parallel gripper + wrist/scene cameras, `gz_ros2_control`, tabletop world, PBR materials and generated textures |
| `groot_arm_moveit_config` | SRDF, kinematics, OMPL/Pilz, `move_group`, RViz, Servo |
| `groot_arm_bringup` | `sim` / `demo` / `vla` launch files, `kill_stack.sh` |
| `groot_vla` | policy client, observation builder, action mapper, policy node, **SmolVLA server**, mock server, probe, MoveIt helper, recorder, exporter |

Key frames and topics:

- `tcp_link` — grasp frame, midway between the fingers; the IK tip and the frame
  `eef_delta` acts in
- `/wrist_camera/image_raw`, `/scene_camera/image_raw` — 640×480 RGB at 30 Hz,
  downsampled to 224×224 for the policy
- `arm_controller`, `gripper_controller` — shared by MoveIt and the policy

---

## Scene appearance

The world uses PBR materials with generated texture maps rather than flat
colours: a speckled melamine worktop, brushed-steel tray, painted walls and a
concrete floor, lit by a warm shadow-casting key, a cool fill and a soft
overhead. The cell is enclosed by walls on the sides the cameras actually face,
so renders are a grounded room rather than objects floating in grey void.

Textures are **generated, not downloaded**, so the repository stays
self-contained with no third-party asset licensing:

```bash
python3 src/groot_arm_description/materials/make_textures.py
```

Two things worth knowing if you change them:

* A map is stretched **once** across the surface it is on, so a feature 50 px
  wide in a 512 px map is ~14 cm wide on a 1.4 m table. Detail has to live at
  high spatial frequency or it renders as blotches. The first attempt at wood
  used ~5 grain cycles across the map and looked like spilled paint.
* Texture URIs use `model://groot_arm_description/...`. Gazebo resolves those by
  looking for a directory of that name inside each `GZ_SIM_RESOURCE_PATH` entry,
  so `sim.launch.py` puts the **parent** of the package share directory on the
  path. Point it at the package directory itself and every texture silently
  falls back to its flat diffuse colour, with only an `[Err]` line in the log.

The worktop is deliberately light and low-contrast: it keeps the coloured cubes
visually distinct, which is what the policy keys on.

## Collecting data faster

Episode time went from ~45 s to **18 s** (2.5x) through three measured changes:

| Change | Why |
|---|---|
| `real_time_factor: 1.0 -> 0` | physics runs unthrottled instead of pacing to wall-clock. Everything downstream uses `use_sim_time`, so controllers, MoveIt and the recorder speed up together and the recorded data is identical - just collected sooner |
| MoveIt scaling `0.25 -> 0.6` | most of an episode is the arm travelling between waypoints. Biggest single lever |
| cameras `640x480 -> 320x240` | the policy downsamples to 224x224 anyway, so the extra pixels were discarded work. Camera rate 19.8 -> 39 Hz |

Camera size is a launch argument, so raise it for nicer recordings:

```bash
ros2 launch groot_arm_bringup system.launch.py camera_width:=640 camera_height:=480
```

### Running several simulations at once

```bash
ros2 run groot_arm_bringup collect_parallel.sh \
    --workers 3 --episodes 40 --output $PWD/data/demos/run1

python3 src/groot_vla/groot_vla/merge_demos.py \
    --inputs data/demos/run1/worker_* --output data/demos/run1/merged
```

Each worker runs on its own `ROS_DOMAIN_ID` **and** its own `GZ_PARTITION`.
Both are needed: the domain id separates the DDS network, but Gazebo's own
transport discovery ignores it, so without a partition two simulations fight
over world and service names.

Workers number episodes from zero independently, so `merge_demos.py` renumbers
as it copies and records which worker each came from. It copies rather than
moves by default - a mistake should not destroy an hour of collection.

Sizing: roughly 1.6 GB RAM and 0.25 GB VRAM per worker, and physics runs
unthrottled so workers compete for CPU. On this machine (16 cores, 14 GB RAM)
**RAM is the limit, not cores** - two or three workers is the useful range.
At 3 workers x 18 s, 150 episodes takes about 15 minutes rather than two hours.

## How photorealistic can this get?

Short answer: **better, but not photoreal — and past a point it costs you data.**

Gazebo Harmonic renders through Ogre2, a rasteriser. This build ships **no
global-illumination plugin** (no voxel cone tracing), so there is no bounced
light, no ray-traced reflections and no real ambient occlusion. That is the
ceiling.

What the scene does use, and what each thing costs, measured on the RTX 2060 at
640x480:

| Configuration | Camera rate | Verdict |
|---|---|---|
| flat colours (original) | 30 Hz | baseline |
| + PBR albedo/normal/roughness/AO maps | 27 Hz | kept - visible surface detail for ~10% |
| + procedural `<sky>` | 19.8 Hz | **dropped** - 26% slower for a barely visible change |
| **+ AmbientCG photo textures, cameras 320x240** | **39 Hz** | **current** - better materials AND faster |

The scene now uses real CC0 PBR sets from [ambientcg.com](https://ambientcg.com)
(Wood062 worktop, Concrete034 floor, Metal032 tray, Plaster001 walls) and a
generated `workbench.stl` with an apron, chamfered lip and stretcher shelf -
silhouette cues a vision model uses to read "workbench" rather than "grey
rectangle". Collision stays a simple box: a 132-triangle mesh is needless work
for the physics engine when a slab is exact where it matters.

Textures are **not committed** (~13 MB of third-party JPEGs). Fetch them with:

```bash
src/groot_arm_description/materials/fetch_ambientcg.sh
```

Without them the world still loads - the procedural maps from
`make_textures.py` remain as stand-ins.

Note `NormalGL`, not `NormalDX`: Ogre2 expects the OpenGL convention (green
channel up). The DirectX variant has it inverted and lights surfaces subtly
inside-out.

Normal and roughness maps are the highest-value change available: they perturb
the shading normal per pixel, so a flat primitive catches light unevenly, which
is most of what the eye reads as "a material" rather than "coloured cardboard".
They are generated from the albedo by `materials/make_textures.py`, so there is
no third-party asset licensing.

### Do not add `<distortion>` to a camera sensor

Gazebo Harmonic accepts the element, loads the model without error, advertises
the topic - and then never publishes a frame. Verified both ways: adding it
killed both cameras, removing it restored 30 Hz. Apply lens distortion
downstream on the image if you want it.

### Realism is not the bottleneck for VLA training

Worth saying plainly, because it is easy to spend days here: **domain
randomisation matters more than fidelity for sim-to-real.** A policy trained on
one beautifully-rendered scene overfits that scene. A policy trained across
varied lighting, colours, shapes and distractors learns the task, even from
plainer renders.

Framerate is also not free: at 20 Hz instead of 30 the simulation collects
demonstrations proportionally slower, and dataset size is the thing most
limiting policy quality right now. Spend the GPU on more episodes before
spending it on prettier pixels.

## Verified

Checked by running it on ROS 2 Jazzy / Gazebo Harmonic 8.11 / Ubuntu 24.04 /
RTX 2060 6 GB:

- URDF and SRDF load consistently; 21 links, 8 actuated joints
- Gazebo brings up all controllers; `/joint_states` at 500 Hz; both cameras ~30 Hz
- the fallback msgpack codec is **byte-identical** to `msgpack_numpy` across
  ndarray, scalar, non-contiguous and nested payloads
- **SmolVLA loads on the GPU in 0.91 GB** and serves real inferences at
  ~950 ms each
- **full closed loop with real SmolVLA: 58 inferences, 0 failures**, no safety
  trips, arm driven continuously from camera + language input
- mock-server loop: 392 inferences, 0 failures, motion bounded to exactly the
  commanded amplitude
- **scripted pick-and-place: 3/3 cubes** placed within ~2 mm of the tray centre,
  confirmed from Gazebo ground truth (fingers stall at 0.0080 — the exact
  theoretical contact point for a 40 mm cube — and hold through the lift)
- gripper stress test: **24/24 open/close cycles, 0 failures**
- `moveit_servo` moves the TCP along a commanded Cartesian axis with the other
  two axes held under 1 mm
- **OpenVLA-7B in 4-bit: 4.38 GB VRAM**, 24 inferences / 0 failures, driving
  the arm 0.14 m through Servo on the eef_delta path
- **one launch file** brings up sim + MoveIt + world + marker + GUI + server +
  bridge with no errors
- world markers match the Gazebo world exactly (table top at base_link
  z = -0.025 = world 0.575, cubes at 0.62)
- **goal marker: requested poses reached within 2.4 mm and 4.7 mm**
- the Qt panel renders and shows live policy status

## Known issues

**Zero-shot SmolVLA does not solve the task.** It moves the arm smoothly and
the loop is stable, but the motion is not task-directed — an unseen embodiment
with unfamiliar joint ranges and foreign normalisation statistics. Fine-tuning
is required for competent behaviour; the recorder and exporter exist for this.

**`lerobot/smolvla_base` has no gripper dimension** (declared action width 6).
`smolvla_server` detects this and holds the observed gripper value rather than
inventing one, so the base model can position the arm but not grasp. A model
fine-tuned on 7-dim data recorded from this cell drives the gripper normally.

**Inference is ~950 ms on a 2060**, so the loop runs at ~1 Hz. That is why
`smolvla_policy.launch.py` sets `control_rate: 1.0` and
`observation_timeout: 3.0` — with the 10 Hz GR00T defaults the watchdog would
trip on its own latency. Each reply covers `execution_horizon * action_dt` =
0.8 s of motion, so the arm keeps moving between inferences.

**OpenVLA is slow on this card** (~2.8-4.8 s per forward pass at 4-bit on a
2060), so the loop runs at ~0.2 Hz. That is why `openvla_policy.yaml` republishes
the last twist at 20 Hz: `moveit_servo` halts if commands stop for 0.25 s, and
without republishing a 5 s inference gap would make the arm stutter. The held
twist expires after `twist_hold_time` so a dead policy cannot leave it drifting.

### Fixed since the first version

- **gripper joints jammed after a few grasps** — the finger joint limits were
  exactly the commanded range, so every full open/close drove the joint into a
  mechanical stop until the physics engine's joint-limit constraint locked it.
  The limits now sit 5 mm beyond the usable travel. 24/24 cycles clean.
- **`friction="1.0"` on the finger joints** — `gz_ros2_control` drives position
  interfaces with a *force*, so 1 N of Coulomb friction stalled a 50 g finger
  while every layer above reported success. Now 0, matching `ur_description`.
- **gripper goal tolerance (0.05) exceeded the joint's whole 0.04 stroke**, so
  the controller reported "already at goal" instantly and never moved.
- **`goal_time: 0.0`** means *wait forever*; a finger stalled on a cube hung
  the action indefinitely. Now 0.5 s.
- **`tcp_link` sat at the fingertips**, so "move the TCP to the object centre"
  drove the fingers through the table. It is now the centre of the grasp region.
- **`rclpy.spin_until_future_complete` per call** built a throwaway executor
  each time and dropped action goal responses. `MoveItHelper` now owns one
  persistent executor.
- **table collision boxes overlapped the arm pedestal**, which would mark the
  robot permanently in self-collision.
- **`switch_command_type` deadlocked the enable service** — the client shared a
  MutuallyExclusiveCallbackGroup with the `~/enable` callback that blocks on
  its response, so the reply could never be processed. It now has its own
  reentrant group.
- **the goal marker acted on a stale pose** — it read the server's stored
  marker pose instead of the pose carried in the feedback event.
- **`kill_stack.sh` killed its own caller** — its `ros2 launch groot_arm`
  pattern matched the command line of the shell about to start that launch.
  Patterns now match executable paths and the script skips its own ancestry.

## Running out of VRAM

The card is shared with everything else on the desktop, and 6 GB does not leave
much slack. Check before blaming the stack — but check *properly*:

```bash
nvidia-smi        # read the Processes block at the bottom, including 'G' rows
```

`--query-compute-apps` alone is **misleading**: it lists CUDA contexts only, so
the desktop compositor, your browser and Gazebo's renderer are invisible to it.
On this machine those account for over a gigabyte, which produced the actively
wrong report that the only process on the GPU was the policy server itself.

Measured on a 6 GB RTX 2060 (5.60 GiB usable):

| Consumer | VRAM |
|---|---|
| desktop (Xorg + gnome-shell + browser) | ~0.8-1.1 GiB |
| Gazebo headless, two cameras | ~0.2 GiB |
| SmolVLA | 0.91 GiB |
| OpenVLA-7B 4-bit, weights only | 4.32 GiB |
| OpenVLA-7B 4-bit, weights + inference activations | **~4.7 GiB** |

So **SmolVLA fits comfortably; OpenVLA is marginal** — it wants about 4.8 GiB
free while a typical desktop leaves 4.5-4.6 GiB. Free a few hundred MiB of
GPU-using apps and it runs.

Both servers handle a busy GPU rather than dying part way through loading:

* **smolvla_server** falls back to CPU automatically, naming what holds the GPU.
  It still works — about 25 s per inference instead of 850 ms.
* **openvla_server** checks free VRAM up front and refuses with the exact
  shortfall in MiB plus concrete options, instead of crashing after 30 s of
  loading. `--gpu-headroom-gib` tunes the threshold for a marginal fit.

A 4-bit model is **all-or-nothing on the GPU**. Partial CPU offload loads
without complaint and then fails on the first inference with *"Blockwise 4bit
quantization only supports 16/32-bit floats, but got torch.uint8"* -
bitsandbytes cannot dequantize blocks living in host memory. The server
therefore refuses rather than offloading.

## Who owns the arm

The policy streams trajectories straight to `arm_controller`; MoveIt does too.
If both run at once, MoveIt's execution is preempted and returns
`CONTROL_FAILED`, which reads like a bug but is really contention.

So manual control is **interlocked**: while the policy is armed, the goal
marker refuses with a clear message and the panel greys out its motion buttons
with a tooltip saying why. Disable the policy to drive the arm by hand.

## Troubleshooting

**Motions fail with `MoveItErrorCode=-4` (CONTROL_FAILED), or you see
"there may be more than one action server for /move_action".** You have two
stacks running. `ros2 launch` does not always reap its children, and a leftover
`move_group` will happily accept goals while its controllers are gone:

```bash
ros2 run groot_arm_bringup kill_stack.sh     # or src/groot_arm_bringup/scripts/kill_stack.sh
```

**Servo rejects everything with "Command type has not been set".** Jazzy's
`moveit_servo` requires a command type before it accepts input, and there is no
`start_servo` service (that was the Humble API):

```bash
ros2 service call /servo_node/switch_command_type \
    moveit_msgs/srv/ServoCommandType "{command_type: 1}"   # 1 = TWIST
```

**The policy will not enable.** `~/enable` pings the server first and refuses if
it is unreachable. Check with `ros2 run groot_vla probe_server --host ...`.

**Nothing moves and there are no errors.** Check `actual` against `output`:

```bash
ros2 topic echo /gripper_controller/controller_state --once
```

If `output` tracks but `actual` does not, it is physics, not ROS. Note that
`gz_ros2_control` drives position command interfaces with a *force* — non-zero
`<dynamics friction="...">` on a light joint will stall it completely while
every layer above reports success. `ur_description` uses `friction="0"` on every
arm joint for exactly this reason.
