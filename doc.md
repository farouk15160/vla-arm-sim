# doc.md — the code, the data, and the training

`doco.md` documents the **system as deployed**: the node graph, the control
paths, the safety envelope, measured numbers. This document explains **why the
code is shaped the way it is**, and walks the training pipeline end to end —
from a teleoperated episode on disk to a fine-tuned SmolVLA checkpoint serving
actions over ZeroMQ.

Read `doco.md` if you want to operate the cell. Read this if you want to change
it, retrain it, or port it to another robot.

---

## 1. The one idea the whole design rests on

A vision-language-action model is a **function**, not a controller:

```
(images, proprioceptive state, natural-language instruction)  ->  action chunk
```

It has no notion of joint limits, no notion of the table, no notion that the
gripper is about to close on the robot's own wrist. It emits numbers at
whatever rate the GPU allows — for SmolVLA on this hardware, roughly one call
every 850 ms.

A robot arm needs commands at 100 Hz+, needs them to respect joint limits, and
needs something to hold the last valid command when the policy is thinking or
has crashed.

**Everything in `groot_vla/` exists to bridge that gap.** The VLA is treated as
an untrusted oracle: its output is decoded, dimension-checked, rate-limited,
joint-limit-clamped, workspace-checked, and only then handed to a real-time
controller that interpolates it up to servo rate. If the policy emits `NaN`,
emits the wrong number of dimensions, or stops replying, the arm holds
position — it does not lurch.

This is the difference between a demo and a test cell.

---

## 2. Process and package layout

Four packages, each with one job.

| Package | Job |
|---|---|
| `groot_arm_description` | URDF/xacro, meshes, materials, Gazebo world. The single source of truth for what the robot *is*. |
| `groot_arm_moveit_config` | SRDF, kinematics, OMPL/Pilz planners, Servo config, controller mapping. |
| `groot_arm_bringup` | Launch files, RViz configs, helper scripts. The only package a user runs directly. |
| `groot_vla` | Everything policy-related: client, servers, observation/action conversion, data collection, dataset export. |

The reason `groot_vla` is a separate package and not part of bringup is
**dependency isolation**. `groot_vla`'s ROS-side code (`policy_node`,
`observation_builder`, `action_mapper`, `groot_client`) imports **numpy and
nothing else** — no torch, no transformers, no lerobot. The heavy ML stack
lives only in the policy *servers*, which run in a separate virtualenv
(`~/vla_venv`) as plain Python processes, not ROS nodes.

That split matters more than it looks:

- ROS 2 Jazzy ships system Python 3.12 with system numpy. LeRobot and
  transformers want their own pinned torch. Installing them into the ROS
  environment breaks the ROS environment.
- The policy server can be restarted, swapped (mock → SmolVLA → OpenVLA), or
  moved to a different machine, without rebuilding or restarting the ROS stack.
- A GPU OOM kills the server, not the robot. The arm holds position and
  `policy_node` reports the failure.

The two halves talk over **ZeroMQ REQ/REP with msgpack framing** — a
deliberately boring, language-agnostic, process-boundary-friendly protocol.

---

## 3. Data flow, one control step

```
Gazebo / Unity
   │  sensor_msgs/Image  ×2 (wrist_view, ego_view)
   │  sensor_msgs/JointState
   ▼
policy_node                          ← ROS 2, numpy only, no torch
   │  ObservationBuilder
   │    · decode Image → HWC uint8 RGB
   │    · resize to 224×224
   │    · joints → state vector (arm 6 + gripper 1, gripper normalised to [0,1])
   │    · assemble nested dict {video, state, language}
   ▼  msgpack-numpy over ZMQ REQ
smolvla_server                       ← ~/vla_venv, torch + lerobot
   │    · rename camera keys to what the checkpoint expects
   │    · pad/trim state to the checkpoint's declared width
   │    · normalise with the checkpoint's own stats
   │    · SmolVLA forward → action chunk (50, 7)
   ▼  msgpack-numpy over ZMQ REP
policy_node
   │  ActionMapper
   │    · decode + dimension-check
   │    · rate-limit ≤0.15 rad per step, FROM THE LIVE JOINT STATE
   │    · clamp to joint limits
   │    · workspace bounds check on the TCP via TF
   ▼  trajectory_msgs/JointTrajectory
arm_controller (JointTrajectoryController)
   │    · interpolates the chunk to controller rate
   ▼
ros2_control → Gazebo physics
```

Three details in that diagram are non-obvious and each one was a bug first.

### 3.1 Camera order is declared, not discovered

`ObservationBuilder.build()` emits video modalities **in the order the
`cameras` list declares**, not in the order of the `images` dict.

The images dict is filled by subscription callbacks, so its insertion order
depends on which camera happened to publish first after startup — which varies
run to run. A server that maps incoming views onto its own image slots
positionally would silently swap wrist and scene between runs, destroying the
correspondence with the training data. The policy would still run. It would
just be wrong, in a way that looks like "the model is bad".

### 3.2 The state is 6-dim, the action is 7-dim

Deliberate asymmetry, and it is the single most common thing to get wrong.

- **State = 6** — the arm joints only. The finger position is *not* fed back.
- **Action = 7** — 6 arm joints + 1 gripper command.

Feeding the gripper into the state produced a checkpoint that loaded fine and
then died on the first inference with `size of tensor a (6) must match tensor b
(7)`. The exporter now hard-codes `STATE_DIM = 6` / `ACTION_DIM = 7` so the
dataset cannot drift from the policy config.

The server defends the boundary anyway: it reads
`policy.config.input_features["observation.state"]` from the **checkpoint**,
and pads or trims whatever arrives to that width. A dataset/policy mismatch
degrades accuracy instead of crashing mid-episode.

### 3.3 Rate limiting starts from the live joint state, not the snapshot

`_step()` captures the joint positions, runs inference, then **re-reads the
joint state** before building the trajectory.

Without that re-read, the snapshot is a full inference period old — ~850 ms for
SmolVLA, several seconds for OpenVLA — and the arm has moved. Every chunk began
with a step discontinuity the controller could not track, producing a stream of
`Holding position due to state tolerance violation` (error 0.500069 against a
0.5 tolerance — right at the edge, which is exactly what a stale-by-one-period
snapshot looks like). Re-reading live took six violations per run to zero.

This is a general lesson for slow policies: **a slow policy's observation is
stale by construction; never use it as the base for a relative command.**

---

## 4. The wire protocol

`groot_client.py` speaks msgpack with a `msgpack-numpy` codec, and carries its
own byte-identical fallback implementation for when `msgpack-numpy` is not
installed on one side. The fallback was verified byte-for-byte against the real
library across six payload types — arrays of every dtype the pipeline uses,
nested dicts, and plain scalars — because a codec that is *almost* compatible
is worse than no codec: it works in testing and corrupts one field in
production.

Endpoints: `get_action`, `ping`, `get_modality_config`, `reset`, `kill_server`.

`get_modality_config` matters for debugging: it asks the server what shapes it
expects, so a shape mismatch can be diagnosed without reading the checkpoint.

REQ/REP is strictly lockstep, which is the right choice here — the policy is
the bottleneck and there is nothing useful to pipeline. It also means a crashed
server is detected immediately (the socket is reset and the arm holds) rather
than silently dropping requests into a queue.

---

## 5. Collecting data

```bash
# terminal 1 — simulator
ros2 launch groot_arm_bringup system.launch.py policy:=none

# terminal 2 — recorder
ros2 run groot_vla episode_recorder --ros-args \
  -p output_dir:=data/demos/my_run -p task:="pick up the red cube and place it in the tray"

# terminal 3 — drive it (scripted, or by hand in RViz)
ros2 run groot_vla pick_place_demo
```

Each episode lands as a directory:

```
episode_000/
  meta.json                    fps, cameras, arm_joints, gripper_joint, task, image_size
  frames.jsonl                 one row per timestep: state.*, action.*, frame_index
  wrist_view/frame_000000.png
  ego_view/frame_000000.png
```

PNG, not video, at collection time. Encoding is deferred to export so a
crashed or discarded episode costs nothing, and so frames can be inspected
individually when an episode looks wrong.

### 5.1 Domain randomisation

`domain_randomizer.py` re-rolls the scene between episodes: lighting colour and
intensity, table material, object shapes and colours, object positions.

The reason is **sim-to-real** and also plain **sim-to-sim** robustness. A policy
trained on one fixed scene learns the scene, not the task — it will reach for
where the red cube *always was*, not where the red cube *is*. Randomisation
forces the visual features that survive across scenes (a cube-shaped thing that
is red, relative to the gripper) to be the ones that carry the signal.

Two bugs shaped this module and both are worth knowing:

**Static model respawn loses collision.** In gz-sim, deleting and respawning a
static model drops every object resting on it to `z=0.02`. The randomiser
therefore never replaces the table — it overlays a **visual-only** "table cloth"
model and leaves the collision geometry untouched.

**`clear_objects` was stateless.** It removed only names in `self._spawned`,
which is empty in a fresh process. Leftover objects from a previous run blocked
new spawns, so the scene randomised its *colours* while the objects never
actually moved — the dataset looked varied and was not. It now sweeps the whole
`obj_*` namespace and retries spawns.

Both failures are of the same species: **they produce a plausible-looking
dataset.** Nothing errors. You find out at evaluation time, after the training
run.

### 5.2 Parallel collection

`collect_parallel.sh` runs N headless Gazebo instances on separate
`ROS_DOMAIN_ID`s, each writing to its own output directory; `merge_demos.py`
renumbers and concatenates them. Verified with 2 workers merging correctly.

Gazebo's `real_time_factor: 0` in the world file means "run as fast as physics
allows" rather than throttling to wall clock, which is the single biggest
collection speedup and costs nothing since no human is watching.

---

## 6. Export to LeRobot format

```bash
~/vla_venv/bin/python -m groot_vla.export_lerobot \
  --input data/demos/pickplace_randomized \
  --output data/datasets/pickplace_randomized \
  --repo-id local/groot_ur5e
```

The exporter uses **LeRobot's own `LeRobotDataset.create()` / `add_frame()` /
`save_episode()` API** rather than writing the on-disk format directly.

That was a rewrite. The first version hand-wrote the v2.1 layout, which LeRobot
0.4.x rejects — it needs v3.0, and their official converter requires a round
trip through the HuggingFace Hub, which is unacceptable for local data. Going
through the API means the format version is whatever the installed LeRobot
wants, and it stays correct across upgrades.

`use_videos=True` makes LeRobot encode the PNG sequences to MP4 per episode.
This is not just disk savings — LeRobot's dataloader decodes video ranges
directly, which is substantially faster than reading tens of thousands of
individual PNGs during training.

**The dataset currently on disk:**

| | |
|---|---|
| episodes | 41 |
| frames | 6378 |
| fps | 10 |
| state | `observation.state` (6,) — arm joints |
| action | `action` (7,) — arm joints + gripper |
| cameras | `observation.images.wrist_view`, `observation.images.ego_view` |
| version | v3.0 |

41 episodes is a **smoke-test dataset, not a training dataset.** Published
SmolVLA fine-tunes use 50–100+ episodes per task, and this task has a
randomised scene, which raises the requirement rather than lowering it. Expect
the current run to learn the *motion prior* — reach toward the table, close near
contact — without reliably completing the task. Target ~150 episodes for a
result worth evaluating.

---

## 7. SmolVLA, concretely

SmolVLA is a 450M-parameter VLA from the LeRobot team:

- **Vision-language backbone** — SmolVLM2-500M-Video-Instruct, **truncated to
  its first 16 layers**. The upper layers of a VLM specialise in language
  generation, which a policy does not need; the lower layers carry the
  grounded visual-semantic features that do transfer. Truncation is where most
  of the parameter savings come from.
- **Action expert** — a flow-matching head that maps VLM features + state to an
  action chunk.
- **Chunk size 50** at 10 fps = 5 seconds of future actions per inference.

Flow matching rather than autoregressive decoding is what makes 450M usable at
robot rates: the whole 50-step chunk is produced in ~10 denoising steps in one
forward pass, instead of 350 sequential token decodes.

From the live run:

```
num_learnable_params = 99,880,992  (100M)
num_total_params     = 450,046,176 (450M)
```

**Only 100M of 450M parameters are trained.** The VLM backbone is frozen; the
action expert and the projection layers are not. That is why this fits in 6 GB
of VRAM at all — optimiser state (Adam keeps two moments per trainable
parameter) is the dominant memory cost in fine-tuning, and freezing the
backbone removes 78% of it.

### 7.1 Execution horizon: why we throw most of the chunk away

The policy emits 50 steps. `execution_horizon` executes only the first few
before the next inference replaces them.

Open-loop execution of a full 5-second chunk means the arm is blind for 5
seconds — any error compounds with no correction. Short horizon with frequent
re-planning is closed-loop and self-correcting; it costs inference throughput,
which is the right trade when the alternative is driving into the table.

### 7.2 Camera key renaming

The training config carries:

```json
"rename_map": {
  "observation.images.wrist_view": "observation.images.camera1",
  "observation.images.ego_view":   "observation.images.camera2"
}
```

`smolvla_base` was pretrained with cameras named `camera1`/`camera2`. Renaming
at the config level lets our descriptive names survive in the dataset — where
they aid debugging — while presenting the checkpoint the keys it was pretrained
with, so pretrained visual features actually get reused instead of being
initialised fresh.

---

## 8. Training

### 8.1 Starting a run

```bash
~/vla_venv/bin/lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id=local/smolvla_ur5e --policy.push_to_hub=false \
  --dataset.repo_id=local/groot_ur5e \
  --dataset.root=data/datasets/pickplace_randomized \
  --batch_size=4 --steps=20000 \
  --output_dir=data/checkpoints/pickplace
```

### 8.2 Resuming a stopped run

```bash
~/vla_venv/bin/lerobot-train \
  --config_path=data/checkpoints/pickplace/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

`checkpoints/last` is a symlink to the newest step directory. `--resume=true`
restores the optimiser moments, the LR-scheduler position, **and the RNG state**
— so the resumed run sees the same data order it would have seen had it never
stopped. A resume that only restored weights would re-warm the optimiser and
re-shuffle, which shows up as a loss spike at the resume point.

**Training can always be stopped and resumed.** `save_freq=500` means you lose
at most 500 steps.

### 8.3 The hyperparameters, and why

| Setting | Value | Reason |
|---|---|---|
| `batch_size` | 4 | VRAM-bound. 6 GB total, ~4.3 free with a desktop running. |
| `steps` | 20000 | ~12.5 epochs over 6378 frames at batch 4. |
| `optimizer.lr` | 1e-4 | LeRobot's SmolVLA preset. Fine-tuning, not pretraining. |
| `scheduler` | cosine decay, 1000 warmup, 30000 decay | Warmup avoids wrecking pretrained features in the first steps, when gradients from a randomly-initialised action head are largest. |
| `chunk_size` / `n_action_steps` | 50 / 50 | 5 s at 10 fps. |
| `n_obs_steps` | 1 | Single frame in, no history. |
| `save_freq` | 500 | Cheap insurance. |
| `num_workers` | 4 | Dataloader; video decode is CPU-bound. |
| `seed` | 1000 | Reproducibility. |

Note the decay schedule runs to 30000 while training stops at 20000 — the run
ends before the LR fully decays, on purpose. It leaves room to extend the run
without a schedule discontinuity.

### 8.4 Reading the loss

Flow-matching loss is a **velocity-prediction MSE in noise space**, not a task
metric. It falls fast early and then goes nearly flat while behaviour is still
improving. Do not read it as a success rate. The only measurement that means
anything is running the checkpoint on the cell and counting completed picks.

### 8.5 Serving a fine-tuned checkpoint

```bash
ros2 launch groot_arm_bringup system.launch.py policy:=smolvla \
  policy_model:=data/checkpoints/pickplace/checkpoints/last/pretrained_model
```

The server reads the checkpoint's own normalisation statistics and feature
declarations, so nothing needs to be configured to match — a mismatch is
detected rather than silently applied.

### 8.6 LoRA

Deferred by choice. With the VLM backbone already frozen, LoRA's marginal
saving on the remaining 100M action expert is small, and it adds an adapter
merge step to every deployment. Worth revisiting only if the backbone needs
unfreezing.

---

## 9. VRAM, honestly

6 GB total on this machine, and the desktop takes 1.0–1.5 GB.

| Workload | Needs | Verdict |
|---|---|---|
| SmolVLA inference | ~0.9 GB | Comfortable |
| SmolVLA training, batch 4 | ~2.3 GB | Fits with the desktop running |
| OpenVLA-7B, NF4 4-bit | ~4.65 GB | Marginal — fails if a browser is open |
| GR00T N1.7-3B | >6 GB | Does not fit |

`openvla_server.py` performs a **VRAM preflight and refuses to start** if free
memory is below `4.4 GiB weights + 0.2 GiB activations`, printing the actual
GPU consumers and the alternatives.

This is deliberate, and the reason is specific: a 4-bit model **cannot be
partially offloaded to CPU**. Accelerate accepts the offload, the model loads
looking healthy, and then the first inference dies with `Blockwise 4bit
quantization only supports 16/32-bit floats, but got torch.uint8`. Refusing up
front with an actionable message beats a successful load followed by a failure
several minutes later in a different part of the stack.

Related: `nvidia-smi --query-compute-apps` lists **CUDA contexts only**. The
desktop compositor, browser, and Gazebo hold *graphics* contexts and never
appear there. The preflight parses the full `nvidia-smi` table instead, which is
why it can name the browser tab that is eating your training run.

---

## 10. If you put this on a real robot

Directly? It would move, and it would be wrong.

The gap is not the code — the ROS interfaces are the same ones a real UR5e
exposes. The gap is that the policy has only ever seen this simulator:

- **Visuals.** Even with Ogre2 and PBR materials, rendered images differ from a
  real camera in noise, motion blur, rolling shutter, exposure response, and
  lighting. Domain randomisation narrows this; it does not close it.
- **Contact.** Gazebo's contact model is a convenient fiction. A real cube slips,
  deforms slightly, and rocks. The finger stall point that lands at exactly
  0.0080 m in sim will not be exact on hardware.
- **Latency.** Real cameras and real controllers add delay the sim does not.

The realistic path is sim pretraining followed by real-data fine-tuning: collect
20–50 real teleoperated episodes, resume training from the sim checkpoint. The
sim work buys the motion prior and the language grounding; the real data
supplies the contact dynamics and the true visual distribution.

Before any of that: the workspace bounds in `policy_node` are checked against
the TCP via TF and **halt the arm on violation**. On hardware, keep that
enabled, keep the rate limit, and keep a hand on the e-stop. The safety
envelope in this codebase was written assuming the policy will eventually emit
something insane, because it will.

---

## 11. Cross-references

- **Operating the cell, node/topic graph, measured numbers** → `doco.md`
- **Every runnable command** → `README.md`
- **Unity Robotics setup** → `unity/README.md`
