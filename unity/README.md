# Unity Robotics bridge

Runs Unity as an alternative simulator to Gazebo, feeding the **same** ROS 2
topics. The policy stack cannot tell which simulator produced an observation,
which is the point: you can compare renderers, or train on Unity's HDRP output
and evaluate in Gazebo.

```
Unity (ROS-TCP-Connector) ──TCP:10000──> ros_tcp_endpoint ──> the same ROS graph
```

## What is already done, and what you must do

| Part | Status |
|---|---|
| `ros_tcp_endpoint` cloned, patched for Jazzy, builds, listens on :10000 | **done and verified** |
| `unity_bridge.launch.py` | **done and verified** |
| `Packages/manifest.json` pinning ROS-TCP-Connector + URDF-Importer | **done** |
| `Assets/Scripts/GrootArmBridge.cs` | **done**, not compiled - needs the Editor |
| Unity Hub + Editor installed | **you** - ~10 GB, needs a Unity account |
| Importing the URDF, building the scene | **you** - Editor work |

Unity Editor cannot be installed non-interactively: it requires a Unity ID and
licence activation. Nothing here pretends otherwise.

## 1. Install Unity (once)

```bash
# Unity Hub, from Unity's apt repository
sudo sh -c 'echo "deb https://hub.unity3d.com/linux/repos/deb stable main" \
  > /etc/apt/sources.list.d/unityhub.list'
wget -qO - https://hub.unity3d.com/linux/keys/public | gpg --dearmor \
  | sudo tee /usr/share/keyrings/Unity_Technologies_ApS.gpg > /dev/null
sudo apt update && sudo apt install unityhub
```

Then in Unity Hub: sign in, and install **Unity 2022.3 LTS** or **Unity 6**.
Add the *Linux Build Support* module if you want standalone builds.

Personal licences are free for individuals and small revenue.

## 1b. Editor version and render pipeline

`ProjectSettings/ProjectVersion.txt` pins **Unity 6000.5.9f1**. Unity Hub reads
the editor version from that file alone - without it the project lists as
"Editor version: Unknown" with a warning and Hub cannot choose an Editor.

If you have a different Editor installed, edit that file to match, or let Hub
upgrade the project when it offers to.

The manifest deliberately does **not** pin HDRP. HDRP is worth having (global
illumination, area lights, reflections - the reason to use Unity at all), but
enabling it is a multi-step Editor task: install the package, create a pipeline
asset, assign it in Graphics settings, then convert materials. Doing that before
the project opens once tends to produce a project that will not open at all.
Get the bridge working on the built-in pipeline first, then add HDRP through
**Window -> Package Manager -> High Definition RP** and run its wizard.

### If the project opens in Safe Mode

Read *which package* the errors are in before assuming the robotics packages are
at fault. A wall of `CS0619: 'TreeView' is obsolete` in
`Library/PackageCache/com.unity.inputsystem@...` is the Input System, not
ROS-TCP-Connector: Unity 6.5 deprecated `TreeView`, `TreeViewItem` and
`TreeViewState` in favour of generic versions, and older Input System releases
still use the non-generic ones.

The manifest no longer requests Input System at all. Nothing here uses it -
`GrootArmBridge.cs` reads no input - and it was the sole source of ~90 compile
errors. If it reappears, check `Packages/packages-lock.json`: an entry with
`"depth": 0` came from the manifest, a deeper one was pulled in by another
package.

To recover after editing the manifest, exit Safe Mode and let Unity re-resolve.
If the errors persist, the stale copy is still cached:

```bash
rm -rf unity/Library/PackageCache/com.unity.inputsystem@*
```

then reopen the project.

The robotics packages are pulled from their **default branches** rather than the
v0.7.0 / v0.5.2 tags. Those tags date from 2022 and predate Unity 6; the tips
are more likely to compile. If Unity reports compile errors in
`Unity.Robotics.*`, that is the cause - pin back to a tag that matches your
Editor, or use Unity 2022.3 LTS instead.

## 2. Open this project

Unity Hub -> **Add** -> select this `unity/` directory. On first open it
resolves `Packages/manifest.json`, which pulls ROS-TCP-Connector and
URDF-Importer straight from GitHub. That takes a few minutes.

## 3. Import the robot

The URDF must be flattened first, because Unity's importer does not run xacro:

```bash
cd ..                       # the workspace root
source install/setup.bash
xacro install/groot_arm_description/share/groot_arm_description/urdf/groot_arm.urdf.xacro \
    ur_type:=ur5e name:=groot_arm sim_gazebo:=false > unity/Assets/groot_arm.urdf
```

In Unity: right-click `groot_arm.urdf` -> **Import Robot from Selected URDF
file**. Set **Convex Decomposer: VHACD** for usable gripper collisions.

### Mesh placement matters, and is not obvious

The URDF references meshes as
`package://ur_description/meshes/ur5e/collision/base.stl`. URDF-Importer
resolves a `package://<pkg>/...` URI to `Assets/<pkg>/...`, so the meshes must
land at **`Assets/ur_description/meshes/`** — not `Assets/meshes/`:

```bash
mkdir -p unity/Assets/ur_description
cp -r /opt/ros/jazzy/share/ur_description/meshes unity/Assets/ur_description/
```

Two failure modes if this is wrong:

* `DirectoryNotFoundException: Could not find a part of the path
  .../Assets/ur_description/meshes/ur5e/collision/base.stl` — the import stops
  part-way, leaving a partial robot.
* `Assets/meshes cannot be created! It may already exist.` — `Assets/meshes` is
  where the importer writes its **own** generated primitives (`Cylinder.asset`
  for cylinder colliders). Putting the UR meshes there collides with its
  workspace.

If a previous attempt left meshes in the wrong place, remove them from
`Assets/meshes/` and re-import.

## 4. Build the scene (one click)

An empty scene after opening is expected - the project ships the URDF and the
scripts, not a scene.

**Robotics -> Build GR00T Arm Scene**

That creates a floor and key light, places both cameras, adds `RosBridge` with
`GrootArmBridge`, and resolves the six `ArticulationBody` references **by name**
into the order the bridge indexes by. Then save the scene (Ctrl+S) as
`Assets/GrootArm.unity`.

It is scripted rather than left as Inspector work because two of those steps are
easy to get wrong in a way that looks fine:

* **Axis conventions differ.** ROS is Z-up right-handed (x forward, y left,
  z up); Unity is Y-up left-handed. The mapping is
  `(x, y, z)_ros -> (-y, z, x)_unity`. Typing URDF numbers straight into the
  Inspector gives a camera pose that is almost right and silently wrong. The
  builder also aims cameras with `LookAt` at a target point rather than a
  quaternion, because a quaternion cannot be checked by eye.
* **Joint order is positional.** The bridge indexes `armJoints` against a fixed
  name list. Six bodies dragged into the array in the wrong order produce a
  robot that moves confidently in the wrong directions - no error, just
  nonsense.

Check the Console: it logs which joints were resolved. If any are missing,
assign those slots by hand.

## 4b. Wire up the bridge manually (only if the builder failed)

1. Create an empty GameObject, attach `GrootArmBridge.cs`.
2. Assign **wristCamera** and **sceneCamera** (two Cameras placed to match
   `groot_arm.urdf.xacro`: wrist on the gripper, scene on the mast).
3. Assign **armJoints** in URDF order: `shoulder_pan`, `shoulder_lift`,
   `elbow`, `wrist_1`, `wrist_2`, `wrist_3`.
4. **Robotics -> ROS Settings**: protocol `ROS2`, IP = the machine running the
   endpoint, port `10000`.

## 5. Run

```bash
# terminal 1 - the bridge
ros2 launch groot_arm_bringup unity_bridge.launch.py

# terminal 2 - MoveIt and the policy, with Gazebo NOT started
ros2 launch groot_arm_moveit_config move_group.launch.py use_sim_time:=false
ros2 launch groot_vla smolvla_policy.launch.py
```

Then press **Play** in Unity. Check the data is flowing:

```bash
ros2 topic hz /scene_camera/image_raw
ros2 topic echo /joint_states --once
```

## Things that will bite you

**Image rows are flipped.** Unity's texture origin is bottom-left, ROS images
are top-left. `GrootArmBridge.cs` flips them on publish. Get this wrong and you
get a vertically mirrored image that looks plausible and silently poisons
training.

**Unity articulation drives are in degrees**, ROS joints are in radians. The
bridge converts; anything you add must too.

**`use_sim_time` must be false** when Unity drives, unless you also publish
`/clock` from Unity. Gazebo publishes `/clock`; Unity does not by default, so
nodes left on sim time will sit waiting for a clock that never ticks.

**Do not run Gazebo and Unity at once** on the same `ROS_DOMAIN_ID`. Both
publish `/joint_states` and `/wrist_camera/image_raw`, and the policy will
receive an interleaved mixture of two different worlds.

## Is Unity worth it here?

Honestly: **only if you need the rendering.** Unity's HDRP is substantially
better looking than Ogre2 - real global illumination, area lights, proper
reflections - which is worth having for a vision policy.

The costs are real though. There is no `ros2_control` in Unity, so its
articulation drives are not the same controller the real UR5e uses; MoveIt's
trajectory execution is approximated by the bridge rather than executed
faithfully. The Gazebo path is closer to the real robot's control stack. Use
Unity for visual variety in the *dataset*, and Gazebo for anything about
control fidelity.
