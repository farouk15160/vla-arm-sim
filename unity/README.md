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

The UR meshes live in `/opt/ros/jazzy/share/ur_description/meshes/`; copy that
folder next to the URDF if Unity reports missing meshes, since it cannot follow
`package://` URIs.

## 4. Wire up the bridge

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
