// One-click scene construction: Robotics -> Build GR00T Arm Scene
//
// Everything here is mechanical but easy to get wrong by hand, which is exactly
// why it is scripted:
//
//   * ROS and Unity use different axis conventions. ROS is Z-up right-handed
//     (x forward, y left, z up); Unity is Y-up left-handed (x right, y up,
//     z forward). The mapping is (x, y, z)_ros -> (-y, z, x)_unity. Placing a
//     camera by typing numbers from the URDF into the Inspector produces a view
//     that looks almost right and is silently wrong.
//   * The bridge indexes armJoints POSITIONALLY against a fixed name list.
//     Dragging six ArticulationBodies into an array in the wrong order gives a
//     robot that moves confidently in the wrong directions.
//   * Camera aim is expressed as "look at this point" rather than a rotation,
//     because a quaternion typed into the Inspector is unverifiable by eye.
//
// Run it AFTER importing the URDF (right-click groot_arm.urdf -> Import Robot
// from Selected URDF file, Convex Decomposer: VHACD).

using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEngine;

public static class BuildGrootScene
{
    // Joint order must match GrootArmBridge.JointNames and the URDF.
    static readonly string[] JointNames =
    {
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    };

    /// ROS (x fwd, y left, z up, right-handed) -> Unity (x right, y up, z fwd, left-handed).
    static Vector3 RosToUnity(float x, float y, float z) => new Vector3(-y, z, x);

    [MenuItem("Robotics/Build GR00T Arm Scene")]
    public static void Build()
    {
        var root = FindRobotRoot();
        if (root == null)
        {
            EditorUtility.DisplayDialog(
                "Robot not found",
                "Import the robot first:\n\n" +
                "  Assets/groot_arm.urdf -> right-click ->\n" +
                "  Import Robot from Selected URDF file\n\n" +
                "Set Convex Decomposer to VHACD, or the gripper fingers get box\n" +
                "colliders that cannot grasp.",
                "OK");
            return;
        }

        BuildEnvironment();
        BuildWorld();
        ConfigureRobot(root);
        var wrist = BuildWristCamera(root);
        var scene = BuildSceneCamera();
        WireBridge(root, wrist, scene);

        Debug.Log("[GR00T] Scene built. Start unity_bridge.launch.py, then press Play.");
        EditorUtility.DisplayDialog(
            "Scene built",
            "Robot, cameras and bridge are wired.\n\n" +
            "Next:\n" +
            "  1. ros2 launch groot_arm_bringup unity_bridge.launch.py\n" +
            "  2. Robotics -> ROS Settings: ROS2, 127.0.0.1, port 10000\n" +
            "  3. Press Play\n\n" +
            "Verify with:  ros2 topic hz /scene_camera/image_raw",
            "OK");
    }

    static GameObject FindRobotRoot()
    {
        // The importer names the root after the URDF's <robot name="...">.
        var byName = GameObject.Find("groot_arm");
        if (byName != null) return byName;

        // Fall back to any object owning the shoulder joint.
        // Unity 6 deprecated the FindObjectsSortMode overload.
        return Object.FindObjectsByType<ArticulationBody>(FindObjectsInactive.Include)
            .Select(a => a.transform.root.gameObject)
            .FirstOrDefault(go => go.GetComponentsInChildren<ArticulationBody>()
                                    .Any(a => a.name.Contains("shoulder_pan")));
    }

    static void BuildEnvironment()
    {
        // A floor, so the scene is not an infinite void and shadows have
        // somewhere to land. The table itself comes from the URDF world in
        // Gazebo; here a plane is enough to sit the robot on.
        if (GameObject.Find("Floor") == null)
        {
            var floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
            floor.name = "Floor";
            floor.transform.position = Vector3.zero;
            floor.transform.localScale = new Vector3(2f, 1f, 2f);   // 20 x 20 m
        }

        if (GameObject.Find("Key Light") == null)
        {
            var lightObject = new GameObject("Key Light");
            var light = lightObject.AddComponent<Light>();
            light.type = LightType.Directional;
            light.intensity = 1.1f;
            light.shadows = LightShadows.Soft;
            light.color = new Color(1.0f, 0.96f, 0.9f);
            lightObject.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
        }
    }

    /// ROS size (sx, sy, sz) -> Unity localScale, following the same axis
    /// mapping as positions: Unity x is ROS y, Unity y is ROS z, Unity z is ROS x.
    static Vector3 RosSizeToUnity(float sx, float sy, float sz) => new Vector3(sy, sz, sx);

    static GameObject Box(string name, Vector3 rosPos, Vector3 rosSize, Color colour,
                          bool isStatic, float mass = 0f)
    {
        var existing = GameObject.Find(name);
        if (existing != null) Object.DestroyImmediate(existing);

        var box = GameObject.CreatePrimitive(PrimitiveType.Cube);
        box.name = name;
        box.transform.position = RosToUnity(rosPos.x, rosPos.y, rosPos.z);
        box.transform.localScale = RosSizeToUnity(rosSize.x, rosSize.y, rosSize.z);

        var renderer = box.GetComponent<Renderer>();
        var material = new Material(Shader.Find("Universal Render Pipeline/Lit")
                                    ?? Shader.Find("Standard"));
        material.color = colour;
        renderer.sharedMaterial = material;

        if (!isStatic)
        {
            var body = box.AddComponent<Rigidbody>();
            body.mass = mass;
            // Continuous detection: a 4 cm cube closed on by fast-moving fingers
            // will tunnel through them with the default discrete solver.
            body.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
        }
        return box;
    }

    /// Rebuilds the Gazebo tabletop: work surface, tray and three cubes, at the
    /// same coordinates worlds/tabletop.sdf uses, so a policy trained in one
    /// simulator sees the same layout in the other.
    static void BuildWorld()
    {
        var world = GameObject.Find("World") ?? new GameObject("World");

        var table = Box("work_table", new Vector3(0.50f, 0f, 0.575f),
                        new Vector3(1.4f, 1.8f, 0.05f),
                        new Color(0.72f, 0.60f, 0.45f), isStatic: true);
        table.transform.SetParent(world.transform, true);

        var pedestal = Box("table_base", new Vector3(0.50f, 0f, 0.275f),
                           new Vector3(1.2f, 1.6f, 0.55f),
                           new Color(0.26f, 0.27f, 0.30f), isStatic: true);
        pedestal.transform.SetParent(world.transform, true);

        var tray = Box("tray", new Vector3(0.50f, -0.35f, 0.605f),
                       new Vector3(0.22f, 0.22f, 0.01f),
                       new Color(0.55f, 0.57f, 0.60f), isStatic: true);
        tray.transform.SetParent(world.transform, true);

        // 50 g cubes, matching the SDF. Light enough for the gripper to lift,
        // heavy enough to sit still.
        var cubes = new (string name, Vector3 pos, Color colour)[]
        {
            ("red_cube",   new Vector3(0.45f,  0.16f, 0.62f), new Color(0.90f, 0.10f, 0.10f)),
            ("green_cube", new Vector3(0.55f, -0.05f, 0.62f), new Color(0.10f, 0.80f, 0.10f)),
            ("blue_cube",  new Vector3(0.38f, -0.18f, 0.62f), new Color(0.10f, 0.20f, 0.90f)),
        };
        foreach (var (name, pos, colour) in cubes)
        {
            var cube = Box(name, pos, new Vector3(0.04f, 0.04f, 0.04f), colour,
                           isStatic: false, mass: 0.05f);
            cube.transform.SetParent(world.transform, true);

            // High friction, matching the SDF surface parameters. Without it the
            // cube slides out of the fingers instead of being carried.
            var physicMaterial = new PhysicsMaterial($"{name}_grip")
            {
                dynamicFriction = 1.2f,
                staticFriction = 1.4f,
                frictionCombine = PhysicsMaterialCombine.Maximum,
            };
            cube.GetComponent<Collider>().sharedMaterial = physicMaterial;
        }
    }

    /// Makes the imported robot hold itself up and follow commands.
    ///
    /// URDF-Importer creates ArticulationBody drives with ZERO stiffness, so a
    /// freshly imported arm collapses under gravity the moment you press Play
    /// and never tracks a target. Gains have to be set explicitly; this is the
    /// single most common reason an imported robot "does not work".
    static void ConfigureRobot(GameObject root)
    {
        var rootBody = root.GetComponent<ArticulationBody>();
        if (rootBody != null) rootBody.immovable = true;   // bolt the base down

        int configured = 0;
        foreach (var body in root.GetComponentsInChildren<ArticulationBody>())
        {
            if (body.jointType != ArticulationJointType.RevoluteJoint &&
                body.jointType != ArticulationJointType.PrismaticJoint) continue;

            var drive = body.xDrive;
            // Stiff enough to hold a 6 kg forearm against gravity and track a
            // position target; damped enough not to oscillate.
            drive.stiffness = 10000f;
            drive.damping = 100f;
            drive.forceLimit = 1000f;
            drive.target = 0f;
            body.xDrive = drive;

            body.jointFriction = 0f;      // matches the URDF, which uses 0
            body.angularDamping = 0f;
            configured++;
        }
        Debug.Log($"[GR00T] Configured drives on {configured} joints " +
                  "(stiffness 10000, damping 100). Without this the arm collapses.");
    }

    static Camera BuildWristCamera(GameObject root)
    {
        var palm = root.GetComponentsInChildren<Transform>()
            .FirstOrDefault(t => t.name.Contains("gripper_base") || t.name.Contains("tool0"));
        if (palm == null)
        {
            Debug.LogWarning("[GR00T] No gripper_base_link/tool0 found; " +
                             "wrist camera parented to the robot root instead.");
            palm = root.transform;
        }

        var camera = MakeCamera("wrist_camera", 80f);
        camera.transform.SetParent(palm, false);
        // URDF: xyz="0.05 0 0.025" on the palm, aimed at the grasp point.
        camera.transform.localPosition = RosToUnity(0.05f, 0f, 0.025f);

        // tcp_link sits at palm_depth + finger_length/2 = 0.0825 m along tool +z.
        var target = palm.TransformPoint(RosToUnity(0f, 0f, 0.0825f));
        camera.transform.LookAt(target, palm.up);
        return camera;
    }

    static Camera BuildSceneCamera()
    {
        var camera = MakeCamera("scene_camera", 66f);
        // Mast at ROS world (-0.50, 0.85, 1.70), aimed at the cube field.
        camera.transform.position = RosToUnity(-0.50f, 0.85f, 1.70f);
        camera.transform.LookAt(RosToUnity(0.46f, 0f, 0.62f), Vector3.up);
        return camera;
    }

    static Camera MakeCamera(string name, float fieldOfView)
    {
        var existing = GameObject.Find(name);
        if (existing != null) Object.DestroyImmediate(existing);

        var cameraObject = new GameObject(name);
        var camera = cameraObject.AddComponent<Camera>();
        camera.fieldOfView = fieldOfView;
        camera.nearClipPlane = 0.02f;
        camera.farClipPlane = 50f;
        // The bridge assigns its own RenderTexture per frame; leaving one here
        // would mean this camera never draws to the Game view.
        camera.targetTexture = null;
        return camera;
    }

    static void WireBridge(GameObject root, Camera wrist, Camera scene)
    {
        var holder = GameObject.Find("RosBridge") ?? new GameObject("RosBridge");
        var bridge = holder.GetComponent<GrootArmBridge>() ?? holder.AddComponent<GrootArmBridge>();

        bridge.wristCamera = wrist;
        bridge.sceneCamera = scene;

        // Resolve joints BY NAME into the fixed order the bridge indexes by.
        // Order is the whole point: getting it wrong is not a crash, it is a
        // robot that moves plausibly and incorrectly.
        var bodies = root.GetComponentsInChildren<ArticulationBody>();
        var ordered = new List<ArticulationBody>();
        var missing = new List<string>();

        foreach (var jointName in JointNames)
        {
            // URDF-Importer names the GameObject after the CHILD LINK, so match
            // on the link stem rather than the joint name itself.
            var stem = jointName.Replace("_joint", "");
            var body = bodies.FirstOrDefault(b => b.name.Contains(stem))
                    ?? bodies.FirstOrDefault(b => b.name.Contains(jointName));
            if (body == null) missing.Add(jointName); else ordered.Add(body);
        }

        if (missing.Count > 0)
        {
            Debug.LogWarning($"[GR00T] Could not resolve: {string.Join(", ", missing)}. " +
                             "Assign those slots by hand in the Inspector.");
        }

        bridge.armJoints = ordered.ToArray();
        EditorUtility.SetDirty(bridge);

        Debug.Log($"[GR00T] Wired {ordered.Count}/{JointNames.Length} joints: " +
                  string.Join(", ", ordered.Select(b => b.name)));
    }
}
