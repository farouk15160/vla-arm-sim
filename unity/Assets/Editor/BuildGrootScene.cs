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
