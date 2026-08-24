// Robotics -> Joint Jog : drag sliders, watch the arm move.
//
// This exists to separate two questions that otherwise get debugged together:
//
//   1. Does the robot itself work in Unity - drives configured, joints not
//      collapsing, axes sane?
//   2. Does the ROS bridge work - connection, topics, message conversion?
//
// If the arm moves here, Unity is fine and any remaining problem is the
// bridge. If it does not move here, no amount of ROS debugging will help.
//
// Works in Play mode (drives act on the physics solver) and is read-only
// otherwise.

using System.Linq;
using UnityEditor;
using UnityEngine;

public class JointJogWindow : EditorWindow
{
    static readonly string[] JointStems =
    {
        "shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3",
    };

    // Home posture from the SRDF, in degrees. Same pose Gazebo starts in, so
    // the two simulators can be compared from an identical configuration.
    static readonly float[] HomeDegrees = { 0f, -90f, 90f, -90f, -90f, 0f };

    ArticulationBody[] joints;
    float[] targets;
    float gripper;

    [MenuItem("Robotics/Joint Jog")]
    public static void Open() => GetWindow<JointJogWindow>("Joint Jog");

    void OnGUI()
    {
        if (GUILayout.Button("Find robot") || joints == null) Refresh();

        if (joints == null || joints.Length == 0)
        {
            EditorGUILayout.HelpBox(
                "No ArticulationBody joints found.\n\n" +
                "Import the URDF, then run Robotics -> Build GR00T Arm Scene.",
                MessageType.Warning);
            return;
        }

        if (!Application.isPlaying)
        {
            EditorGUILayout.HelpBox(
                "Press Play. Drives act through the physics solver, so nothing " +
                "moves while the editor is stopped.", MessageType.Info);
        }

        EditorGUILayout.Space();
        for (int i = 0; i < joints.Length; i++)
        {
            if (joints[i] == null) continue;
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField(JointStems[i], GUILayout.Width(100));
            // Unity articulation drives are in DEGREES; ROS joints are radians.
            targets[i] = EditorGUILayout.Slider(targets[i], -180f, 180f);
            EditorGUILayout.EndHorizontal();
        }

        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Gripper (0 closed .. 0.04 open)");
        gripper = EditorGUILayout.Slider(gripper, 0f, 0.04f);

        EditorGUILayout.Space();
        EditorGUILayout.BeginHorizontal();
        if (GUILayout.Button("Apply")) Apply();
        if (GUILayout.Button("Home"))
        {
            System.Array.Copy(HomeDegrees, targets, Mathf.Min(HomeDegrees.Length, targets.Length));
            gripper = 0.04f;
            Apply();
        }
        if (GUILayout.Button("Zero"))
        {
            for (int i = 0; i < targets.Length; i++) targets[i] = 0f;
            Apply();
        }
        EditorGUILayout.EndHorizontal();

        // Continuous apply while dragging, so the sliders feel live.
        if (GUI.changed && Application.isPlaying) Apply();
    }

    void Refresh()
    {
        var bodies = Object.FindObjectsByType<ArticulationBody>(FindObjectsInactive.Include);
        joints = JointStems
            .Select(stem => bodies.FirstOrDefault(b => b.name.Contains(stem)))
            .ToArray();
        targets = new float[joints.Length];

        for (int i = 0; i < joints.Length; i++)
            if (joints[i] != null) targets[i] = joints[i].xDrive.target;

        int found = joints.Count(j => j != null);
        Debug.Log($"[GR00T] Joint Jog found {found}/{JointStems.Length} joints.");
    }

    void Apply()
    {
        for (int i = 0; i < joints.Length; i++)
        {
            if (joints[i] == null) continue;
            var drive = joints[i].xDrive;
            drive.target = targets[i];
            joints[i].xDrive = drive;
        }

        // Fingers move together and are prismatic, so their target is in metres.
        foreach (var body in Object.FindObjectsByType<ArticulationBody>(FindObjectsInactive.Include))
        {
            if (!body.name.Contains("finger")) continue;
            var drive = body.xDrive;
            drive.target = gripper;
            body.xDrive = drive;
        }
    }
}
