// Publishes the two camera views and the joint state from Unity to ROS 2, and
// drives the arm from trajectories coming back.
//
// This exists so the SAME groot_vla policy stack can be fed by Unity instead of
// Gazebo without changing anything on the ROS side: identical topic names,
// identical message types, identical image size. The policy cannot tell which
// simulator produced the observation, which is the whole point - it lets you
// compare renderers, or train on Unity's HDRP output and evaluate in Gazebo.
//
// Attach to an empty GameObject in the scene and assign the references in the
// Inspector.

using System.Collections.Generic;
using RosMessageTypes.Sensor;
using RosMessageTypes.Trajectory;
using Unity.Robotics.ROSTCPConnector;
using UnityEngine;

public class GrootArmBridge : MonoBehaviour
{
    [Header("ROS topics (must match the Gazebo side exactly)")]
    public string wristCameraTopic = "/wrist_camera/image_raw";
    public string sceneCameraTopic = "/scene_camera/image_raw";
    public string jointStateTopic = "/joint_states";
    public string armCommandTopic = "/arm_controller/joint_trajectory";

    [Header("Scene references")]
    public Camera wristCamera;
    public Camera sceneCamera;
    // Assign in URDF order: shoulder_pan, shoulder_lift, elbow, wrist_1..3.
    public ArticulationBody[] armJoints;

    [Header("Capture")]
    // 320x240 matches the Gazebo cameras. The policy downsamples to 224x224, so
    // rendering larger is wasted work on both sides.
    public int imageWidth = 320;
    public int imageHeight = 240;
    public float publishRate = 30f;

    static readonly string[] JointNames =
    {
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    };

    ROSConnection ros;
    RenderTexture wristTarget, sceneTarget;
    Texture2D readback;
    float nextPublish;
    double[] commanded;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<ImageMsg>(wristCameraTopic);
        ros.RegisterPublisher<ImageMsg>(sceneCameraTopic);
        ros.RegisterPublisher<JointStateMsg>(jointStateTopic);
        ros.Subscribe<JointTrajectoryMsg>(armCommandTopic, OnTrajectory);

        wristTarget = new RenderTexture(imageWidth, imageHeight, 24);
        sceneTarget = new RenderTexture(imageWidth, imageHeight, 24);
        // RGB24 so the buffer maps straight onto sensor_msgs/Image rgb8 with no
        // per-pixel conversion.
        readback = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        commanded = new double[armJoints.Length];
    }

    void Update()
    {
        if (Time.time < nextPublish) return;
        nextPublish = Time.time + 1f / publishRate;

        PublishCamera(wristCamera, wristTarget, wristCameraTopic, "wrist_camera_link");
        PublishCamera(sceneCamera, sceneTarget, sceneCameraTopic, "scene_camera_link");
        PublishJointState();
    }

    void PublishCamera(Camera camera, RenderTexture target, string topic, string frame)
    {
        if (camera == null) return;

        camera.targetTexture = target;
        camera.Render();

        var previous = RenderTexture.active;
        RenderTexture.active = target;
        readback.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        readback.Apply();
        RenderTexture.active = previous;
        camera.targetTexture = null;

        // Unity's origin is bottom-left, ROS images are top-left, so the rows
        // are flipped. Getting this wrong gives a vertically mirrored image
        // that looks plausible and silently poisons training.
        var source = readback.GetRawTextureData<byte>();
        int stride = imageWidth * 3;
        var data = new byte[stride * imageHeight];
        for (int row = 0; row < imageHeight; row++)
        {
            int from = (imageHeight - 1 - row) * stride;
            for (int i = 0; i < stride; i++) data[row * stride + i] = source[from + i];
        }

        ros.Publish(topic, new ImageMsg
        {
            header = new RosMessageTypes.Std.HeaderMsg { frame_id = frame },
            height = (uint)imageHeight,
            width = (uint)imageWidth,
            encoding = "rgb8",
            is_bigendian = 0,
            step = (uint)stride,
            data = data,
        });
    }

    void PublishJointState()
    {
        var positions = new double[armJoints.Length];
        var velocities = new double[armJoints.Length];
        for (int i = 0; i < armJoints.Length; i++)
        {
            if (armJoints[i] == null) continue;
            positions[i] = armJoints[i].jointPosition[0];
            velocities[i] = armJoints[i].jointVelocity[0];
        }

        ros.Publish(jointStateTopic, new JointStateMsg
        {
            name = JointNames,
            position = positions,
            velocity = velocities,
            effort = new double[armJoints.Length],
        });
    }

    void OnTrajectory(JointTrajectoryMsg message)
    {
        if (message.points == null || message.points.Length == 0) return;

        // Take the LAST point of the chunk as the target. Unity's articulation
        // drives interpolate to it themselves, so replaying every intermediate
        // point would fight the drive rather than help it.
        var point = message.points[message.points.Length - 1];
        for (int i = 0; i < message.joint_names.Length && i < point.positions.Length; i++)
        {
            int index = System.Array.IndexOf(JointNames, message.joint_names[i]);
            if (index < 0 || index >= armJoints.Length || armJoints[index] == null) continue;

            commanded[index] = point.positions[i];
            var drive = armJoints[index].xDrive;
            drive.target = Mathf.Rad2Deg * (float)commanded[index];  // Unity drives are in degrees
            armJoints[index].xDrive = drive;
        }
    }
}
