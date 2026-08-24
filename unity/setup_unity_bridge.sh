#!/usr/bin/env bash
# Set up the ROS 2 half of the Unity bridge.
#
# The Unity half needs the Unity Editor, which is a ~10 GB install tied to a
# Unity account, so it cannot be scripted here - see unity/README.md.
#
# This clones ros_tcp_endpoint and applies one fix: upstream calls
# rclpy.shutdown() twice, which older ROS distros tolerated but Jazzy rejects
# with "rcl_shutdown already called on the given context".
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$WS/src/ros_tcp_endpoint"

if [ ! -d "$SRC" ]; then
  echo "cloning ros_tcp_endpoint ..."
  git clone -q --branch main-ros2 --depth 1 \
    https://github.com/Unity-Technologies/ROS-TCP-Endpoint.git "$SRC"
else
  echo "ros_tcp_endpoint already present"
fi

ENTRY="$SRC/ros_tcp_endpoint/default_server_endpoint.py"
if ! grep -q "if rclpy.ok():" "$ENTRY"; then
  echo "patching double rclpy.shutdown() for Jazzy ..."
  python3 - "$ENTRY" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
s = s.replace("    tcp_server.destroy_nodes()\n    rclpy.shutdown()",
              "    tcp_server.destroy_nodes()\n"
              "    # setup_executor() already shuts the context down; calling it again\n"
              "    # raises on Jazzy. Older distros tolerated the double call.\n"
              "    if rclpy.ok():\n        rclpy.shutdown()")
p.write_text(s)
PY
else
  echo "patch already applied"
fi

echo
echo "now build it:"
echo "  cd $WS && colcon build --packages-select ros_tcp_endpoint && source install/setup.bash"
echo "then:"
echo "  ros2 launch groot_arm_bringup unity_bridge.launch.py"
