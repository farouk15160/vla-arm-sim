#!/usr/bin/env bash
# Tear down every process belonging to the GR00T/VLA arm stack.
#
# Why this exists: ros2 launch does not always reap its children when the
# terminal dies, and a leftover move_group or gz sim is genuinely hard to spot.
# Two move_group instances on the same /move_action leave an action client
# talking to a server whose controllers are gone, which shows up as
# MoveItErrorCode=-4 (CONTROL_FAILED) on motions that look perfectly valid.
#
# Patterns match EXECUTABLE PATHS, not loose strings: a pattern like
# "ros2 launch groot_arm" also matches the command line of whatever shell is
# about to start that launch, so a naive version kills its own caller.
set -u

PATTERNS=(
  "bin/ros2 launch groot_arm"
  "moveit_ros_move_group/move_group"
  "moveit_servo/servo_node"
  "gz_tools_vendor/bin/gz sim"
  "^gz sim "
  "ros_gz_bridge/parameter_bridge"
  "ros_gz_image/image_bridge"
  "robot_state_publisher/robot_state_publisher"
  "lib/groot_vla/"
  "groot_vla.mock_policy_server"
  "groot_vla.smolvla_server"
  "controller_manager/spawner"
)

# Never kill ourselves or anything we are descended from.
protected=" $$ $PPID "
pid=$PPID
while [ "$pid" -gt 1 ] 2>/dev/null; do
  pid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
  [ -z "$pid" ] && break
  protected="$protected $pid "
done

for _ in 1 2 3; do
  found=0
  for pattern in "${PATTERNS[@]}"; do
    for target in $(pgrep -f "$pattern" 2>/dev/null); do
      case "$protected" in *" $target "*) continue ;; esac
      kill -9 "$target" 2>/dev/null && found=1
    done
  done
  [ "$found" -eq 0 ] && break
  sleep 1
done

sleep 1
# Re-check by pid, filtering the protected ancestry: a shell whose command line
# merely CONTAINS these patterns (this script's own text, for instance) is not
# a stack process.
leftover=""
for target in $(pgrep -f "moveit_ros_move_group/move_group|gz_tools_vendor/bin/gz sim|moveit_servo/servo_node" 2>/dev/null); do
  case "$protected" in *" $target "*) continue ;; esac
  leftover="$leftover $target"
done
if [ -n "$leftover" ]; then
  echo "still running:"
  for target in $leftover; do
    ps -o pid=,cmd= -p "$target" 2>/dev/null | cut -c1-100
  done
  exit 1
fi
echo "stack is clean"
