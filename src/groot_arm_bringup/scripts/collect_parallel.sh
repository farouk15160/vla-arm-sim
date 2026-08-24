#!/usr/bin/env bash
# Run several independent simulations at once to collect demonstrations faster.
#
# Each worker gets its own ROS_DOMAIN_ID, which puts it on a separate DDS
# network: separate /clock, /joint_states, move_group and Gazebo instance, with
# no cross-talk. Each also gets its own Gazebo partition so the gz transport
# layer stays separate too, and its own output directory and RNG seed.
#
#   ./collect_parallel.sh --workers 3 --episodes 40 --output ~/ws/data/demos/run1
#
# Sizing: each worker is a full Gazebo + MoveIt stack. Measured here at roughly
# 1.6 GB RAM and 0.25 GB VRAM per worker, and it is CPU-bound - physics runs
# unthrottled, so workers compete for cores. Two or three is usually the sweet
# spot on a 6-core laptop; beyond that each worker slows down and total
# throughput stops improving. Check with `nproc` and watch the episode rate.
set -euo pipefail

WORKERS=2
EPISODES=40
OUTPUT=""
DISTRACTORS=2
BASE_DOMAIN=41

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers)     WORKERS="$2"; shift 2 ;;
    --episodes)    EPISODES="$2"; shift 2 ;;   # per worker
    --output)      OUTPUT="$2"; shift 2 ;;
    --distractors) DISTRACTORS="$2"; shift 2 ;;
    --base-domain) BASE_DOMAIN="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$OUTPUT" ]; then
  echo "--output is required (e.g. --output ~/ws/data/demos/run1)" >&2
  exit 2
fi
OUTPUT="${OUTPUT/#\~/$HOME}"

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
echo "workspace : $WS"
echo "workers   : $WORKERS x $EPISODES episodes -> $((WORKERS * EPISODES)) total"
echo "output    : $OUTPUT/worker_N"
echo

PIDS=()
cleanup() {
  echo; echo "stopping workers ..."
  for pid in "${PIDS[@]:-}"; do kill -- "-$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for ((i = 0; i < WORKERS; i++)); do
  domain=$((BASE_DOMAIN + i))
  dir="$OUTPUT/worker_$i"
  log="$OUTPUT/worker_$i.log"
  mkdir -p "$dir"

  # setsid so each worker is its own process group and can be killed cleanly.
  setsid bash -c "
    source /opt/ros/jazzy/setup.bash
    source '$WS/install/setup.bash'
    export ROS_DOMAIN_ID=$domain
    # Keep the gz transport layer separate too; ROS_DOMAIN_ID alone does not
    # isolate Gazebo's own discovery, and two sims sharing a partition fight
    # over world and service names.
    export GZ_PARTITION=collect_$domain
    export PYTHONUNBUFFERED=1

    ros2 launch groot_arm_bringup system.launch.py \
        policy:=none gazebo_gui:=false rviz:=false gui:=false goal_marker:=false &
    LAUNCH=\$!

    # Wait for the stack instead of sleeping a guessed amount.
    for _ in \$(seq 1 90); do
      if ros2 topic list 2>/dev/null | grep -q /world_markers; then break; fi
      sleep 2
    done

    ros2 run groot_vla episode_recorder --ros-args \
        -p output_dir:='$dir' -p fps:=10.0 -p use_sim_time:=true &
    sleep 8

    ros2 run groot_vla collect_demos --ros-args \
        -p episodes:=$EPISODES -p distractors:=$DISTRACTORS \
        -p seed:=\$((1000 + $i)) -p use_sim_time:=true

    kill \$LAUNCH 2>/dev/null || true
  " > "$log" 2>&1 &

  PIDS+=($!)
  echo "  worker $i: ROS_DOMAIN_ID=$domain -> $dir  (log: $log)"
  # Stagger starts so N Gazebo instances do not all compile shaders at once.
  sleep 12
done

echo
echo "collecting; watch with:  tail -f $OUTPUT/worker_0.log"
wait
echo
echo "done. Episodes per worker:"
for ((i = 0; i < WORKERS; i++)); do
  echo "  worker_$i: $(ls "$OUTPUT/worker_$i" 2>/dev/null | wc -l)"
done
echo
echo "Merge them into one dataset before exporting:"
echo "  python3 $WS/src/groot_vla/groot_vla/merge_demos.py \\"
echo "      --inputs $OUTPUT/worker_* --output $OUTPUT/merged"
