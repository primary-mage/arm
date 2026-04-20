#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_SETUP="/opt/ros/humble/setup.bash"
WS_SETUP="$SCRIPT_DIR/install/setup.bash"

DEFAULT_ARM_PARAMS_FILE="$SCRIPT_DIR/src/arm2_task/config/params.yaml"
DEFAULT_DRIVER_PARAMS_FILE="$SCRIPT_DIR/src/dm_motor_sdk_ros/config/dm_motor_robot_driver.yaml"
DEFAULT_TEACH_CONFIG_FILE="$SCRIPT_DIR/src/ftservo_hls3625_teach/config/servo_bus.yaml"
DEFAULT_CAMERA_PARAMS_FILE="$SCRIPT_DIR/src/realsense_v4l_ros/config/realsense_v4l.yaml"

ARM_PARAMS_FILE="$DEFAULT_ARM_PARAMS_FILE"
DRIVER_PARAMS_FILE="$DEFAULT_DRIVER_PARAMS_FILE"
TEACH_CONFIG_FILE="$DEFAULT_TEACH_CONFIG_FILE"
CAMERA_PARAMS_FILE="$DEFAULT_CAMERA_PARAMS_FILE"
DATASET_ROOT="dataset"
TASK_NAME="unnamed_task"
READY_TIMEOUT=10
AUTO_START=1

DRIVER_PID=""
INVERSE_DYNAMICS_PID=""
CAMERA_PID=""
TEACH_STATE_PUBLISHER_PID=""
TEACH_BRIDGE_PID=""
RECORDER_PID=""
CLEANUP_RUNNING=0
ROS2_DAEMON_PATTERN='[r]os2cli\.daemon\.daemonize.*--name ros2-daemon'
declare -a ROS2_DAEMON_PIDS_BEFORE=()

usage() {
    cat <<USAGE
Usage: $(basename "$0") [options]

One-click startup for real imitation-learning data collection:
1. Launch dm_motor_sdk_ros driver
2. Wait for /robot_driver/ready == true
3. Launch realsense_v4l_ros camera node
4. Launch arm2_task inverse_dynamics_node
5. Launch ftservo_hls3625_teach bus_state_publisher
6. Launch arm2_task teach_pendant_follow_node
7. Launch arm2_task imitation_dataset_recorder_node
7. Optionally publish /episode_event = start:<task>

Options:
  --task <name>            task name written into each episode directory
  --dataset-root <dir>     dataset root directory for recorder output
  --arm-params <file>      arm2_task parameter file
  --driver-params <file>   dm_motor_sdk_ros driver parameter file
  --teach-config <file>    ftservo_hls3625_teach config YAML
  --camera-params <file>   realsense_v4l_ros camera parameter file
  --ready-timeout <sec>    wait timeout for ready/joint_states (default: ${READY_TIMEOUT})
  --no-auto-start          do not publish start:<task> automatically
  -h, --help               show this help

After startup, end the current episode with one of:
  ros2 topic pub --once /episode_event std_msgs/msg/String "{data: 'success'}"
  ros2 topic pub --once /episode_event std_msgs/msg/String "{data: 'fail'}"
  ros2 topic pub --once /episode_event std_msgs/msg/String "{data: 'stop'}"

Start a new episode manually with:
  ros2 topic pub --once /episode_event std_msgs/msg/String "{data: 'start:<task_name>'}"
USAGE
}

stop_process_group() {
    local pid="$1"
    local name="$2"

    [[ -n "${pid}" ]] || return 0

    echo "[INFO] Stopping ${name} (pgid=${pid})"
    kill -TERM -- "-${pid}" 2>/dev/null || true

    for _ in $(seq 1 20); do
        if ! kill -0 -- "-${pid}" 2>/dev/null; then
            wait "${pid}" 2>/dev/null || true
            return 0
        fi
        sleep 0.1
    done

    if kill -0 -- "-${pid}" 2>/dev/null; then
        echo "[WARN] ${name} did not stop after SIGTERM; sending SIGKILL to process group ${pid}"
        kill -KILL -- "-${pid}" 2>/dev/null || true
    fi
    wait "${pid}" 2>/dev/null || true
}

launch_in_group() {
    local __resultvar="$1"
    shift

    setsid "$@" &
    local pid=$!
    printf -v "${__resultvar}" '%s' "${pid}"
}

record_existing_ros2_daemons() {
    mapfile -t ROS2_DAEMON_PIDS_BEFORE < <(pgrep -f "${ROS2_DAEMON_PATTERN}" || true)
}

stop_new_ros2_daemons() {
    local -a current_pids=()
    mapfile -t current_pids < <(pgrep -f "${ROS2_DAEMON_PATTERN}" || true)

    for pid in "${current_pids[@]}"; do
        local seen_before=0
        for old_pid in "${ROS2_DAEMON_PIDS_BEFORE[@]}"; do
            if [[ "${pid}" == "${old_pid}" ]]; then
                seen_before=1
                break
            fi
        done

        if (( ! seen_before )); then
            echo "[INFO] Stopping ros2-daemon (pid=${pid})"
            kill -TERM "${pid}" 2>/dev/null || true
            for _ in $(seq 1 20); do
                if ! kill -0 "${pid}" 2>/dev/null; then
                    break
                fi
                sleep 0.1
            done
            if kill -0 "${pid}" 2>/dev/null; then
                echo "[WARN] ros2-daemon did not stop after SIGTERM; sending SIGKILL to pid ${pid}"
                kill -KILL "${pid}" 2>/dev/null || true
            fi
        fi
    done
}

cleanup() {
    if (( CLEANUP_RUNNING )); then
        return
    fi
    CLEANUP_RUNNING=1

    local pids=(
        "${RECORDER_PID}"
        "${TEACH_BRIDGE_PID}"
        "${TEACH_STATE_PUBLISHER_PID}"
        "${CAMERA_PID}"
        "${INVERSE_DYNAMICS_PID}"
        "${DRIVER_PID}"
    )
    local names=(
        "imitation_dataset_recorder_node"
        "teach_pendant_follow_node"
        "ftservo_hls3625_teach bus_state_publisher"
        "realsense_v4l_camera_node"
        "inverse_dynamics_node"
        "dm_motor_sdk_ros driver"
    )

    for i in "${!pids[@]}"; do
        stop_process_group "${pids[$i]}" "${names[$i]}"
    done

    stop_new_ros2_daemons
}

source_setup() {
    local setup_file="$1"
    set +u
    source "${setup_file}"
    set -u
}

assert_binary_runnable() {
    local binary="$1"
    local label="$2"
    local -a missing_libs=()

    if [[ ! -e "${binary}" ]]; then
        echo "[ERROR] ${label} executable not found: ${binary}" >&2
        return 1
    fi

    mapfile -t missing_libs < <(ldd "${binary}" 2>&1 | awk '/=> not found/ { print $1 }' || true)
    if (( ${#missing_libs[@]} > 0 )); then
        echo "[ERROR] ${label} has unresolved shared library dependencies:" >&2
        echo "        ${binary}" >&2
        printf '        %s\n' "${missing_libs[@]}" >&2
        echo "[ERROR] Rebuild the affected package(s) against the current environment before launching the stack." >&2
        return 1
    fi
}

wait_for_ready_true() {
    local timeout_sec="$1"
    local deadline=$((SECONDS + timeout_sec))

    while (( SECONDS < deadline )); do
        if [[ -n "${DRIVER_PID}" ]] && ! kill -0 "${DRIVER_PID}" 2>/dev/null; then
            echo "[ERROR] dm_motor_sdk_ros driver exited before publishing /robot_driver/ready == true" >&2
            return 1
        fi

        if ros2 topic echo /robot_driver/ready std_msgs/msg/Bool \
            --once \
            --qos-reliability reliable \
            --qos-durability transient_local 2>/dev/null | grep -q "data: true"; then
            return 0
        fi

        sleep 0.2
    done

    return 1
}

wait_for_joint_states() {
    local timeout_sec="$1"
    local deadline=$((SECONDS + timeout_sec))

    while (( SECONDS < deadline )); do
        if [[ -n "${TEACH_STATE_PUBLISHER_PID}" ]] && ! kill -0 "${TEACH_STATE_PUBLISHER_PID}" 2>/dev/null; then
            echo "[ERROR] ftservo_hls3625_teach bus_state_publisher exited before publishing /joint_states" >&2
            return 1
        fi

        if ros2 topic echo /joint_states sensor_msgs/msg/JointState --once >/dev/null 2>&1; then
            return 0
        fi

        sleep 0.2
    done

    return 1
}

wait_for_color_image() {
    local timeout_sec="$1"
    local deadline=$((SECONDS + timeout_sec))

    while (( SECONDS < deadline )); do
        if [[ -n "${CAMERA_PID}" ]] && ! kill -0 "${CAMERA_PID}" 2>/dev/null; then
            echo "[ERROR] realsense_v4l_camera_node exited before publishing /camera/color/image_raw" >&2
            return 1
        fi

        if ros2 topic echo /camera/color/image_raw sensor_msgs/msg/Image \
            --once \
            --qos-reliability best_effort >/dev/null 2>&1; then
            return 0
        fi

        sleep 0.2
    done

    return 1
}

publish_episode_start() {
    local task_name="$1"
    ros2 topic pub --once /episode_event std_msgs/msg/String "{data: 'start:${task_name}'}" >/dev/null
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)
            TASK_NAME="$2"
            shift 2
            ;;
        --dataset-root)
            DATASET_ROOT="$2"
            shift 2
            ;;
        --arm-params)
            ARM_PARAMS_FILE="$2"
            shift 2
            ;;
        --driver-params)
            DRIVER_PARAMS_FILE="$2"
            shift 2
            ;;
        --teach-config)
            TEACH_CONFIG_FILE="$2"
            shift 2
            ;;
        --camera-params)
            CAMERA_PARAMS_FILE="$2"
            shift 2
            ;;
        --ready-timeout)
            READY_TIMEOUT="$2"
            shift 2
            ;;
        --no-auto-start)
            AUTO_START=0
            shift 1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ ! -f "${ROS_SETUP}" ]]; then
    echo "[ERROR] ROS setup not found: ${ROS_SETUP}" >&2
    exit 1
fi

if [[ ! -f "${WS_SETUP}" ]]; then
    echo "[ERROR] Workspace setup not found: ${WS_SETUP}" >&2
    echo "        Build the workspace first, e.g. colcon build --base-paths src --packages-select dm_motor_sdk_ros arm2_task ftservo_hls3625_teach robot_msgs" >&2
    exit 1
fi

if [[ ! -f "${ARM_PARAMS_FILE}" ]]; then
    echo "[ERROR] arm2_task params file not found: ${ARM_PARAMS_FILE}" >&2
    exit 1
fi

if [[ ! -f "${DRIVER_PARAMS_FILE}" ]]; then
    echo "[ERROR] driver params file not found: ${DRIVER_PARAMS_FILE}" >&2
    exit 1
fi

if [[ ! -f "${TEACH_CONFIG_FILE}" ]]; then
    echo "[ERROR] teach config file not found: ${TEACH_CONFIG_FILE}" >&2
    exit 1
fi

if [[ ! -f "${CAMERA_PARAMS_FILE}" ]]; then
    echo "[ERROR] camera params file not found: ${CAMERA_PARAMS_FILE}" >&2
    exit 1
fi

trap cleanup EXIT INT TERM

cd "${SCRIPT_DIR}"
source_setup "${ROS_SETUP}"
source_setup "${WS_SETUP}"
export ARM_WORKSPACE_DIR="${SCRIPT_DIR}"

record_existing_ros2_daemons

assert_binary_runnable "${SCRIPT_DIR}/install/arm2_task/lib/arm2_task/inverse_dynamics_node" "inverse_dynamics_node"
assert_binary_runnable "${SCRIPT_DIR}/install/arm2_task/lib/arm2_task/teach_pendant_follow_node" "teach_pendant_follow_node"
assert_binary_runnable "${SCRIPT_DIR}/install/arm2_task/lib/arm2_task/imitation_dataset_recorder_node" "imitation_dataset_recorder_node"
assert_binary_runnable "${SCRIPT_DIR}/install/ftservo_hls3625_teach/lib/ftservo_hls3625_teach/bus_state_publisher" "bus_state_publisher"
assert_binary_runnable "${SCRIPT_DIR}/install/realsense_v4l_ros/lib/realsense_v4l_ros/realsense_v4l_camera_node" "realsense_v4l_camera_node"

echo "[INFO] arm params: ${ARM_PARAMS_FILE}"
echo "[INFO] driver params: ${DRIVER_PARAMS_FILE}"
echo "[INFO] teach config: ${TEACH_CONFIG_FILE}"
echo "[INFO] camera params: ${CAMERA_PARAMS_FILE}"
echo "[INFO] dataset root: ${DATASET_ROOT}"
echo "[INFO] task name: ${TASK_NAME}"
echo "[INFO] auto start: ${AUTO_START}"

echo "[INFO] Launching dm_motor_sdk_ros driver..."
launch_in_group DRIVER_PID ros2 launch dm_motor_sdk_ros dm_motor_robot_driver.launch.py params_path:="${DRIVER_PARAMS_FILE}"

echo "[INFO] Waiting for /robot_driver/ready ..."
if ! wait_for_ready_true "${READY_TIMEOUT}"; then
    echo "[ERROR] Timed out waiting for /robot_driver/ready == true" >&2
    exit 1
fi

echo "[INFO] Launching realsense_v4l_camera_node..."
launch_in_group CAMERA_PID ros2 launch realsense_v4l_ros realsense_v4l_launch.py params_path:="${CAMERA_PARAMS_FILE}"

echo "[INFO] Waiting for /camera/color/image_raw ..."
if ! wait_for_color_image "${READY_TIMEOUT}"; then
    echo "[ERROR] Timed out waiting for /camera/color/image_raw" >&2
    exit 1
fi

echo "[INFO] Launching inverse_dynamics_node..."
launch_in_group INVERSE_DYNAMICS_PID ros2 launch arm2_task inverse_dynamics_launch.py params_path:="${ARM_PARAMS_FILE}"

echo "[INFO] Launching ftservo_hls3625_teach state publisher..."
launch_in_group TEACH_STATE_PUBLISHER_PID ros2 launch ftservo_hls3625_teach state_publisher.launch.py config:="${TEACH_CONFIG_FILE}"

echo "[INFO] Waiting for /joint_states ..."
if ! wait_for_joint_states "${READY_TIMEOUT}"; then
    echo "[ERROR] Timed out waiting for /joint_states" >&2
    exit 1
fi

echo "[INFO] Launching teach_pendant_follow_node..."
launch_in_group TEACH_BRIDGE_PID ros2 launch arm2_task teach_pendant_follow_launch.py params_path:="${ARM_PARAMS_FILE}"

RECORDER_CMD=(
    ros2 run arm2_task imitation_dataset_recorder_node
    --ros-args
    --params-file "${ARM_PARAMS_FILE}"
    -p "imitation_dataset.dataset_root:=${DATASET_ROOT}"
    -p "imitation_dataset.task_name:=${TASK_NAME}"
    -p "imitation_dataset.auto_start:=false"
)

echo "[INFO] Launching imitation_dataset_recorder_node..."
launch_in_group RECORDER_PID "${RECORDER_CMD[@]}"

sleep 1

if (( AUTO_START )); then
    echo "[INFO] Publishing /episode_event start:${TASK_NAME}"
    publish_episode_start "${TASK_NAME}"
fi

echo "[INFO] Imitation recording stack is ready."
echo "[INFO] End current episode with:"
echo "       ros2 topic pub --once /episode_event std_msgs/msg/String \"{data: 'success'}\""
echo "       ros2 topic pub --once /episode_event std_msgs/msg/String \"{data: 'fail'}\""
echo "       ros2 topic pub --once /episode_event std_msgs/msg/String \"{data: 'stop'}\""
echo "[INFO] Start next episode with:"
echo "       ros2 topic pub --once /episode_event std_msgs/msg/String \"{data: 'start:<task_name>'}\""

wait "${RECORDER_PID}"
