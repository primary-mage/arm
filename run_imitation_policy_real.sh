#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_SETUP="/opt/ros/humble/setup.bash"
WS_SETUP="$SCRIPT_DIR/install/setup.bash"
STATE_FILE="${STATE_FILE:-/tmp/arm_imitation_policy_real.state}"

DEFAULT_ARM_PARAMS_FILE="$SCRIPT_DIR/src/arm2_task/config/params.yaml"
DEFAULT_DRIVER_PARAMS_FILE="$SCRIPT_DIR/src/dm_motor_sdk_ros/config/dm_motor_robot_driver.yaml"
DEFAULT_CAMERA_PARAMS_FILE="$SCRIPT_DIR/src/realsense_v4l_ros/config/realsense_v4l.yaml"
DEFAULT_POLICY_PARAMS_FILE="$SCRIPT_DIR/src/arm_imitation_train/config/inference_config.yaml"
DEFAULT_POLICY_EXECUTABLE="arm_imitation_policy"
DEFAULT_VIEWER_TOPIC="/camera/color/image_raw"
DEFAULT_VIEWER_FIXED_FRAME="camera_color_optical_frame"

ARM_PARAMS_FILE="$DEFAULT_ARM_PARAMS_FILE"
DRIVER_PARAMS_FILE="$DEFAULT_DRIVER_PARAMS_FILE"
CAMERA_PARAMS_FILE="$DEFAULT_CAMERA_PARAMS_FILE"
POLICY_PARAMS_FILE="$DEFAULT_POLICY_PARAMS_FILE"
POLICY_EXECUTABLE="$DEFAULT_POLICY_EXECUTABLE"
READY_TIMEOUT=10
IMAGE_TIMEOUT=10
SERVICE_TIMEOUT=10
POLICY_TIMEOUT=10
MODE_NAME="tracking"
MODE_SERVICE_NAME="set_controller_mode"
VIEWER_ENABLED=1
VIEWER_TOPIC="$DEFAULT_VIEWER_TOPIC"
VIEWER_STARTUP_GRACE_SEC=1
VIEWER_FIXED_FRAME="$DEFAULT_VIEWER_FIXED_FRAME"
VIEWER_BACKEND=""
VIEWER_CONFIG_FILE=""

DRIVER_PID=""
CAMERA_PID=""
CONTROLLER_PID=""
POLICY_PID=""
VIEWER_PID=""
STATE_TMP_FILE=""
CLEANUP_RUNNING=0
ROS2_DAEMON_PATTERN='[r]os2cli\.daemon\.daemonize.*--name ros2-daemon'
declare -a ROS2_DAEMON_PIDS_BEFORE=()

usage() {
    cat <<USAGE
Usage: $(basename "$0") [options]

One-click startup for real-arm imitation-policy testing:
1. Launch dm_motor_sdk_ros driver
2. Wait for /robot_driver/ready == true
3. Launch realsense_v4l_ros camera node
4. Wait for /camera/color/image_raw
5. Launch arm2_task inverse_dynamics_node
6. Wait for the controller mode service, then switch to the configured mode
7. Launch arm_imitation_train policy node
8. Wait for /joint_target_state
9. Open a live camera viewer window

Options:
  --arm-params <file>      arm2_task parameter file
  --driver-params <file>   dm_motor_sdk_ros driver parameter file
  --camera-params <file>   realsense_v4l_ros camera parameter file
  --policy-params <file>   arm_imitation_train inference parameter file
  --policy-executable <name>
                           arm_imitation_train executable name (default: ${POLICY_EXECUTABLE})
  --mode <name>            controller mode before policy starts (default: ${MODE_NAME})
  --mode-service <name>    controller mode service name (default: ${MODE_SERVICE_NAME})
  --ready-timeout <sec>    wait timeout for /robot_driver/ready (default: ${READY_TIMEOUT})
  --image-timeout <sec>    wait timeout for /camera/color/image_raw (default: ${IMAGE_TIMEOUT})
  --service-timeout <sec>  wait timeout for mode service (default: ${SERVICE_TIMEOUT})
  --policy-timeout <sec>   wait timeout for /joint_target_state (default: ${POLICY_TIMEOUT})
  --viewer-topic <topic>   camera topic shown by image_tools showimage (default: ${VIEWER_TOPIC})
  --no-viewer              do not open the camera viewer window after startup
  --state-file <file>      where to store process-group state (default: ${STATE_FILE})
  -h, --help               show this help

Stop the stack later with:
  ${SCRIPT_DIR}/stop_imitation_policy_real.sh --state-file <same path>
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

    stop_process_group "${POLICY_PID}" "arm_imitation_policy_node"
    stop_process_group "${CONTROLLER_PID}" "inverse_dynamics_node"
    stop_process_group "${VIEWER_PID}" "camera viewer"
    stop_process_group "${CAMERA_PID}" "realsense_v4l_camera_node"
    stop_process_group "${DRIVER_PID}" "dm_motor_sdk_ros driver"
    stop_new_ros2_daemons

    if [[ -n "${STATE_TMP_FILE}" ]]; then
        rm -f "${STATE_TMP_FILE}"
    fi
    if [[ -n "${VIEWER_CONFIG_FILE}" ]]; then
        rm -f "${VIEWER_CONFIG_FILE}"
    fi
    rm -f "${STATE_FILE}"
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
        return 1
    fi
}

normalize_ros_name() {
    local name="$1"
    if [[ "${name}" == /* ]]; then
        printf '%s' "${name}"
    else
        printf '/%s' "${name}"
    fi
}

state_file_has_live_pid() {
    local file="$1"
    local pid=""

    [[ -f "${file}" ]] || return 1

    while IFS='=' read -r key value; do
        case "${key}" in
            DRIVER_PID|CAMERA_PID|CONTROLLER_PID|POLICY_PID|VIEWER_PID)
                pid="${value}"
                if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
                    return 0
                fi
                ;;
        esac
    done < "${file}"

    return 1
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

wait_for_service_available() {
    local service_name="$1"
    local timeout_sec="$2"
    local deadline=$((SECONDS + timeout_sec))

    while (( SECONDS < deadline )); do
        if [[ -n "${CONTROLLER_PID}" ]] && ! kill -0 "${CONTROLLER_PID}" 2>/dev/null; then
            echo "[ERROR] inverse_dynamics_node exited before service ${service_name} became available" >&2
            return 1
        fi

        if ros2 service list 2>/dev/null | grep -qx "${service_name}"; then
            return 0
        fi

        sleep 0.2
    done

    return 1
}

call_mode_service() {
    local service_name="$1"
    local mode_name="$2"
    local output=""

    if ! output=$(ros2 service call "${service_name}" robot_msgs/srv/SetControllerMode \
        "{mode: ${mode_name}}" 2>&1); then
        echo "[ERROR] Failed to call ${service_name}" >&2
        echo "${output}" >&2
        return 1
    fi

    echo "${output}"
    if ! grep -Eqi "success:[[:space:]]*true|success=True" <<< "${output}"; then
        echo "[ERROR] Mode service responded without success=true" >&2
        return 1
    fi

    return 0
}

wait_for_policy_target() {
    local timeout_sec="$1"
    local deadline=$((SECONDS + timeout_sec))

    while (( SECONDS < deadline )); do
        if [[ -n "${POLICY_PID}" ]] && ! kill -0 "${POLICY_PID}" 2>/dev/null; then
            echo "[ERROR] arm_imitation_policy_node exited before publishing /joint_target_state" >&2
            return 1
        fi

        if ros2 topic echo /joint_target_state robot_msgs/msg/RobotState --once >/dev/null 2>&1; then
            return 0
        fi

        sleep 0.2
    done

    return 1
}

write_state_file() {
    local daemon_pids_before="${ROS2_DAEMON_PIDS_BEFORE[*]}"

    STATE_TMP_FILE="${STATE_FILE}.tmp.$$"
    {
        printf 'SCRIPT_DIR=%s\n' "${SCRIPT_DIR}"
        printf 'DRIVER_PID=%s\n' "${DRIVER_PID}"
        printf 'CAMERA_PID=%s\n' "${CAMERA_PID}"
        printf 'CONTROLLER_PID=%s\n' "${CONTROLLER_PID}"
        printf 'POLICY_PID=%s\n' "${POLICY_PID}"
        printf 'VIEWER_PID=%s\n' "${VIEWER_PID}"
        printf 'VIEWER_BACKEND=%s\n' "${VIEWER_BACKEND}"
        printf 'VIEWER_CONFIG_FILE=%s\n' "${VIEWER_CONFIG_FILE}"
        printf 'ROS2_DAEMON_PIDS_BEFORE=%s\n' "${daemon_pids_before}"
    } > "${STATE_TMP_FILE}"
    mv "${STATE_TMP_FILE}" "${STATE_FILE}"
    STATE_TMP_FILE=""
}

resolve_viewer_backend() {
    if (( ! VIEWER_ENABLED )); then
        VIEWER_BACKEND=""
        return 0
    fi

    if command -v rviz2 >/dev/null 2>&1; then
        VIEWER_BACKEND="rviz2"
        return 0
    fi

    if [[ -x /opt/ros/humble/lib/image_tools/showimage ]]; then
        VIEWER_BACKEND="showimage"
        return 0
    fi

    echo "[WARN] No supported camera viewer found. Continuing without a viewer." >&2
    VIEWER_ENABLED=0
    VIEWER_BACKEND=""
    return 0
}

prepare_rviz_viewer_config() {
    VIEWER_CONFIG_FILE="/tmp/arm_camera_viewer_${$}.rviz"
    cat > "${VIEWER_CONFIG_FILE}" <<EOF
Panels:
  - Class: rviz_common/Displays
    Name: Displays
Visualization Manager:
  Class: ""
  Displays:
    - Class: rviz_default_plugins/Image
      Enabled: true
      Image Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Best Effort
        Value: ${VIEWER_TOPIC}
      Max Value: 1
      Median window: 5
      Min Value: 0
      Name: Camera
      Normalize Range: true
      Queue Size: 2
      Transport Hint: raw
      Value: true
  Enabled: true
  Global Options:
    Background Color: 48; 48; 48
    Fixed Frame: ${VIEWER_FIXED_FRAME}
  Name: root
  Tools:
    - Class: rviz_default_plugins/Interact
    - Class: rviz_default_plugins/MoveCamera
    - Class: rviz_default_plugins/Select
    - Class: rviz_default_plugins/FocusCamera
  Value: true
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 2
      Focal Point:
        X: 0
        Y: 0
        Z: 0
      Name: Current View
      Near Clip Distance: 0.01
      Pitch: 0.785398
      Target Frame: <Fixed Frame>
      Value: Orbit (rviz)
      Yaw: 0.785398
Window Geometry:
  Height: 900
  Width: 1400
  X: 60
  Y: 60
EOF
}

launch_camera_viewer() {
    local topic_name="$1"

    if (( ! VIEWER_ENABLED )); then
        return 0
    fi

    echo "[INFO] Launching camera viewer for ${topic_name}..."
    if [[ "${VIEWER_BACKEND}" == "rviz2" ]]; then
        prepare_rviz_viewer_config
        launch_in_group VIEWER_PID rviz2 -d "${VIEWER_CONFIG_FILE}"
    else
        launch_in_group VIEWER_PID /opt/ros/humble/lib/image_tools/showimage --ros-args -r __node:=arm_camera_viewer -r "image:=${topic_name}"
    fi

    sleep "${VIEWER_STARTUP_GRACE_SEC}"
    if ! kill -0 "${VIEWER_PID}" 2>/dev/null; then
        echo "[WARN] Camera viewer exited immediately. The control stack is still running." >&2
        VIEWER_PID=""
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arm-params)
            ARM_PARAMS_FILE="$2"
            shift 2
            ;;
        --driver-params)
            DRIVER_PARAMS_FILE="$2"
            shift 2
            ;;
        --camera-params)
            CAMERA_PARAMS_FILE="$2"
            shift 2
            ;;
        --policy-params)
            POLICY_PARAMS_FILE="$2"
            shift 2
            ;;
        --policy-executable)
            POLICY_EXECUTABLE="$2"
            shift 2
            ;;
        --mode)
            MODE_NAME="$2"
            shift 2
            ;;
        --mode-service)
            MODE_SERVICE_NAME="$2"
            shift 2
            ;;
        --ready-timeout)
            READY_TIMEOUT="$2"
            shift 2
            ;;
        --image-timeout)
            IMAGE_TIMEOUT="$2"
            shift 2
            ;;
        --service-timeout)
            SERVICE_TIMEOUT="$2"
            shift 2
            ;;
        --policy-timeout)
            POLICY_TIMEOUT="$2"
            shift 2
            ;;
        --viewer-topic)
            VIEWER_TOPIC="$2"
            shift 2
            ;;
        --no-viewer)
            VIEWER_ENABLED=0
            shift
            ;;
        --state-file)
            STATE_FILE="$2"
            shift 2
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

MODE_SERVICE_NAME="$(normalize_ros_name "${MODE_SERVICE_NAME}")"
VIEWER_TOPIC="$(normalize_ros_name "${VIEWER_TOPIC}")"

if [[ -f "${STATE_FILE}" ]]; then
    if state_file_has_live_pid "${STATE_FILE}"; then
        echo "[ERROR] Existing live stack detected via state file: ${STATE_FILE}" >&2
        echo "        Stop it first with ${SCRIPT_DIR}/stop_imitation_policy_real.sh --state-file ${STATE_FILE}" >&2
        exit 1
    fi
    echo "[WARN] Removing stale state file: ${STATE_FILE}"
    rm -f "${STATE_FILE}"
fi

if [[ ! -f "${ROS_SETUP}" ]]; then
    echo "[ERROR] ROS setup not found: ${ROS_SETUP}" >&2
    exit 1
fi

if [[ ! -f "${WS_SETUP}" ]]; then
    echo "[ERROR] Workspace setup not found: ${WS_SETUP}" >&2
    echo "        Build the workspace first, e.g. colcon build --packages-select dm_motor_sdk_ros realsense_v4l_ros arm2_task arm_imitation_train" >&2
    exit 1
fi

for file_path in \
    "${ARM_PARAMS_FILE}" \
    "${DRIVER_PARAMS_FILE}" \
    "${CAMERA_PARAMS_FILE}" \
    "${POLICY_PARAMS_FILE}"; do
    if [[ ! -f "${file_path}" ]]; then
        echo "[ERROR] Required config file not found: ${file_path}" >&2
        exit 1
    fi
done

trap cleanup EXIT INT TERM

cd "${SCRIPT_DIR}"
source_setup "${ROS_SETUP}"
source_setup "${WS_SETUP}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/ros_logs}"
record_existing_ros2_daemons
resolve_viewer_backend

assert_binary_runnable "${SCRIPT_DIR}/install/dm_motor_sdk_ros/lib/dm_motor_sdk_ros/dm_motor_robot_driver_node" "dm_motor_sdk_ros driver"
assert_binary_runnable "${SCRIPT_DIR}/install/realsense_v4l_ros/lib/realsense_v4l_ros/realsense_v4l_camera_node" "realsense_v4l_ros camera"
assert_binary_runnable "${SCRIPT_DIR}/install/arm2_task/lib/arm2_task/inverse_dynamics_node" "arm2_task inverse_dynamics_node"
assert_binary_runnable "${SCRIPT_DIR}/install/arm_imitation_train/lib/arm_imitation_train/${POLICY_EXECUTABLE}" "arm_imitation_train policy"
if (( VIEWER_ENABLED )) && [[ "${VIEWER_BACKEND}" == "showimage" ]]; then
    assert_binary_runnable "/opt/ros/humble/lib/image_tools/showimage" "image_tools showimage"
fi

echo "[INFO] arm params: ${ARM_PARAMS_FILE}"
echo "[INFO] driver params: ${DRIVER_PARAMS_FILE}"
echo "[INFO] camera params: ${CAMERA_PARAMS_FILE}"
echo "[INFO] policy params: ${POLICY_PARAMS_FILE}"
echo "[INFO] policy executable: ${POLICY_EXECUTABLE}"
echo "[INFO] controller mode: ${MODE_NAME}"
if (( VIEWER_ENABLED )); then
    echo "[INFO] viewer topic: ${VIEWER_TOPIC}"
    echo "[INFO] viewer backend: ${VIEWER_BACKEND}"
else
    echo "[INFO] camera viewer: disabled"
fi

echo "[INFO] Launching dm_motor_sdk_ros driver..."
launch_in_group DRIVER_PID ros2 launch dm_motor_sdk_ros dm_motor_robot_driver.launch.py params_path:="${DRIVER_PARAMS_FILE}"

echo "[INFO] Waiting for /robot_driver/ready == true (timeout: ${READY_TIMEOUT}s)..."
if ! wait_for_ready_true "${READY_TIMEOUT}"; then
    exit 1
fi

echo "[INFO] Launching realsense_v4l_ros camera..."
launch_in_group CAMERA_PID ros2 launch realsense_v4l_ros realsense_v4l_launch.py params_path:="${CAMERA_PARAMS_FILE}"

echo "[INFO] Waiting for /camera/color/image_raw (timeout: ${IMAGE_TIMEOUT}s)..."
if ! wait_for_color_image "${IMAGE_TIMEOUT}"; then
    exit 1
fi

echo "[INFO] Launching inverse_dynamics_node..."
launch_in_group CONTROLLER_PID ros2 launch arm2_task inverse_dynamics_launch.py params_path:="${ARM_PARAMS_FILE}"

echo "[INFO] Waiting for controller mode service ${MODE_SERVICE_NAME} (timeout: ${SERVICE_TIMEOUT}s)..."
if ! wait_for_service_available "${MODE_SERVICE_NAME}" "${SERVICE_TIMEOUT}"; then
    exit 1
fi

echo "[INFO] Switching controller mode to ${MODE_NAME}..."
call_mode_service "${MODE_SERVICE_NAME}" "${MODE_NAME}"

echo "[INFO] Launching imitation policy node..."
launch_in_group POLICY_PID ros2 launch arm_imitation_train imitation_policy_launch.py params_path:="${POLICY_PARAMS_FILE}" policy_executable:="${POLICY_EXECUTABLE}"

echo "[INFO] Waiting for /joint_target_state (timeout: ${POLICY_TIMEOUT}s)..."
if ! wait_for_policy_target "${POLICY_TIMEOUT}"; then
    exit 1
fi

launch_camera_viewer "${VIEWER_TOPIC}"

write_state_file
trap - EXIT INT TERM

echo "[INFO] Imitation-policy stack is running."
echo "       State file: ${STATE_FILE}"
echo "       Stop with: ${SCRIPT_DIR}/stop_imitation_policy_real.sh --state-file ${STATE_FILE}"
exit 0
