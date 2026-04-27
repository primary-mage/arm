#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_FILE="${STATE_FILE:-/tmp/arm_imitation_policy_real.state}"
ROS2_DAEMON_PATTERN='[r]os2cli\.daemon\.daemonize.*--name ros2-daemon'

DRIVER_PID=""
CAMERA_PID=""
CONTROLLER_PID=""
POLICY_PID=""
VIEWER_PID=""
VIEWER_BACKEND=""
VIEWER_CONFIG_FILE=""
ROS2_DAEMON_PIDS_BEFORE=""
declare -a ROS2_DAEMON_PIDS_BEFORE_ARR=()

usage() {
    cat <<USAGE
Usage: $(basename "$0") [options]

Stop the real-arm imitation-policy stack in this order:
1. arm_imitation_policy_node
2. inverse_dynamics_node
3. camera viewer
4. realsense_v4l_camera_node
5. dm_motor_sdk_ros driver

Options:
  --state-file <file>  state file written by run_imitation_policy_real.sh
  -h, --help           show this help
USAGE
}

stop_process_group() {
    local pid="$1"
    local name="$2"

    [[ -n "${pid}" ]] || return 0
    if [[ ! "${pid}" =~ ^[0-9]+$ ]]; then
        echo "[WARN] Skip invalid pid for ${name}: ${pid}"
        return 0
    fi

    if ! kill -0 -- "-${pid}" 2>/dev/null; then
        echo "[INFO] ${name} already stopped (pgid=${pid})"
        return 0
    fi

    echo "[INFO] Stopping ${name} (pgid=${pid})"
    kill -TERM -- "-${pid}" 2>/dev/null || true

    for _ in $(seq 1 20); do
        if ! kill -0 -- "-${pid}" 2>/dev/null; then
            return 0
        fi
        sleep 0.1
    done

    if kill -0 -- "-${pid}" 2>/dev/null; then
        echo "[WARN] ${name} did not stop after SIGTERM; sending SIGKILL to process group ${pid}"
        kill -KILL -- "-${pid}" 2>/dev/null || true
    fi
}

stop_new_ros2_daemons() {
    local -a current_pids=()
    mapfile -t current_pids < <(pgrep -f "${ROS2_DAEMON_PATTERN}" || true)

    for pid in "${current_pids[@]}"; do
        local seen_before=0
        for old_pid in "${ROS2_DAEMON_PIDS_BEFORE_ARR[@]}"; do
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

while [[ $# -gt 0 ]]; do
    case "$1" in
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

if [[ ! -f "${STATE_FILE}" ]]; then
    echo "[ERROR] State file not found: ${STATE_FILE}" >&2
    exit 1
fi

while IFS='=' read -r key value; do
    case "${key}" in
        DRIVER_PID)
            DRIVER_PID="${value}"
            ;;
        CAMERA_PID)
            CAMERA_PID="${value}"
            ;;
        CONTROLLER_PID)
            CONTROLLER_PID="${value}"
            ;;
        POLICY_PID)
            POLICY_PID="${value}"
            ;;
        VIEWER_PID)
            VIEWER_PID="${value}"
            ;;
        VIEWER_BACKEND)
            VIEWER_BACKEND="${value}"
            ;;
        VIEWER_CONFIG_FILE)
            VIEWER_CONFIG_FILE="${value}"
            ;;
        ROS2_DAEMON_PIDS_BEFORE)
            ROS2_DAEMON_PIDS_BEFORE="${value}"
            ;;
    esac
done < "${STATE_FILE}"

if [[ -n "${ROS2_DAEMON_PIDS_BEFORE}" ]]; then
    read -r -a ROS2_DAEMON_PIDS_BEFORE_ARR <<< "${ROS2_DAEMON_PIDS_BEFORE}"
fi

stop_process_group "${POLICY_PID}" "arm_imitation_policy_node"
stop_process_group "${CONTROLLER_PID}" "inverse_dynamics_node"
stop_process_group "${VIEWER_PID}" "camera viewer"
stop_process_group "${CAMERA_PID}" "realsense_v4l_camera_node"
stop_process_group "${DRIVER_PID}" "dm_motor_sdk_ros driver"
stop_new_ros2_daemons

if [[ -n "${VIEWER_CONFIG_FILE}" ]]; then
    rm -f "${VIEWER_CONFIG_FILE}"
fi
rm -f "${STATE_FILE}"
echo "[INFO] Imitation-policy stack stop sequence finished."
