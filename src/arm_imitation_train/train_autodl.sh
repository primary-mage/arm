#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DEFAULT_CONFIG="${SCRIPT_DIR}/config/default_config.yaml"

CONFIG_PATH="${DEFAULT_CONFIG}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_DEPS=0

usage() {
    cat <<USAGE
Usage: $(basename "$0") [options]

AutoDL helper for ARM imitation learning training.

Options:
  --config <file>       training config path
  --install-deps        install Python dependencies from requirements.txt first
  --python <bin>        python executable to use (default: ${PYTHON_BIN})
  -h, --help            show this help

Example:
  bash src/arm_imitation_train/train_autodl.sh --install-deps
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --install-deps)
            INSTALL_DEPS=1
            shift 1
            ;;
        --python)
            PYTHON_BIN="$2"
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

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "[ERROR] Config file not found: ${CONFIG_PATH}" >&2
    exit 1
fi

if (( INSTALL_DEPS )); then
    echo "[INFO] Installing Python dependencies..."
    "${PYTHON_BIN}" -m pip install -r "${SCRIPT_DIR}/requirements.txt"
fi

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

echo "[INFO] Repo root: ${REPO_ROOT}"
echo "[INFO] Config: ${CONFIG_PATH}"
echo "[INFO] Python: ${PYTHON_BIN}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m arm_imitation_train.train --config "${CONFIG_PATH}"
