#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MBPO_TASK_NAME="${MBPO_TASK_NAME:-libero_10}"
MBPO_CONFIG="${MBPO_CONFIG:-libero_10_mbpo_openpi_pi05_config}"

MBPO_TASK_NAME="${MBPO_TASK_NAME}" \
MBPO_CONFIG="${MBPO_CONFIG}" \
  bash "${REPO_PATH}/examples/embodiment/run_wm_rl_vla.sh"
