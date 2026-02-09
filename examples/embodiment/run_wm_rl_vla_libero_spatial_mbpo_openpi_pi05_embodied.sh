#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export REPO_PATH
export PRETRAINED_ROOT="${REPO_PATH}/pretrained_models"
export EMBODIED_PATH="${SCRIPT_DIR}"

MBPO_TASK_NAME="${MBPO_TASK_NAME:-libero_spatial}"
MBPO_CONFIG="${MBPO_CONFIG:-libero_spatial_mbpo_openpi_pi05}"
MBPO_FRAMES="${MBPO_FRAMES:-1000002}"
MBPO_DEMO="${MBPO_DEMO:-true}"
MBPO_WM_UPDATE_EVERY="${MBPO_WM_UPDATE_EVERY:-}"
MBPO_GPU="${MBPO_GPU:-0}"
VLA_GPUS="${VLA_GPUS:-1,2,3}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
if [[ "${MUJOCO_GL}" == "egl" ]]; then
  export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
  if [[ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" && -f "/opt/nvidia-egl/egl_vendor.d/10_nvidia.json" ]]; then
    export __EGL_VENDOR_LIBRARY_FILENAMES="/opt/nvidia-egl/egl_vendor.d/10_nvidia.json"
  fi
  if [[ -d "/opt/nvidia-egl/lib" ]]; then
    export LD_LIBRARY_PATH="/opt/nvidia-egl/lib:/.singularity.d/libs${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
fi

# LIBERO assets linkage (container-friendly)
LIBERO_PKG_ROOT="/opt/venv/openpi/lib/python3.11/site-packages/libero/libero"
LIBERO_SRC_ROOT="/opt/libero/libero/libero"
for sub in assets bddl_files init_files; do
  if [[ ! -d "${LIBERO_PKG_ROOT}/${sub}" && -d "${LIBERO_SRC_ROOT}/${sub}" ]]; then
    ln -sfn "${LIBERO_SRC_ROOT}/${sub}" "${LIBERO_PKG_ROOT}/${sub}" 2>/dev/null || echo "Warning: unable to link ${sub}; run with --writable-tmpfs or bind the path into the container." >&2
  fi
done

# Build CUDA_VISIBLE_DEVICES so GPU0 is the world model and the rest are VLA.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${MBPO_GPU}"
  if [[ -n "${VLA_GPUS}" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES},${VLA_GPUS}"
  fi
fi

# Map VLA GPUs to indices after CUDA_VISIBLE_DEVICES (world model is index 0).
IFS=',' read -r -a VLA_GPU_ARRAY <<< "${VLA_GPUS}"
VLA_COUNT=${#VLA_GPU_ARRAY[@]}
VLA_IDS=""
if [[ ${VLA_COUNT} -gt 0 ]]; then
  if [[ ${VLA_COUNT} -eq 1 ]]; then
    VLA_IDS="1"
  else
    VLA_IDS="1-$((VLA_COUNT))"
  fi
fi
export VLA_PLACEMENT="${VLA_IDS}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
export USE_TF="${USE_TF:-0}"
export USE_TB="${USE_TB:-0}"
export USE_FLAX="${USE_FLAX:-0}"

HOST_SITE="${REPO_PATH}/.singularity_pkgs/ivideogpt"
OPENPI_SITE="/opt/venv/openpi/lib/python3.11/site-packages"
export PYTHONPATH="${HOST_SITE}:${OPENPI_SITE}:${REPO_PATH}:${REPO_PATH}/iVideoGPT:${REPO_PATH}/iVideoGPT/mbrl${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  export WANDB_MODE="${WANDB_MODE:-offline}"
  export WANDB_SILENT="${WANDB_SILENT:-true}"
fi

LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H%M%S')"
mkdir -p "${LOG_DIR}"

EXTRA_ARGS=()
if [[ -n "${MBPO_EXTRA_ARGS:-}" ]]; then
  read -r -a EXTRA_ARGS <<< "${MBPO_EXTRA_ARGS}"
fi
if [[ -n "${MBPO_WM_UPDATE_EVERY}" ]]; then
  EXTRA_ARGS+=("update_gen_every_step=${MBPO_WM_UPDATE_EVERY}")
fi
if [[ -n "${VLA_EXTRA_ARGS:-}" ]]; then
  read -r -a VLA_ARGS <<< "${VLA_EXTRA_ARGS}"
  EXTRA_ARGS+=("${VLA_ARGS[@]}")
fi

CMD=(python "${REPO_PATH}/examples/embodiment/train_embodied_agent_mbpo_openpi.py" \
  --config-path "${REPO_PATH}/examples/embodiment/config" \
  --config-name "${MBPO_CONFIG}" \
  runner.logger.log_path="${LOG_DIR}" \
  task_name="${MBPO_TASK_NAME}" \
  num_train_frames="${MBPO_FRAMES}" \
  demo="${MBPO_DEMO}")

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "${CMD[@]}"
"${CMD[@]}"
