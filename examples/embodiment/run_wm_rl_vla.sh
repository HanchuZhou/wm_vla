#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export REPO_PATH
export PRETRAINED_ROOT="${REPO_PATH}/pretrained_models"

MBPO_TASK_NAME="${MBPO_TASK_NAME:-libero_10}"
MBPO_FRAMES="${MBPO_FRAMES:-1000002}"
MBPO_DEMO="${MBPO_DEMO:-true}"
MBPO_GPU="${MBPO_GPU:-0}"
MBPO_CONFIG="${MBPO_CONFIG:-libero_10_mbpo_openpi_pi05_config}"
MBPO_EXTRA_ARGS_STR="${MBPO_EXTRA_ARGS:-}"

export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
unset LD_PRELOAD
export PYTHONPATH="/opt/libero${PYTHONPATH:+:${PYTHONPATH}}"
export USE_TF="${USE_TF:-0}"
export USE_TB="${USE_TB:-0}"
export USE_FLAX="${USE_FLAX:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
if [[ "${MUJOCO_GL}" == "osmesa" ]]; then
  export PYOPENGL_PLATFORM=osmesa
  export LIBGL_ALWAYS_SOFTWARE=1
elif [[ "${MUJOCO_GL}" == "egl" ]]; then
  export PYOPENGL_PLATFORM=egl
  export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
  if [[ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" && -f "/opt/nvidia-egl/egl_vendor.d/10_nvidia.json" ]]; then
    export __EGL_VENDOR_LIBRARY_FILENAMES="/opt/nvidia-egl/egl_vendor.d/10_nvidia.json"
  fi
  if [[ -d "/opt/nvidia-egl/lib" ]]; then
    export LD_LIBRARY_PATH="/opt/nvidia-egl/lib:/.singularity.d/libs${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
fi

if command -v link_assets >/dev/null 2>&1; then
  link_assets || true
fi

LIBERO_PKG_ROOT="/opt/venv/openpi/lib/python3.11/site-packages/libero/libero"
LIBERO_SRC_ROOT="/opt/libero/libero/libero"
for sub in assets bddl_files init_files; do
  if [[ ! -d "${LIBERO_PKG_ROOT}/${sub}" && -d "${LIBERO_SRC_ROOT}/${sub}" ]]; then
    ln -sfn "${LIBERO_SRC_ROOT}/${sub}" "${LIBERO_PKG_ROOT}/${sub}" 2>/dev/null || echo "Warning: unable to link ${sub}; run with --writable-tmpfs or bind the path into the container." >&2
  fi
done

if [[ "${SKIP_MBRL:-0}" != "1" ]]; then
  echo "[mbpo] task=${MBPO_TASK_NAME} config=${MBPO_CONFIG} gpu=${MBPO_GPU} frames=${MBPO_FRAMES}"
  if command -v switch_env >/dev/null 2>&1; then
    source switch_env ivideogpt
  elif [[ -f "${REPO_PATH}/.venv/bin/activate" ]]; then
    source "${REPO_PATH}/.venv/bin/activate"
  fi

  export NUMBA_CACHE_DIR="${SCRATCH:-/tmp}/numba_cache"
  mkdir -p "${NUMBA_CACHE_DIR}"

  HOST_SITE="${REPO_PATH}/.singularity_pkgs/ivideogpt"
  OPENPI_SITE="/opt/venv/openpi/lib/python3.11/site-packages"
  export PYTHONPATH="${HOST_SITE}:${OPENPI_SITE}:${REPO_PATH}:${REPO_PATH}/iVideoGPT:${REPO_PATH}/iVideoGPT/mbrl${PYTHONPATH:+:${PYTHONPATH}}"

  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export WANDB_SILENT="${WANDB_SILENT:-true}"
  fi

  MBPO_EXTRA_ARGS=()
  if [[ -n "${MBPO_EXTRA_ARGS_STR}" ]]; then
    read -r -a MBPO_EXTRA_ARGS <<< "${MBPO_EXTRA_ARGS_STR}"
  fi

  CUDA_VISIBLE_DEVICES="${MBPO_GPU}" \
    python "${REPO_PATH}/iVideoGPT/mbrl/train_libero_mbpo_openpi.py" \
    --config-name "${MBPO_CONFIG}" \
    task_name="${MBPO_TASK_NAME}" \
    num_train_frames="${MBPO_FRAMES}" \
    demo="${MBPO_DEMO}" \
    demo_path_prefix="${REPO_PATH}/iVideoGPT/mbrl/demonstrations" \
    "${MBPO_EXTRA_ARGS[@]}"
fi
