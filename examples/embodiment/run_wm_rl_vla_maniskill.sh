#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export REPO_PATH
export PRETRAINED_ROOT="${REPO_PATH}/pretrained_models"

MBPO_TASK_NAME="${MBPO_TASK_NAME:-PutOnPlateInScene25Main-v3}"
MBPO_FRAMES="${MBPO_FRAMES:-3000000}"
MBPO_DEMO="${MBPO_DEMO:-false}"
MBPO_GPU="${MBPO_GPU:-0}"
MBPO_CONFIG="${MBPO_CONFIG:-maniskill_mbpo_openpi_config}"
MBPO_EXTRA_ARGS_STR="${MBPO_EXTRA_ARGS:-}"

VLA_CONFIG="${VLA_CONFIG:-maniskill_ppo_openvlaoft}"
VLA_GPUS="${VLA_GPUS:-0,1,2}"
RAY_NUM_GPUS="${RAY_NUM_GPUS:-}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-16}"
RAY_OBJ_STORE_BYTES="${RAY_OBJ_STORE_BYTES:-2147483648}"
START_RAY="${START_RAY:-1}"
VLA_EXTRA_ARGS_STR="${VLA_EXTRA_ARGS:-}"

if [[ -z "${RAY_NUM_GPUS}" ]]; then
  IFS=',' read -r -a _vla_gpu_list <<< "${VLA_GPUS}"
  RAY_NUM_GPUS="${#_vla_gpu_list[@]}"
fi

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
    python "${REPO_PATH}/iVideoGPT/mbrl/train_maniskill_mbpo_openpi.py" \
    --config-name "${MBPO_CONFIG}" \
    task_name="${MBPO_TASK_NAME}" \
    num_train_frames="${MBPO_FRAMES}" \
    demo="${MBPO_DEMO}" \
    demo_path_prefix="${REPO_PATH}/iVideoGPT/mbrl/demonstrations" \
    "${MBPO_EXTRA_ARGS[@]}"
fi

if [[ "${SKIP_VLA:-0}" != "1" ]]; then
  echo "[vla] config=${VLA_CONFIG} gpus=${VLA_GPUS} ray_gpus=${RAY_NUM_GPUS}"
  if command -v switch_env >/dev/null 2>&1; then
    source switch_env openpi
  fi

  export CUDA_VISIBLE_DEVICES="${VLA_GPUS}"
  if [[ -z "${RAY_TMPDIR:-}" ]]; then
    export RAY_TMPDIR="${SCRATCH:-/tmp}/ray_tmp_${USER}_$(date +%s)"
  fi
  if [[ -z "${RAY_STORAGE:-}" ]]; then
    export RAY_STORAGE="${SCRATCH:-/tmp}/ray_storage_${USER}_$(date +%s)"
  fi
  mkdir -p "${RAY_TMPDIR}" "${RAY_STORAGE}"

  if [[ "${START_RAY}" == "1" ]]; then
    ray stop --force >/dev/null 2>&1 || true
    ray start --head \
      --dashboard-port=0 \
      --object-store-memory="${RAY_OBJ_STORE_BYTES}" \
      --temp-dir "${RAY_TMPDIR}" \
      --storage "${RAY_STORAGE}" \
      --num-gpus="${RAY_NUM_GPUS}" \
      --num-cpus="${RAY_NUM_CPUS}"
  fi

  export RAY_ADDRESS=auto
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "WANDB_API_KEY not set; wandb logging may be disabled." >&2
  fi

  EMBODIMENT_EXTRA_ARGS="${VLA_EXTRA_ARGS_STR}" \
    bash "${REPO_PATH}/examples/embodiment/run_embodiment.sh" "${VLA_CONFIG}"
fi
