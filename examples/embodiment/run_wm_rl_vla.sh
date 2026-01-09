#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export REPO_PATH
export PRETRAINED_ROOT="${REPO_PATH}/pretrained_models"

MBPO_TASK_NAME="${MBPO_TASK_NAME:-libero_10}"
MBPO_FRAMES="${MBPO_FRAMES:-1000002}"
MBPO_DEMO="${MBPO_DEMO:-false}"
MBPO_GPU="${MBPO_GPU:-0}"
MBPO_CONFIG="${MBPO_CONFIG:-libero_mbpo_openpi_config}"

VLA_CONFIG="${VLA_CONFIG:-libero_10_grpo_openpi_pi05}"
VLA_GPUS="${VLA_GPUS:-1,2}"
RAY_NUM_GPUS="${RAY_NUM_GPUS:-2}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-16}"
RAY_OBJ_STORE_BYTES="${RAY_OBJ_STORE_BYTES:-2147483648}"
START_RAY="${START_RAY:-1}"

export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
unset LD_PRELOAD
export PYTHONPATH="/opt/libero${PYTHONPATH:+:${PYTHONPATH}}"

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

  CUDA_VISIBLE_DEVICES="${MBPO_GPU}" \
    python "${REPO_PATH}/iVideoGPT/mbrl/train_libero_mbpo_openpi.py" \
    --config-name "${MBPO_CONFIG}" \
    task_name="${MBPO_TASK_NAME}" \
    num_train_frames="${MBPO_FRAMES}" \
    demo="${MBPO_DEMO}" \
    demo_path_prefix="${REPO_PATH}/iVideoGPT/mbrl/demonstrations"
fi

if [[ "${SKIP_VLA:-0}" != "1" ]]; then
  if command -v switch_env >/dev/null 2>&1; then
    source switch_env openpi
  fi

  export CUDA_VISIBLE_DEVICES="${VLA_GPUS}"
  export RAY_TMPDIR="${SCRATCH:-/tmp}/ray_tmp"
  export RAY_STORAGE="${SCRATCH:-/tmp}/ray_storage"
  mkdir -p "${RAY_TMPDIR}" "${RAY_STORAGE}"

  if [[ "${START_RAY}" == "1" ]]; then
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

  bash "${REPO_PATH}/examples/embodiment/run_embodiment.sh" "${VLA_CONFIG}"
fi
