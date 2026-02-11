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

parse_gpu_csv() {
  local csv="$1"
  local -n out_arr="$2"
  local raw=()
  out_arr=()
  IFS=',' read -r -a raw <<< "${csv}"
  for gpu in "${raw[@]}"; do
    gpu="${gpu//[[:space:]]/}"
    [[ -z "${gpu}" ]] && continue
    if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
      echo "Invalid GPU id '${gpu}' in '${csv}'. Expected comma-separated integers." >&2
      exit 1
    fi
    out_arr+=("${gpu}")
  done
}

parse_gpu_csv "${MBPO_GPU}" MBPO_GPU_ARRAY
parse_gpu_csv "${VLA_GPUS}" VLA_GPU_ARRAY
if [[ ${#MBPO_GPU_ARRAY[@]} -eq 0 ]]; then
  echo "MBPO_GPU is empty after parsing. Please provide at least one GPU id." >&2
  exit 1
fi

# Remove overlapping devices from VLA if user keeps previous defaults.
declare -A MBPO_GPU_SET=()
for gpu in "${MBPO_GPU_ARRAY[@]}"; do
  MBPO_GPU_SET["${gpu}"]=1
done
FILTERED_VLA_GPU_ARRAY=()
for gpu in "${VLA_GPU_ARRAY[@]}"; do
  if [[ -n "${MBPO_GPU_SET[${gpu}]:-}" ]]; then
    echo "Warning: GPU ${gpu} appears in both MBPO_GPU and VLA_GPUS; reserving it for MBPO and removing from VLA_GPUS." >&2
    continue
  fi
  FILTERED_VLA_GPU_ARRAY+=("${gpu}")
done
VLA_GPU_ARRAY=("${FILTERED_VLA_GPU_ARRAY[@]}")

MBPO_GPU_COUNT=${#MBPO_GPU_ARRAY[@]}
VLA_GPU_COUNT=${#VLA_GPU_ARRAY[@]}
if [[ ${VLA_GPU_COUNT} -eq 0 ]]; then
  echo "No GPUs left for VLA after resolving overlaps. Please set VLA_GPUS to include at least one non-MBPO GPU." >&2
  exit 1
fi

join_by_comma() {
  local -n arr="$1"
  local out=""
  for item in "${arr[@]}"; do
    if [[ -n "${out}" ]]; then
      out+=","
    fi
    out+="${item}"
  done
  echo "${out}"
}

MBPO_GPU_CSV="$(join_by_comma MBPO_GPU_ARRAY)"
VLA_GPU_CSV="$(join_by_comma VLA_GPU_ARRAY)"

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

# Build CUDA_VISIBLE_DEVICES so MBPO GPUs are first and VLA GPUs follow.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${MBPO_GPU_CSV}"
  if [[ -n "${VLA_GPU_CSV}" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES},${VLA_GPU_CSV}"
  fi
fi

# Map VLA GPUs to indices after CUDA_VISIBLE_DEVICES.
# For example, MBPO_GPU=0,1,2 and VLA_GPUS=3 => VLA_PLACEMENT=3.
VLA_IDS=""
if [[ ${VLA_GPU_COUNT} -gt 0 ]]; then
  VLA_START="${MBPO_GPU_COUNT}"
  VLA_END="$((MBPO_GPU_COUNT + VLA_GPU_COUNT - 1))"
  if [[ ${VLA_GPU_COUNT} -eq 1 ]]; then
    VLA_IDS="${VLA_START}"
  else
    VLA_IDS="${VLA_START}-${VLA_END}"
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

HAS_WM_DEVICE_ARG=0
HAS_WM_GEN_DEVICES_ARG=0
for arg in "${EXTRA_ARGS[@]}"; do
  case "${arg}" in
    world_model.device=*) HAS_WM_DEVICE_ARG=1 ;;
    world_model.gen_devices=*) HAS_WM_GEN_DEVICES_ARG=1 ;;
  esac
done

# Keep world model primary device local to the first MBPO GPU.
if [[ ${HAS_WM_DEVICE_ARG} -eq 0 ]]; then
  EXTRA_ARGS+=("world_model.device=cuda:0")
fi

# If MBPO uses multiple GPUs, enable multi-GPU WM generation by default.
if [[ ${MBPO_GPU_COUNT} -gt 1 && ${HAS_WM_GEN_DEVICES_ARG} -eq 0 ]]; then
  WM_GEN_DEVICES=""
  for ((i = 0; i < MBPO_GPU_COUNT; ++i)); do
    if [[ -n "${WM_GEN_DEVICES}" ]]; then
      WM_GEN_DEVICES+=","
    fi
    WM_GEN_DEVICES+="cuda:${i}"
  done
  EXTRA_ARGS+=("world_model.gen_devices=[${WM_GEN_DEVICES}]")
fi

echo "[launch] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[launch] MBPO host GPUs=${MBPO_GPU_CSV} (logical cuda:0-$((MBPO_GPU_COUNT - 1)))"
echo "[launch] VLA host GPUs=${VLA_GPU_CSV} (placement=${VLA_PLACEMENT})"

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
