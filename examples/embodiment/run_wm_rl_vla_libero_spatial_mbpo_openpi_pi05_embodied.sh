#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export REPO_PATH
export PRETRAINED_ROOT="${REPO_PATH}/pretrained_models"
export EMBODIED_PATH="${SCRIPT_DIR}"
# Avoid host-side LD_PRELOAD entries leaking into singularity payloads.
# This suppresses repeated non-fatal loader warnings like missing libffi.so.7.
unset LD_PRELOAD

MBPO_TASK_NAME="${MBPO_TASK_NAME:-libero_spatial}"
MBPO_CONFIG="${MBPO_CONFIG:-libero_spatial_mbpo_openpi_pi05}"
MBPO_FRAMES="${MBPO_FRAMES:-1000002}"
MBPO_DEMO="${MBPO_DEMO:-true}"
MBPO_WM_UPDATE_EVERY="${MBPO_WM_UPDATE_EVERY:-}"
MBPO_GPU="${MBPO_GPU:-0}"
VLA_GPUS="${VLA_GPUS:-1,2,3}"
FORCE_SET_CUDA_VISIBLE_DEVICES="${FORCE_SET_CUDA_VISIBLE_DEVICES:-1}"
RAY_RESTART_LOCAL="${RAY_RESTART_LOCAL:-0}"

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
  # Keep host driver libs first to avoid NVML/NCCL driver-library mismatches.
  NVIDIA_LIB_PATHS=()
  [[ -d "/.singularity.d/libs" ]] && NVIDIA_LIB_PATHS+=("/.singularity.d/libs")
  [[ -d "/usr/local/nvidia/lib64" ]] && NVIDIA_LIB_PATHS+=("/usr/local/nvidia/lib64")
  [[ -d "/usr/local/nvidia/lib" ]] && NVIDIA_LIB_PATHS+=("/usr/local/nvidia/lib")
  for p in "${NVIDIA_LIB_PATHS[@]}"; do
    if [[ ":${LD_LIBRARY_PATH:-}:" != *":${p}:"* ]]; then
      export LD_LIBRARY_PATH="${p}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
  done
  # EGL helper path is appended so it does not override host NVML.
  if [[ -d "/opt/nvidia-egl/lib" && ":${LD_LIBRARY_PATH:-}:" != *":/opt/nvidia-egl/lib:"* ]]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}/opt/nvidia-egl/lib"
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
TARGET_CUDA_VISIBLE_DEVICES="${MBPO_GPU_CSV}"
if [[ -n "${VLA_GPU_CSV}" ]]; then
  TARGET_CUDA_VISIBLE_DEVICES="${TARGET_CUDA_VISIBLE_DEVICES},${VLA_GPU_CSV}"
fi

if [[ "${FORCE_SET_CUDA_VISIBLE_DEVICES}" == "1" ]]; then
  export CUDA_VISIBLE_DEVICES="${TARGET_CUDA_VISIBLE_DEVICES}"
elif [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${TARGET_CUDA_VISIBLE_DEVICES}"
else
  echo "[launch] preserving pre-set CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  if [[ "${CUDA_VISIBLE_DEVICES}" != "${TARGET_CUDA_VISIBLE_DEVICES}" ]]; then
    echo "[launch warning] requested MBPO/VLA GPUs imply CUDA_VISIBLE_DEVICES=${TARGET_CUDA_VISIBLE_DEVICES}, but current value is ${CUDA_VISIBLE_DEVICES}." >&2
    echo "[launch warning] set FORCE_SET_CUDA_VISIBLE_DEVICES=1 (default) to enforce the requested mapping." >&2
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
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-48}"
export RAY_OBJECT_STORE_BYTES="${RAY_OBJECT_STORE_BYTES:-17179869184}"
export RAY_ADDRESS="${RAY_ADDRESS:-local}"
export RLINF_NAMESPACE="${RLINF_NAMESPACE:-wm_vla_${MBPO_TASK_NAME}_$(date +%Y%m%d_%H%M%S)_$$}"

if [[ "${RAY_RESTART_LOCAL}" == "1" ]]; then
  echo "[launch warning] RAY_RESTART_LOCAL=1, stopping any existing local Ray cluster on this node."
  unset RAY_ADDRESS
  if command -v ray >/dev/null 2>&1; then
    ray stop --force >/dev/null 2>&1 || true
  fi
fi

# Ensure model/cache downloads always go to writable paths.
ensure_writable_dir() {
  local preferred="$1"
  local fallback="$2"

  if [[ -n "${preferred}" ]] && mkdir -p "${preferred}" 2>/dev/null && [[ -w "${preferred}" ]]; then
    echo "${preferred}"
    return 0
  fi

  if mkdir -p "${fallback}" 2>/dev/null && [[ -w "${fallback}" ]]; then
    echo "${fallback}"
    return 0
  fi

  echo "[launch error] No writable directory available (preferred='${preferred}', fallback='${fallback}')." >&2
  exit 1
}

DEFAULT_CACHE_ROOT="${REPO_PATH}/.cache"
TMP_CACHE_ROOT="/tmp/${USER:-wm_vla}/wm_vla_cache"
CACHE_ROOT="$(ensure_writable_dir "${CACHE_ROOT:-${DEFAULT_CACHE_ROOT}}" "${TMP_CACHE_ROOT}")"
export CACHE_ROOT

export HF_HOME="$(ensure_writable_dir "${HF_HOME:-${CACHE_ROOT}/huggingface}" "${CACHE_ROOT}/huggingface")"
export TORCH_HOME="$(ensure_writable_dir "${TORCH_HOME:-${CACHE_ROOT}/torch}" "${CACHE_ROOT}/torch")"
export HUGGINGFACE_HUB_CACHE="$(ensure_writable_dir "${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}" "${HF_HOME}/hub")"
export TRANSFORMERS_CACHE="$(ensure_writable_dir "${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}" "${HF_HOME}/transformers")"
export NUMBA_CACHE_DIR="$(ensure_writable_dir "${NUMBA_CACHE_DIR:-${CACHE_ROOT}/numba_cache}" "${CACHE_ROOT}/numba_cache")"

HOST_SITE="${REPO_PATH}/.singularity_pkgs/ivideogpt"
OPENPI_SITE="/opt/venv/openpi/lib/python3.11/site-packages"
export PYTHONPATH="${HOST_SITE}:${OPENPI_SITE}:${REPO_PATH}:${REPO_PATH}/iVideoGPT:${REPO_PATH}/iVideoGPT/mbrl${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "[wandb] WANDB_API_KEY detected; logging in before training."
  if ! python -m wandb login --relogin "${WANDB_API_KEY}" >/dev/null 2>&1; then
    echo "[wandb] login failed; falling back to offline mode." >&2
    export WANDB_MODE=offline
    export WANDB_SILENT="${WANDB_SILENT:-true}"
  else
    export WANDB_MODE="${WANDB_MODE:-online}"
  fi
else
  if [[ -z "${WANDB_MODE:-}" ]]; then
    if [[ -f "${HOME}/.netrc" ]] && grep -q "api.wandb.ai" "${HOME}/.netrc"; then
      echo "[wandb] existing login found in ${HOME}/.netrc; using online mode."
      export WANDB_MODE=online
    else
      echo "[wandb] no WANDB_API_KEY/login found; defaulting to offline mode." >&2
      export WANDB_MODE=offline
      export WANDB_SILENT="${WANDB_SILENT:-true}"
    fi
  fi
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
echo "[launch] HF_HOME=${HF_HOME}"
echo "[launch] TORCH_HOME=${TORCH_HOME}"
echo "[launch] NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR}"
echo "[launch] RLINF_NAMESPACE=${RLINF_NAMESPACE}"
echo "[launch] RAY_ADDRESS=${RAY_ADDRESS}"
echo "[launch] RAY_RESTART_LOCAL=${RAY_RESTART_LOCAL}"
echo "[launch] RAY_NUM_CPUS=${RAY_NUM_CPUS}"
echo "[launch] RAY_OBJECT_STORE_BYTES=${RAY_OBJECT_STORE_BYTES}"
echo "[launch] LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"

# Fail fast if NVML cannot initialize with current library resolution.
python - <<'PY'
import ctypes
import sys

name = "libnvidia-ml.so.1"
try:
    nvml = ctypes.CDLL(name)
except OSError as e:
    print(f"[launch error] Failed to load NVML library ({name}): {e}", file=sys.stderr)
    sys.exit(1)

nvml.nvmlErrorString.restype = ctypes.c_char_p

def err(rc):
    try:
        return nvml.nvmlErrorString(ctypes.c_int(rc)).decode("utf-8", errors="replace")
    except Exception:
        return f"code={rc}"

rc = int(nvml.nvmlInit_v2())
if rc != 0:
    print(f"[launch error] nvmlInit_v2 failed: {err(rc)}", file=sys.stderr)
    sys.exit(1)

count = ctypes.c_uint(0)
rc = int(nvml.nvmlDeviceGetCount_v2(ctypes.byref(count)))
if rc != 0:
    print(f"[launch error] nvmlDeviceGetCount_v2 failed: {err(rc)}", file=sys.stderr)
    sys.exit(1)

loaded_path = "unknown"
try:
    with open("/proc/self/maps", "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "libnvidia-ml.so" in line:
                loaded_path = line.strip().split()[-1]
                break
except Exception:
    pass

print(f"[launch] NVML library path={loaded_path}")
print(f"[launch] NVML check passed, visible GPU count={count.value}")

try:
    import torch
    print(f"[launch] torch.cuda.device_count={torch.cuda.device_count()}")
except Exception as e:
    print(f"[launch warning] torch.cuda.device_count check failed: {e}")

nvml.nvmlShutdown()
PY

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
