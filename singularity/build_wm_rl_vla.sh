#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEF_PATH="${DEF_PATH:-${ROOT_DIR}/singularity/wm_rl_vla.def}"
SIF_PATH="${SIF_PATH:-${ROOT_DIR}/wm_rl_vla.sif}"

cd "${ROOT_DIR}"

if command -v module >/dev/null 2>&1; then
  set +u
  source /etc/profile
  set -u
  module load singularity
fi

singularity build ${SINGULARITY_BUILD_ARGS:-} "${SIF_PATH}" "${DEF_PATH}"
