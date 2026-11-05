#!/usr/bin/env bash
# Single entry point for terminal and GUI: scripts/pose.sh <command> [args]
# Runs the configured uv-managed environment; see README.md. Exit status is the CLI's.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BIGSS_POSE_ENV_DIR="${BIGSS_POSE_ENV_DIR:-${REPO}/.venv}"
export BIGSS_POSE_WEIGHTS_DIR="${BIGSS_POSE_WEIGHTS_DIR:-${REPO}/models}"
[[ "${BIGSS_POSE_ENV_DIR}" = /* ]] || export BIGSS_POSE_ENV_DIR="${PWD}/${BIGSS_POSE_ENV_DIR}"
[[ "${BIGSS_POSE_WEIGHTS_DIR}" = /* ]] || export BIGSS_POSE_WEIGHTS_DIR="${PWD}/${BIGSS_POSE_WEIGHTS_DIR}"
export UV_PROJECT_ENVIRONMENT="${BIGSS_POSE_ENV_DIR}"
PY="${BIGSS_POSE_ENV_DIR}/bin/python"
if [ ! -x "${PY}" ]; then
  echo "[bigss-pose] environment missing at ${BIGSS_POSE_ENV_DIR}: run BIGSS_POSE_ENV_DIR=\"${BIGSS_POSE_ENV_DIR}\" \"${REPO}/scripts/setup_env.sh\"" >&2
  exit 1
fi
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
# Training peaks near the free memory of a shared 24 GB GPU; expandable segments avoid the
# fragmentation (GiBs reserved but unallocated) that otherwise makes the peak OOM intermittently.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
exec "${PY}" -m bigss_pose "$@"
