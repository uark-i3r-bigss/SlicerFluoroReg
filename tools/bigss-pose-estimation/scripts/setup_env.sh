#!/usr/bin/env bash
# Create the environment from uv.lock and fetch the pinned foundation weights.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BIGSS_POSE_ENV_DIR="${BIGSS_POSE_ENV_DIR:-${REPO}/.venv}"
export BIGSS_POSE_WEIGHTS_DIR="${BIGSS_POSE_WEIGHTS_DIR:-${REPO}/models}"
[[ "${BIGSS_POSE_ENV_DIR}" = /* ]] || export BIGSS_POSE_ENV_DIR="${PWD}/${BIGSS_POSE_ENV_DIR}"
[[ "${BIGSS_POSE_WEIGHTS_DIR}" = /* ]] || export BIGSS_POSE_WEIGHTS_DIR="${PWD}/${BIGSS_POSE_WEIGHTS_DIR}"
export UV_PROJECT_ENVIRONMENT="${BIGSS_POSE_ENV_DIR}"
cd "${REPO}"
command -v uv >/dev/null || { echo "uv is required (https://docs.astral.sh/uv/)" >&2; exit 1; }
uv_args=(sync --frozen --reinstall)
# Preserve standalone development; external snapshot environments must not link a checkout.
if [[ "${BIGSS_POSE_ENV_DIR}" != "${REPO}/.venv" ]]; then
  uv_args+=(--no-editable)
fi
PYTHONNOUSERSITE=1 UV_LINK_MODE=copy uv "${uv_args[@]}"
W="${BIGSS_POSE_WEIGHTS_DIR}/xvr_wbct/model.pth"
SHA=82a9fd05a61f22da949885e890bce9333e4d5abc091ec011ddfc4b3454d5ffba
if [ ! -f "${W}" ] || ! echo "${SHA}  ${W}" | sha256sum -c --status; then
  mkdir -p "$(dirname "${W}")"
  curl -fsSL -o "${W}.part" https://huggingface.co/eigenvivek/xvr/resolve/main/wbct/model.pth
  echo "${SHA}  ${W}.part" | sha256sum -c --status || { echo "foundation weights hash mismatch" >&2; rm -f "${W}.part"; exit 1; }
  mv "${W}.part" "${W}"
fi
"${REPO}/scripts/pose.sh" check
