#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_CKPT_URL="${BASE_CKPT_URL:-https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/vi/vi_VN/vais1000/medium/epoch%3D4769-step%3D919580.ckpt}"
BASE_CKPT_PATH="${BASE_CKPT_PATH:-${SCRIPT_DIR}/local/checkpoints/vi_VN-vais1000-medium.ckpt}"
SANITIZED_CKPT_PATH="${SANITIZED_CKPT_PATH:-${BASE_CKPT_PATH%.ckpt}.sanitized.ckpt}"

mkdir -p "$(dirname "${BASE_CKPT_PATH}")"

if [[ ! -f "${BASE_CKPT_PATH}" ]]; then
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --output "${BASE_CKPT_PATH}" "${BASE_CKPT_URL}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${BASE_CKPT_PATH}" "${BASE_CKPT_URL}"
  else
    echo "need curl or wget to download checkpoint" >&2
    exit 1
  fi
else
  echo "checkpoint exists: ${BASE_CKPT_PATH}"
fi

python3 "${SCRIPT_DIR}/sanitize_piper_checkpoint.py" \
  --checkpoint "${BASE_CKPT_PATH}" \
  --output "${SANITIZED_CKPT_PATH}" \
  --piper-dir "${SCRIPT_DIR}/piper1-gpl"

echo "set this before training:"
echo "export CHECKPOINT=${SANITIZED_CKPT_PATH}"

