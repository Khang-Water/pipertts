#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPER_DIR="${PIPER_DIR:-${SCRIPT_DIR}/piper1-gpl}"
DATASET_DIR="${DATASET_DIR:-${SCRIPT_DIR}/local/secret_single_speaker}"

if [[ -f "${DATASET_DIR}/training.env" ]]; then
  # shellcheck disable=SC1091
  source "${DATASET_DIR}/training.env"
fi

VOICE_NAME="${VOICE_NAME:-secret-single-speaker-medium}"
RUN_DIR="${RUN_DIR:-${DATASET_DIR}/runs}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATASET_DIR}/export}"
CHECKPOINT="${CHECKPOINT:-}"

newest_ckpt() {
  find "${RUN_DIR}" -name "$1" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | awk 'NR == 1 {$1=""; sub(/^ /, ""); print}'
}

if [[ -z "${CHECKPOINT}" && -d "${RUN_DIR}" ]]; then
  # Prefer last.ckpt (written by ModelCheckpoint save_last=true); fall back
  # to the newest step checkpoint, e.g. "epoch=X-step=Y.ckpt".
  CHECKPOINT="$(newest_ckpt 'last.ckpt')"
  if [[ -z "${CHECKPOINT}" ]]; then
    CHECKPOINT="$(newest_ckpt '*.ckpt')"
  fi
fi

if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
  echo "checkpoint not found; set CHECKPOINT=/path/to/file.ckpt" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
cd "${PIPER_DIR}"

python3 -m piper.train.export_onnx \
  --checkpoint "${CHECKPOINT}" \
  --output-file "${OUTPUT_DIR}/${VOICE_NAME}.onnx"

CONFIG_PATH="${CONFIG_PATH:-${DATASET_DIR}/config.json}"
if [[ -f "${CONFIG_PATH}" ]]; then
  cp "${CONFIG_PATH}" "${OUTPUT_DIR}/${VOICE_NAME}.onnx.json"
fi

echo "exported: ${OUTPUT_DIR}/${VOICE_NAME}.onnx"
