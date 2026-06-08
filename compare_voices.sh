#!/usr/bin/env bash
# Synthesize one sentence with the base vais1000 voice and the fine-tuned
# voice, side by side, for quick A/B listening.
#
#   bash compare_voices.sh 'Câu cần đọc thử.'
#   bash compare_voices.sh 'Câu khác.' my_prefix
#
# The base vais1000 voice is downloaded from HuggingFace on first run.
# Override the fine-tuned model with FT_ONNX=/path/to/voice.onnx.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TEXT="${1:?usage: bash compare_voices.sh '<text to speak>' [output_prefix]}"
PREFIX="${2:-compare}"

BASE_ONNX="vi_VN-vais1000-medium.onnx"
HF_DIR="https://huggingface.co/rhasspy/piper-voices/resolve/main/vi/vi_VN/vais1000/medium"
FT_ONNX="${FT_ONNX:-vi_VN-thuyduyen-medium.onnx}"
OUT_DIR="${OUT_DIR:-compare_out}"

# Download the base voice (model + config) if missing.
if [[ ! -f "${BASE_ONNX}" ]]; then
  echo "downloading base voice: ${BASE_ONNX}"
  curl -L --fail -o "${BASE_ONNX}" "${HF_DIR}/${BASE_ONNX}"
  curl -L --fail -o "${BASE_ONNX}.json" "${HF_DIR}/${BASE_ONNX}.json"
fi

if [[ -f "${SCRIPT_DIR}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/.venv/bin/activate"
fi

mkdir -p "${OUT_DIR}"

say() {
  local model="$1" out="$2"
  if [[ ! -f "${model}" ]]; then
    echo "skip (model not found): ${model}"
    return
  fi
  python3 -m piper -m "${model}" -f "${out}" "${TEXT}" 2>/dev/null
  echo "wrote ${out}  <-  ${model}"
}

say "${BASE_ONNX}" "${OUT_DIR}/${PREFIX}_vais1000.wav"
say "${FT_ONNX}" "${OUT_DIR}/${PREFIX}_thuyduyen.wav"

echo
echo "compare:"
echo "  base      : ${OUT_DIR}/${PREFIX}_vais1000.wav"
echo "  fine-tuned: ${OUT_DIR}/${PREFIX}_thuyduyen.wav"
