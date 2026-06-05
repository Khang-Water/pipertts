#!/usr/bin/env bash
# One-command pipeline for the secret single-speaker Piper fine-tune.
#
# Designed for unattended runs inside tmux:
#   tmux new -s piper
#   bash run_pipeline.sh
#
# Every stage is idempotent. Re-running after a crash resumes training from
# the newest last.ckpt and keeps the original step target (no moving target).
# Configure once in pipeline.env (see pipeline.env.example).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -f "${SCRIPT_DIR}/pipeline.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/pipeline.env"
  set +a
fi

# --- Config (pipeline.env or environment overrides these defaults) ---------
PIPER_DIR="${PIPER_DIR:-${SCRIPT_DIR}/piper1-gpl}"
DATASET_DIR="${DATASET_DIR:-${SCRIPT_DIR}/local/secret_single_speaker}"
MANIFEST="${MANIFEST:-}"
SRC_AUDIO_DIR="${SRC_AUDIO_DIR:-/}"
MANIFEST_DELIMITER="${MANIFEST_DELIMITER:-|}"
MANIFEST_HAS_HEADER="${MANIFEST_HAS_HEADER:-0}"
PATH_COLUMN="${PATH_COLUMN:-0}"
TEXT_COLUMN="${TEXT_COLUMN:-1}"
VOICE_NAME="${VOICE_NAME:-vi_VN-secret-medium}"
ESPEAK_VOICE="${ESPEAK_VOICE:-vi}"
SAMPLE_RATE="${SAMPLE_RATE:-22050}"
MIN_DURATION_S="${MIN_DURATION_S:-0.5}"
MAX_DURATION_S="${MAX_DURATION_S:-30}"
NUM_GPUS="${NUM_GPUS:-2}"
EXTRA_STEPS="${EXTRA_STEPS:-20000}"
STRICT_PREFLIGHT="${STRICT_PREFLIGHT:-1}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"

RUN_DIR="${RUN_DIR:-${DATASET_DIR}/runs}"
STATE_DIR="${DATASET_DIR}/.pipeline"
LOG_DIR="${SCRIPT_DIR}/local/logs"
BASE_CKPT="${SANITIZED_CKPT_PATH:-${SCRIPT_DIR}/local/checkpoints/vi_VN-vais1000-medium.sanitized.ckpt}"
TARGET_STEP_FILE="${STATE_DIR}/target_step"

mkdir -p "${LOG_DIR}" "${STATE_DIR}"
LOG_FILE="${LOG_DIR}/pipeline_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${LOG_FILE}") 2>&1

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() { log "FATAL: $*"; exit 1; }
trap 'log "FAILED at line ${LINENO} (see ${LOG_FILE})"' ERR

ckpt_step() {
  python3 - "$1" <<'PY'
import sys

import torch

ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
step = ckpt.get("global_step")
if step is None:
    raise SystemExit("checkpoint has no global_step key")
print(int(step))
PY
}

newest_last_ckpt() {
  [[ -d "${RUN_DIR}" ]] || return 0
  find "${RUN_DIR}" -name 'last.ckpt' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | awk 'NR == 1 {$1=""; sub(/^ /, ""); print}'
}

log "pipeline start: log=${LOG_FILE}"

# --- Stage 1: system dependencies -------------------------------------------
log "stage 1/9: system dependencies"
missing=()
for cmd in cmake ninja espeak-ng cc python3; do
  command -v "${cmd}" >/dev/null 2>&1 || missing+=("${cmd}")
done
if (( ${#missing[@]} > 0 )); then
  die "missing system commands: ${missing[*]} — run: sudo apt-get install -y build-essential cmake ninja-build espeak-ng"
fi

# --- Stage 2: venv + piper install + compiled extensions --------------------
log "stage 2/9: python environment"
if [[ ! -f "${SCRIPT_DIR}/.venv/bin/activate" ]]; then
  python3 -m venv "${SCRIPT_DIR}/.venv"
fi
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/.venv/bin/activate"

if python3 - <<'PY' >/dev/null 2>&1
import lightning
import torch

import piper.espeakbridge
import piper.train.vits.monotonic_align
PY
then
  log "environment ready (skip install)"
else
  log "installing piper[train] + building extensions (first run: slow, downloads torch)"
  pip install -e "${PIPER_DIR}[train]"
  (cd "${PIPER_DIR}" && bash build_monotonic_align.sh)
  (cd "${PIPER_DIR}" && python3 setup.py build_ext --inplace)
fi

# --- Stage 3: dataset --------------------------------------------------------
log "stage 3/9: dataset"
if [[ -f "${DATASET_DIR}/metadata.csv" && -f "${DATASET_DIR}/training.env" && "${FORCE_PREPARE:-0}" != "1" ]]; then
  log "dataset ready: ${DATASET_DIR} (FORCE_PREPARE=1 to rebuild)"
else
  [[ -n "${MANIFEST}" ]] || die "MANIFEST is not set; copy pipeline.env.example to pipeline.env and fill it in"
  [[ -f "${MANIFEST}" ]] || die "manifest not found: ${MANIFEST}"
  prepare_args=(
    --manifest "${MANIFEST}"
    --audio-dir "${SRC_AUDIO_DIR}"
    --output-dir "${DATASET_DIR}"
    --delimiter "${MANIFEST_DELIMITER}"
    --path-column "${PATH_COLUMN}"
    --text-column "${TEXT_COLUMN}"
    --sample-rate "${SAMPLE_RATE}"
    --min-duration-s "${MIN_DURATION_S}"
    --max-duration-s "${MAX_DURATION_S}"
    --voice-name "${VOICE_NAME}"
    --espeak-voice "${ESPEAK_VOICE}"
  )
  [[ "${MANIFEST_HAS_HEADER}" == "1" ]] && prepare_args+=(--has-header)
  python3 "${SCRIPT_DIR}/prepare_single_speaker_dataset.py" "${prepare_args[@]}"
fi

# --- Stage 4: base checkpoint ------------------------------------------------
log "stage 4/9: base checkpoint"
bash "${SCRIPT_DIR}/download_base_checkpoint.sh"
[[ -f "${BASE_CKPT}" ]] || die "sanitized base checkpoint not found: ${BASE_CKPT}"

# --- Stage 5: resolve resume checkpoint + fixed step target ------------------
log "stage 5/9: resolve checkpoint and step target"
RESUME_CKPT="$(newest_last_ckpt)"
if [[ -n "${RESUME_CKPT}" ]]; then
  CHECKPOINT="${RESUME_CKPT}"
  log "resuming from: ${CHECKPOINT}"
else
  CHECKPOINT="${BASE_CKPT}"
  log "starting from base: ${CHECKPOINT}"
fi

if [[ -f "${TARGET_STEP_FILE}" ]]; then
  TARGET_STEP="$(cat "${TARGET_STEP_FILE}")"
  log "existing target: ${TARGET_STEP} global steps (delete ${TARGET_STEP_FILE} to retarget)"
else
  BASE_STEP="$(ckpt_step "${BASE_CKPT}")"
  TARGET_STEP="$((BASE_STEP + EXTRA_STEPS))"
  echo "${TARGET_STEP}" > "${TARGET_STEP_FILE}"
  log "new target: base ${BASE_STEP} + EXTRA_STEPS ${EXTRA_STEPS} = ${TARGET_STEP}"
fi

CURRENT_STEP="$(ckpt_step "${CHECKPOINT}")"
log "current step: ${CURRENT_STEP} / target: ${TARGET_STEP}"

TRAIN_NEEDED=1
if (( CURRENT_STEP >= TARGET_STEP )); then
  TRAIN_NEEDED=0
  log "target already reached — skipping preflight/cache/train, going to export"
fi

if (( TRAIN_NEEDED )); then
  # --- Stage 6: preflight -----------------------------------------------------
  log "stage 6/9: preflight"
  preflight_args=(--dataset-dir "${DATASET_DIR}" --piper-dir "${PIPER_DIR}" --expect-gpus "${NUM_GPUS}")
  [[ "${STRICT_PREFLIGHT}" == "1" ]] && preflight_args+=(--strict)
  CHECKPOINT="${CHECKPOINT}" MAX_STEPS="${TARGET_STEP}" \
    python3 "${SCRIPT_DIR}/preflight_training.py" "${preflight_args[@]}"

  # --- Stage 7: cache (single process, before DDP) ----------------------------
  log "stage 7/9: build cache"
  python3 "${SCRIPT_DIR}/prepare_piper_cache.py" \
    --dataset-dir "${DATASET_DIR}" \
    --piper-dir "${PIPER_DIR}"

  # --- Stage 8: dry-run config validation, then train -------------------------
  log "stage 8/9: train (dry-run first)"
  DRY_RUN=1 CHECKPOINT="${CHECKPOINT}" MAX_STEPS="${TARGET_STEP}" EXTRA_STEPS= \
    bash "${SCRIPT_DIR}/finetune_single_speaker.sh" >/dev/null
  log "dry-run OK, starting real training"
  CHECKPOINT="${CHECKPOINT}" MAX_STEPS="${TARGET_STEP}" EXTRA_STEPS= \
    bash "${SCRIPT_DIR}/finetune_single_speaker.sh"
fi

# --- Stage 9: export ----------------------------------------------------------
if [[ "${SKIP_EXPORT}" == "1" ]]; then
  log "stage 9/9: export skipped (SKIP_EXPORT=1)"
else
  log "stage 9/9: export ONNX"
  CHECKPOINT= bash "${SCRIPT_DIR}/export_single_speaker_onnx.sh"
fi

log "pipeline done. artifacts: ${DATASET_DIR}/export — full log: ${LOG_FILE}"
