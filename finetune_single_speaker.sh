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
ESPEAK_VOICE="${ESPEAK_VOICE:-en-us}"
SAMPLE_RATE="${SAMPLE_RATE:-22050}"
CSV_PATH="${CSV_PATH:-${DATASET_DIR}/metadata.csv}"
AUDIO_DIR="${AUDIO_DIR:-/}"
CACHE_DIR="${CACHE_DIR:-${DATASET_DIR}/cache}"
CONFIG_PATH="${CONFIG_PATH:-${DATASET_DIR}/config.json}"
RUN_DIR="${RUN_DIR:-${DATASET_DIR}/runs}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-16}"
VALIDATION_SPLIT="${VALIDATION_SPLIT:-0.05}"
MAX_STEPS="${MAX_STEPS:-}"
EXTRA_STEPS="${EXTRA_STEPS:-}"
PRECISION="${PRECISION:-bf16-mixed}"
NUM_GPUS="${NUM_GPUS:-2}"
STRATEGY="${STRATEGY:-ddp}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-50}"
CKPT_EVERY_N_STEPS="${CKPT_EVERY_N_STEPS:-5000}"
CKPT_SAVE_TOP_K="${CKPT_SAVE_TOP_K:-1}"
SEED="${SEED:-31337}"
DEFAULT_BASE_CKPT="${DEFAULT_BASE_CKPT:-${SCRIPT_DIR}/local/checkpoints/vi_VN-vais1000-medium.sanitized.ckpt}"
CHECKPOINT="${CHECKPOINT:-}"

if [[ -z "${CHECKPOINT}" && -f "${DEFAULT_BASE_CKPT}" ]]; then
  CHECKPOINT="${DEFAULT_BASE_CKPT}"
fi

if [[ ! -f "${CSV_PATH}" ]]; then
  echo "metadata not found: ${CSV_PATH}" >&2
  exit 1
fi

if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
  echo "checkpoint not found; run download_base_checkpoint.sh or set CHECKPOINT=/path/to/base.ckpt" >&2
  exit 1
fi

# --ckpt_path resumes trainer state, so max_steps is compared against the
# checkpoint's restored global_step (not zero). Read it up front so we can
# fail fast instead of silently training for 0 steps.
BASE_GLOBAL_STEP="$(python3 - "${CHECKPOINT}" <<'PY'
import sys

import torch

ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
step = ckpt.get("global_step")
if step is None:
    raise SystemExit("checkpoint has no global_step key; is this a Lightning checkpoint?")
print(int(step))
PY
)"

if [[ -n "${EXTRA_STEPS}" && -n "${MAX_STEPS}" ]]; then
  echo "set either EXTRA_STEPS (steps beyond the checkpoint) or MAX_STEPS (absolute), not both" >&2
  exit 1
fi

if [[ -n "${EXTRA_STEPS}" ]]; then
  MAX_STEPS="$((BASE_GLOBAL_STEP + EXTRA_STEPS))"
fi

if [[ -n "${MAX_STEPS}" && "${MAX_STEPS}" -le "${BASE_GLOBAL_STEP}" ]]; then
  echo "MAX_STEPS=${MAX_STEPS} <= checkpoint global_step=${BASE_GLOBAL_STEP}: trainer would stop immediately." >&2
  echo "use EXTRA_STEPS=<n> for steps beyond the checkpoint, or raise MAX_STEPS." >&2
  exit 1
fi

mkdir -p "${CACHE_DIR}" "${RUN_DIR}" "$(dirname "${CONFIG_PATH}")"

export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"

cd "${PIPER_DIR}"

strategy_args=()
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  strategy_args=(--trainer.strategy "${STRATEGY}")
else
  strategy_args=(--trainer.strategy "auto")
fi

args=(
  -m piper.train fit
  --seed_everything "${SEED}"
  --data.voice_name "${VOICE_NAME}"
  --data.csv_path "${CSV_PATH}"
  --data.audio_dir "${AUDIO_DIR}"
  --data.cache_dir "${CACHE_DIR}"
  --data.config_path "${CONFIG_PATH}"
  --data.espeak_voice "${ESPEAK_VOICE}"
  --data.batch_size "${BATCH_SIZE}"
  --data.num_workers "${NUM_WORKERS}"
  --data.validation_split "${VALIDATION_SPLIT}"
  --model.sample_rate "${SAMPLE_RATE}"
  --trainer.default_root_dir "${RUN_DIR}"
  --trainer.accelerator gpu
  --trainer.devices "${NUM_GPUS}"
  --trainer.precision "${PRECISION}"
  --trainer.accumulate_grad_batches "${ACCUMULATE_GRAD_BATCHES}"
  --trainer.log_every_n_steps "${LOG_EVERY_N_STEPS}"
  --trainer.enable_checkpointing true
  --trainer.callbacks+=lightning.pytorch.callbacks.ModelCheckpoint
  --trainer.callbacks.every_n_train_steps "${CKPT_EVERY_N_STEPS}"
  --trainer.callbacks.save_top_k "${CKPT_SAVE_TOP_K}"
  --trainer.callbacks.save_last true
  "${strategy_args[@]}"
  --ckpt_path "${CHECKPOINT}"
)

if [[ -n "${MAX_STEPS}" ]]; then
  args+=(--trainer.max_steps "${MAX_STEPS}")
fi

# DRY_RUN=1 validates the full CLI config (jsonargparse parsing, callback
# class paths, linked args) and exits without touching the GPUs.
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  args+=(--print_config)
fi

echo "voice=${VOICE_NAME}"
echo "gpus=${NUM_GPUS} strategy=${STRATEGY} precision=${PRECISION}"
echo "batch_size_per_gpu=${BATCH_SIZE} effective_batch=$((BATCH_SIZE * NUM_GPUS * ACCUMULATE_GRAD_BATCHES))"
echo "checkpoint=${CHECKPOINT}"
echo "base_global_step=${BASE_GLOBAL_STEP} max_steps=${MAX_STEPS:-unlimited}"
echo "ckpt_every_n_steps=${CKPT_EVERY_N_STEPS} save_top_k=${CKPT_SAVE_TOP_K} save_last=true"
echo "cache_dir=${CACHE_DIR}"
python3 "${args[@]}"

