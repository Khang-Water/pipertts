#!/bin/bash
# Finetune PiperTTS with VIVOS dataset

set -e

# Configuration
VIVOS_ROOT="${VIVOS_ROOT:-/path/to/vivos}"  # Change this!
VOICE_NAME="${VOICE_NAME:-vi_VN-vivos-medium}"
BATCH_SIZE="${BATCH_SIZE:-32}"
CHECKPOINT="${CHECKPOINT:-}"  # Optional: path to pretrained checkpoint

# Derived paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output/${VOICE_NAME}"
CSV_PATH="${OUTPUT_DIR}/metadata.csv"
AUDIO_DIR="${VIVOS_ROOT}/train/waves"
CACHE_DIR="${OUTPUT_DIR}/cache"
CONFIG_PATH="${OUTPUT_DIR}/config.json"

echo "=== PiperTTS VIVOS Finetune ==="
echo "VIVOS root: ${VIVOS_ROOT}"
echo "Voice name: ${VOICE_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""

# Check VIVOS exists
if [ ! -d "${VIVOS_ROOT}/train/waves" ]; then
    echo "ERROR: VIVOS dataset not found at ${VIVOS_ROOT}"
    echo "Please set VIVOS_ROOT environment variable or edit this script"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Step 1: Prepare metadata CSV
echo "Step 1: Preparing metadata CSV..."
python3 prepare_vivos_for_piper.py \
    --vivos_root "${VIVOS_ROOT}" \
    --output_csv "${CSV_PATH}" \
    --split train

# Step 2: Train model
echo ""
echo "Step 2: Starting training..."
echo "This will take several hours depending on your GPU..."
echo ""

TRAIN_CMD="python3 -m piper.train fit \
    --data.voice_name \"${VOICE_NAME}\" \
    --data.csv_path \"${CSV_PATH}\" \
    --data.audio_dir \"${AUDIO_DIR}\" \
    --model.sample_rate 22050 \
    --data.espeak_voice vi \
    --data.cache_dir \"${CACHE_DIR}\" \
    --data.config_path \"${CONFIG_PATH}\" \
    --data.batch_size ${BATCH_SIZE}"

# Add checkpoint if provided
if [ -n "${CHECKPOINT}" ]; then
    echo "Using checkpoint: ${CHECKPOINT}"
    TRAIN_CMD="${TRAIN_CMD} --ckpt_path \"${CHECKPOINT}\""
fi

# Execute training
eval "${TRAIN_CMD}"

echo ""
echo "=== Training Complete ==="
echo "Config saved to: ${CONFIG_PATH}"
echo "Checkpoints in: ${OUTPUT_DIR}/lightning_logs/"
echo ""
echo "To export to ONNX:"
echo "  python3 -m piper.train.export_onnx \\"
echo "    --checkpoint ${OUTPUT_DIR}/lightning_logs/version_0/checkpoints/last.ckpt \\"
echo "    --output-file ${OUTPUT_DIR}/${VOICE_NAME}.onnx"
