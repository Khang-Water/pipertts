#!/bin/bash
# Quick start script for PiperTTS VIVOS finetuning

set -e

echo "=== PiperTTS VIVOS Finetune Setup ==="
echo ""

# Check if VIVOS dataset exists
read -p "Enter path to VIVOS dataset (e.g., /home/user/vivos): " VIVOS_PATH

if [ ! -d "$VIVOS_PATH/train/waves" ]; then
    echo "❌ Error: VIVOS dataset not found at $VIVOS_PATH"
    echo "Expected structure:"
    echo "  $VIVOS_PATH/"
    echo "  ├── train/"
    echo "  │   ├── waves/"
    echo "  │   └── prompts.txt"
    echo "  └── test/"
    exit 1
fi

echo "✅ Found VIVOS at: $VIVOS_PATH"
echo ""

# Create working directory
WORK_DIR="./piper_vivos_finetune"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Clone Piper if not exists
if [ ! -d "piper1-gpl" ]; then
    echo "Cloning Piper repository..."
    git clone https://github.com/OHF-Voice/piper1-gpl.git
    cd piper1-gpl
    git checkout 2a60c2bd152356b613673ecaa0e15cbd3c0e502c
    cd ..
fi

# Setup Python environment
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q \
    "lightning>=2,<3" "tensorboard>=2,<3" \
    "jsonargparse[signatures]>=4.27.7" "onnx>=1,<2" \
    "pysilero-vad>=2.1,<3" "cython>=3,<4" "librosa<1" \
    "scikit-build<1" "pathvalidate>=3,<4" "onnxruntime>=1,<2" \
    "soundfile>=0.12,<1"

cd piper1-gpl
pip install -q -e .
bash build_monotonic_align.sh
cd ..

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit config in prepare_vivos_metadata.py (MAX_SAMPLES, BATCH_SIZE, etc.)"
echo "2. Run: python prepare_vivos_metadata.py --vivos_root $VIVOS_PATH"
echo "3. Run: bash train_vivos.sh"
