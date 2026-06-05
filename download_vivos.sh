#!/bin/bash
# Download VIVOS Vietnamese Speech Corpus

set -e

DOWNLOAD_DIR="${1:-./data/vivos}"

echo "Downloading VIVOS dataset to: ${DOWNLOAD_DIR}"
mkdir -p "${DOWNLOAD_DIR}"

# VIVOS is hosted on ailab.hcmus.edu.vn
# You may need to download manually from:
# https://ailab.hcmus.edu.vn/vivos

echo ""
echo "VIVOS dataset download options:"
echo ""
echo "1. Official source (requires manual download):"
echo "   https://ailab.hcmus.edu.vn/vivos"
echo ""
echo "2. Alternative (if available):"
echo "   wget https://huggingface.co/datasets/vivos/resolve/main/data.zip"
echo ""
echo "After downloading, extract to: ${DOWNLOAD_DIR}"
echo ""
echo "Expected structure:"
echo "  ${DOWNLOAD_DIR}/"
echo "  ├── train/"
echo "  │   ├── waves/"
echo "  │   └── prompts.txt"
echo "  └── test/"
echo "      ├── waves/"
echo "      └── prompts.txt"
