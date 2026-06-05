#!/usr/bin/env python3
"""
Load VIVOS dataset from local filesystem for PiperTTS training
Insert this cell after cell 4 (configuration) in the notebook
"""

import os
import csv
import time
import shutil
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa

# =============================================================================
# VIVOS Local Configuration
# =============================================================================

# Set this to your VIVOS dataset path
VIVOS_ROOT = "/path/to/vivos"  # Change this!
USE_LOCAL_VIVOS = True  # Set to True to use local VIVOS instead of HuggingFace

# VIVOS structure:
# vivos/
# ├── train/
# │   ├── waves/
# │   │   ├── VIVOSDEV01/
# │   │   │   ├── VIVOSDEV01_R001.wav
# │   │   │   └── ...
# │   │   └── ...
# │   └── prompts.txt
# └── test/
#     └── ...

def load_vivos_local(
    vivos_root: str,
    audio_dir: str,
    csv_path: str,
    max_samples: int = 3000,
    min_duration_s: float = 1.0,
    max_duration_s: float = 12.0,
    sample_rate: int = 22050,
    rebuild_data: bool = False,
    split: str = "train"
):
    """
    Load VIVOS dataset from local filesystem

    Args:
        vivos_root: Path to VIVOS root directory
        audio_dir: Output directory for processed WAV files
        csv_path: Output CSV file path
        max_samples: Maximum number of samples to process
        min_duration_s: Minimum audio duration in seconds
        max_duration_s: Maximum audio duration in seconds
        sample_rate: Target sample rate (will resample if needed)
        rebuild_data: If True, delete and rebuild audio_dir
        split: 'train' or 'test'
    """
    t0 = time.time()

    vivos_root = Path(vivos_root)
    split_dir = vivos_root / split
    prompts_path = split_dir / "prompts.txt"
    waves_dir = split_dir / "waves"

    # Validate paths
    if not vivos_root.exists():
        raise FileNotFoundError(
            f"VIVOS root not found: {vivos_root}\n"
            f"Please download VIVOS dataset and set VIVOS_ROOT correctly.\n"
            f"Download from: https://ailab.hcmus.edu.vn/vivos"
        )

    if not prompts_path.exists():
        raise FileNotFoundError(f"prompts.txt not found at {prompts_path}")

    if not waves_dir.exists():
        raise FileNotFoundError(f"waves directory not found at {waves_dir}")

    print(f"Loading VIVOS from: {vivos_root}")
    print(f"  prompts: {prompts_path}")
    print(f"  waves: {waves_dir}")

    # Prepare output directory
    audio_dir = Path(audio_dir)
    if rebuild_data and audio_dir.exists():
        print(f"REBUILD_DATA=True → removing {audio_dir}")
        shutil.rmtree(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Read prompts
    print(f"\nReading prompts from {prompts_path}...")
    prompts = {}
    with open(prompts_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            utt_id, text = parts
            prompts[utt_id] = text

    print(f"Found {len(prompts)} prompts")

    # Process audio files
    print(f"\nProcessing audio files...")
    n_written = 0
    n_skip = 0
    csv_rows = []

    for utt_id, text in prompts.items():
        if n_written >= max_samples:
            break

        # Find WAV file (usually in waves/SPEAKER_ID/utt_id.wav)
        wav_matches = list(waves_dir.glob(f"**/{utt_id}.wav"))
        if not wav_matches:
            n_skip += 1
            continue

        wav_path = wav_matches[0]

        try:
            # Load audio
            audio, sr = librosa.load(str(wav_path), sr=sample_rate, mono=True)

            # Check duration
            duration = len(audio) / sr
            if duration < min_duration_s or duration > max_duration_s:
                n_skip += 1
                continue

            # Save processed audio
            wav_name = f"vi_{n_written:06d}.wav"
            output_path = audio_dir / wav_name
            sf.write(str(output_path), audio, sr, subtype="PCM_16")

            # Add to CSV
            csv_rows.append((wav_name, text))
            n_written += 1

            if n_written % 500 == 0:
                print(f"  {n_written}/{max_samples} ... (skip {n_skip})  elapsed {time.time()-t0:.1f}s")

        except Exception as e:
            print(f"  Error processing {wav_path}: {e}")
            n_skip += 1
            continue

    # Write CSV
    print(f"\nWriting CSV to {csv_path}...")
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='|', quoting=csv.QUOTE_NONE, escapechar='\\')
        for wav_name, text in csv_rows:
            # Clean text: remove pipes, newlines
            clean_text = text.replace('|', ' ').replace('\n', ' ').replace('\r', ' ').strip()
            writer.writerow([wav_name, clean_text])

    print(f"\n✅ VIVOS local dataset prepared in {time.time()-t0:.1f}s")
    print(f"   Samples written: {n_written}")
    print(f"   Samples skipped: {n_skip}")
    print(f"   Audio dir: {audio_dir}")
    print(f"   CSV path: {csv_path}")

    return n_written


# =============================================================================
# Main execution (for notebook cell)
# =============================================================================

if __name__ == "__main__" or "get_ipython" in dir():
    # This runs when executed as a notebook cell

    if USE_LOCAL_VIVOS:
        print("=" * 70)
        print("Using LOCAL VIVOS dataset")
        print("=" * 70)

        # Validate VIVOS_ROOT
        if VIVOS_ROOT == "/path/to/vivos":
            raise ValueError(
                "Please set VIVOS_ROOT to your actual VIVOS dataset path!\n"
                "Example: VIVOS_ROOT = '/home/user/datasets/vivos'"
            )

        # Load VIVOS local
        n_samples = load_vivos_local(
            vivos_root=VIVOS_ROOT,
            audio_dir=AUDIO_DIR,
            csv_path=CSV_PATH,
            max_samples=MAX_SAMPLES,
            min_duration_s=MIN_DURATION_S,
            max_duration_s=MAX_DURATION_S,
            sample_rate=SAMPLE_RATE,
            rebuild_data=REBUILD_DATA,
            split="train"
        )

        # Show sample
        print("\nFirst 3 lines of CSV:")
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                print(f"  {line.rstrip()}")

        print(f"\nTotal WAV files: {len(list(Path(AUDIO_DIR).glob('*.wav')))}")

        # Skip cell 6 (HuggingFace dataset loading)
        print("\n⚠️  SKIP CELL 6 (HuggingFace dataset loading) - already loaded from local VIVOS")
    else:
        print("USE_LOCAL_VIVOS=False → will use HuggingFace dataset (run cell 6)")
