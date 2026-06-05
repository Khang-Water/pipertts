#!/usr/bin/env python3
"""
Prepare VIVOS dataset for PiperTTS training
Converts VIVOS format to PiperTTS CSV format
"""

import json
from pathlib import Path
import argparse


def prepare_vivos_metadata(
    vivos_root: str,
    output_csv: str,
    max_samples: int = None,
    split: str = "train"
):
    """
    Convert VIVOS prompts.txt to PiperTTS CSV format

    VIVOS format:
        train/prompts.txt: VIVOSDEV01_R001 Đây là câu văn mẫu
        train/waves/VIVOSDEV01/VIVOSDEV01_R001.wav

    PiperTTS format (CSV with | delimiter):
        audio_file.wav|Text for utterance.

    Args:
        vivos_root: Path to VIVOS dataset root (contains train/ and test/)
        output_csv: Path to output CSV file
        max_samples: Maximum number of samples (None = all)
        split: 'train' or 'test'
    """
    vivos_root = Path(vivos_root)
    split_dir = vivos_root / split
    prompts_path = split_dir / "prompts.txt"
    waves_dir = split_dir / "waves"

    if not prompts_path.exists():
        raise FileNotFoundError(f"prompts.txt not found at {prompts_path}")
    if not waves_dir.exists():
        raise FileNotFoundError(f"waves directory not found at {waves_dir}")

    print(f"Reading prompts from: {prompts_path}")
    print(f"Audio directory: {waves_dir}")

    rows = []
    missing = 0

    with open(prompts_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue

            utt_id, text = parts

            # Find wav file (usually in waves/SPEAKER_ID/utt_id.wav)
            wav_matches = list(waves_dir.glob(f"**/{utt_id}.wav"))
            if not wav_matches:
                missing += 1
                continue

            wav_path = wav_matches[0]

            # PiperTTS CSV format: relative_path|text
            rows.append(f"{wav_path}|{text}")

            if max_samples and len(rows) >= max_samples:
                break

    if not rows:
        raise RuntimeError(f"No valid samples found in {prompts_path}")

    # Write CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(rows))

    print(f"\n✓ Created {output_path}")
    print(f"  Total samples: {len(rows)}")
    print(f"  Missing wavs: {missing}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Prepare VIVOS for PiperTTS")
    parser.add_argument(
        "--vivos_root",
        type=str,
        required=True,
        help="Path to VIVOS dataset root directory"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./vivos_train.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples (default: all)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split to use"
    )

    args = parser.parse_args()

    prepare_vivos_metadata(
        vivos_root=args.vivos_root,
        output_csv=args.output_csv,
        max_samples=args.max_samples,
        split=args.split
    )


if __name__ == "__main__":
    main()
