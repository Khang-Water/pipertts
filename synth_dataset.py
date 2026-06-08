#!/usr/bin/env python3
"""Batch-synthesize a Piper voice over every row in a CSV.

Reads `--text-column` from the CSV and writes one WAV per row named by
`--id-column` into `--out-dir`. The voice is loaded once and reused.
"""

from __future__ import annotations

import argparse
import csv
import sys
import wave
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="vi_VN-thuyduyen-medium.onnx", type=Path)
    parser.add_argument("--csv", default="dataset.csv", type=Path)
    parser.add_argument("--out-dir", default="piper_gpl", type=Path)
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--text-column", default="text_spoken")
    parser.add_argument("--length-scale", type=float)
    parser.add_argument("--noise-scale", type=float)
    parser.add_argument("--noise-w-scale", type=float)
    parser.add_argument("--overwrite", action="store_true", help="Re-synthesize existing WAVs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.model.exists():
        raise SystemExit(f"model not found: {args.model}")
    if not args.csv.exists():
        raise SystemExit(f"csv not found: {args.csv}")

    from piper import PiperVoice, SynthesisConfig

    voice = PiperVoice.load(str(args.model))
    syn_config = SynthesisConfig(
        length_scale=args.length_scale,
        noise_scale=args.noise_scale,
        noise_w_scale=args.noise_w_scale,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    done = 0
    skipped_existing = 0
    empty = 0
    with args.csv.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if args.id_column not in (reader.fieldnames or []):
            raise SystemExit(f"id column '{args.id_column}' not in header: {reader.fieldnames}")
        if args.text_column not in (reader.fieldnames or []):
            raise SystemExit(f"text column '{args.text_column}' not in header: {reader.fieldnames}")

        for row_number, row in enumerate(reader, start=1):
            utt_id = (row.get(args.id_column) or f"row{row_number:04d}").strip()
            text = (row.get(args.text_column) or "").strip()
            wav_path = args.out_dir / f"{utt_id}.wav"

            if wav_path.exists() and not args.overwrite:
                skipped_existing += 1
                continue
            if not text:
                empty += 1
                print(f"[skip] {utt_id}: empty text", file=sys.stderr)
                continue

            # Write to a temp path first; only the synth produces frames do we
            # keep it. synthesize_wav leaves the WAV header unset if the model
            # yields no audio, which would otherwise raise on close.
            tmp_path = wav_path.with_suffix(".wav.tmp")
            try:
                with wave.open(str(tmp_path), "wb") as wav_file:
                    voice.synthesize_wav(text, wav_file, syn_config)
                if tmp_path.stat().st_size <= 44:  # header only, no audio
                    raise ValueError("no audio produced")
                tmp_path.replace(wav_path)
                done += 1
                print(f"[ok]   {utt_id} -> {wav_path}")
            except Exception as exc:  # noqa: BLE001 - one bad row must not kill the batch
                empty += 1
                tmp_path.unlink(missing_ok=True)
                print(f"[fail] {utt_id}: {exc.__class__.__name__}: {exc}", file=sys.stderr)

    print(f"\ndone={done} skipped_existing={skipped_existing} failed_or_empty={empty} -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
