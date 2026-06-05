#!/usr/bin/env python3
"""Prepare a private single-speaker dataset for Piper training.

The script intentionally avoids printing transcript text. Generated metadata,
audio copies, and stats should live under an ignored local/ directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import wave
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--audio-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("local/secret_single_speaker"))
    parser.add_argument("--delimiter", default="|", help=r"Input delimiter, e.g. '|', ',', or '\t'")
    parser.add_argument("--has-header", action="store_true")
    parser.add_argument("--path-column", type=int, default=0)
    parser.add_argument("--text-column", type=int, default=1)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--min-duration-s", type=float, default=0.3)
    parser.add_argument("--max-duration-s", type=float, default=30.0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--mode",
        choices=("reference", "copy-resample"),
        default="reference",
        help="reference keeps original files; copy-resample writes normalized wavs.",
    )
    parser.add_argument("--voice-name", default="secret-single-speaker-medium")
    parser.add_argument("--espeak-voice", default="en-us")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_delimiter(value: str) -> str:
    if value == r"\t":
        return "\t"
    if len(value) != 1:
        raise ValueError("--delimiter must resolve to one character")
    return value


def load_audio_probe(audio_path: Path) -> tuple[float, int]:
    try:
        import soundfile as sf

        info = sf.info(str(audio_path))
        if info.frames <= 0 or info.samplerate <= 0:
            raise ValueError("audio has no frames or invalid sample rate")
        return info.frames / float(info.samplerate), info.samplerate
    except ImportError:
        if audio_path.suffix.lower() != ".wav":
            raise

    with wave.open(str(audio_path), "rb") as wav_file:
        frames = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
    if frames <= 0 or sample_rate <= 0:
        raise ValueError("audio has no frames or invalid sample rate")
    return frames / float(sample_rate), sample_rate


def copy_resample_audio(source: Path, target: Path, sample_rate: int) -> None:
    import librosa
    import soundfile as sf

    audio, _ = librosa.load(str(source), sr=sample_rate, mono=True)
    if not len(audio):
        raise ValueError("audio became empty after loading")
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), audio, sample_rate, subtype="PCM_16")


def iter_rows(args: argparse.Namespace) -> Iterable[tuple[int, str, str]]:
    delimiter = resolve_delimiter(args.delimiter)
    with args.manifest.open("r", encoding="utf-8", newline="") as manifest_file:
        reader = csv.reader(manifest_file, delimiter=delimiter)
        for row_number, row in enumerate(reader, start=1):
            if row_number == 1 and args.has_header:
                continue
            if not row or all(not cell.strip() for cell in row):
                continue
            max_column = max(args.path_column, args.text_column)
            if len(row) <= max_column:
                raise ValueError(f"row {row_number}: expected column {max_column}, got {len(row)}")
            yield row_number, row[args.path_column].strip(), row[args.text_column].strip()


def resolve_audio_path(audio_dir: Path, raw_path: str) -> Path:
    audio_path = Path(raw_path)
    if not audio_path.is_absolute():
        audio_path = audio_dir / audio_path
    return audio_path.expanduser().resolve()


def sanitize_text(text: str) -> str:
    # Piper uses pipe-delimited metadata; keep transcript one field.
    return " ".join(text.replace("|", " ").split())


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    metadata_path = output_dir / "metadata.csv"
    audio_out_dir = output_dir / "audio"
    cache_dir = output_dir / "cache"
    config_path = output_dir / "config.json"
    stats_path = output_dir / "dataset_stats.json"
    env_path = output_dir / "training.env"

    accepted = 0
    skipped = {
        "missing_audio": 0,
        "empty_text": 0,
        "duration": 0,
        "audio_error": 0,
    }
    durations: list[float] = []

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        if args.mode == "copy-resample":
            audio_out_dir.mkdir(parents=True, exist_ok=True)

    metadata_target = metadata_path if not args.dry_run else Path("/dev/null")
    with metadata_target.open("w", encoding="utf-8", newline="") as metadata_file:
        writer = csv.writer(metadata_file, delimiter="|", lineterminator="\n")
        for _, raw_audio_path, raw_text in iter_rows(args):
            if args.max_samples is not None and accepted >= args.max_samples:
                break

            text = sanitize_text(raw_text)
            if not text:
                skipped["empty_text"] += 1
                continue

            source_path = resolve_audio_path(args.audio_dir, raw_audio_path)
            if not source_path.exists():
                skipped["missing_audio"] += 1
                continue

            try:
                duration_s, _ = load_audio_probe(source_path)
            except Exception:
                skipped["audio_error"] += 1
                continue

            if duration_s < args.min_duration_s or duration_s > args.max_duration_s:
                skipped["duration"] += 1
                continue

            if args.mode == "copy-resample":
                csv_audio_path = f"utt_{accepted:08d}.wav"
                target_path = audio_out_dir / csv_audio_path
                if not args.dry_run:
                    try:
                        copy_resample_audio(source_path, target_path, args.sample_rate)
                    except Exception:
                        skipped["audio_error"] += 1
                        continue
            else:
                csv_audio_path = str(source_path)

            writer.writerow([csv_audio_path, text])
            durations.append(duration_s)
            accepted += 1

    duration_stats = {
        "count": accepted,
        "total_hours": round(sum(durations) / 3600.0, 3),
        "min_s": round(min(durations), 3) if durations else None,
        "max_s": round(max(durations), 3) if durations else None,
        "mean_s": round(sum(durations) / len(durations), 3) if durations else None,
    }
    if durations:
        sorted_durations = sorted(durations)
        for percentile in (50, 90, 95, 99):
            index = min(len(sorted_durations) - 1, math.ceil(len(sorted_durations) * percentile / 100) - 1)
            duration_stats[f"p{percentile}_s"] = round(sorted_durations[index], 3)

    stats = {
        "manifest": str(args.manifest.resolve()),
        "mode": args.mode,
        "sample_rate": args.sample_rate,
        "duration_filter_s": {"min": args.min_duration_s, "max": args.max_duration_s},
        "accepted": accepted,
        "skipped": skipped,
        "duration": duration_stats,
    }

    if not args.dry_run:
        stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        audio_dir_for_training = audio_out_dir if args.mode == "copy-resample" else Path("/")
        env_values = {
            "VOICE_NAME": args.voice_name,
            "ESPEAK_VOICE": args.espeak_voice,
            "SAMPLE_RATE": str(args.sample_rate),
            "CSV_PATH": str(metadata_path),
            "AUDIO_DIR": str(audio_dir_for_training),
            "CACHE_DIR": str(cache_dir),
            "CONFIG_PATH": str(config_path),
            "RUN_DIR": str(output_dir / "runs"),
        }
        env_path.write_text(
            "\n".join(
                [f"{key}={shlex.quote(value)}" for key, value in env_values.items()] + [""]
            ),
            encoding="utf-8",
        )

    print(json.dumps(stats, indent=2))
    if not args.dry_run and accepted == 0:
        raise SystemExit("no usable rows found; check manifest columns, audio_dir, and duration filters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
