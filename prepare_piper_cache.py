#!/usr/bin/env python3
"""Build Piper train cache once before DDP training.

Run this on a single process. It avoids multiple DDP ranks writing the same
phoneme/audio/spectrogram cache files at the same time.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("local/secret_single_speaker"))
    parser.add_argument("--piper-dir", type=Path, default=Path("piper1-gpl"))
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    return parser.parse_args()


def load_env(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = shlex.split(value.strip())[0] if value.strip() else ""
    return values


def value(env: dict[str, str], key: str, default: str) -> str:
    return os.environ.get(key) or env.get(key) or default


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    env_path = args.env_file or (dataset_dir / "training.env")
    env = load_env(env_path)

    piper_dir = args.piper_dir.resolve()
    sys.path.insert(0, str(piper_dir / "src"))

    from piper.train.vits.dataset import VitsDataModule

    csv_path = Path(value(env, "CSV_PATH", str(dataset_dir / "metadata.csv"))).resolve()
    audio_dir = Path(value(env, "AUDIO_DIR", "/")).resolve()
    cache_dir = Path(value(env, "CACHE_DIR", str(dataset_dir / "cache"))).resolve()
    config_path = Path(value(env, "CONFIG_PATH", str(dataset_dir / "config.json"))).resolve()

    if not csv_path.exists():
        raise SystemExit(f"metadata not found: {csv_path}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    datamodule = VitsDataModule(
        espeak_voice=value(env, "ESPEAK_VOICE", "en-us"),
        config_path=config_path,
        voice_name=value(env, "VOICE_NAME", "secret-single-speaker-medium"),
        csv_path=csv_path,
        audio_dir=audio_dir,
        cache_dir=cache_dir,
        sample_rate=int(value(env, "SAMPLE_RATE", "22050")),
        batch_size=args.batch_size or int(value(env, "BATCH_SIZE", "16")),
        num_workers=args.num_workers or int(value(env, "NUM_WORKERS", "16")),
        validation_split=float(value(env, "VALIDATION_SPLIT", "0.05")),
    )
    # prepare_data() builds the cache (.phonemes.pt/.audio.pt/.spec.pt) and
    # piper_config; setup() only indexes existing cache files.
    datamodule.prepare_data()
    datamodule.setup("fit")
    print(f"cache ready: {cache_dir}")
    print(f"train_items={len(datamodule.train_dataset)} val_items={len(datamodule.val_dataset)} test_items={len(datamodule.test_dataset)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

