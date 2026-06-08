#!/usr/bin/env python3
"""Preflight checks for serious Piper fine-tuning runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import wave
from pathlib import Path


REQUIRED_IMPORTS = [
    "torch",
    "lightning",
    "librosa",
    "soundfile",
    "pysilero_vad",
    "onnx",
    "jsonargparse",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("local/secret_single_speaker"))
    parser.add_argument("--piper-dir", type=Path, default=Path("piper1-gpl"))
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--expect-gpus", type=int, default=2)
    parser.add_argument("--sample-rows", type=int, default=200)
    parser.add_argument("--strict", action="store_true")
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


def audio_duration(path: Path) -> float | None:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return info.frames / float(info.samplerate)
    except Exception:
        if path.suffix.lower() != ".wav":
            return None
        try:
            with wave.open(str(path), "rb") as wav_file:
                return wav_file.getnframes() / float(wav_file.getframerate())
        except Exception:
            return None


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    env_path = args.env_file or (dataset_dir / "training.env")
    env = load_env(env_path)
    piper_dir = args.piper_dir.resolve()
    sys.path.insert(0, str(piper_dir / "src"))

    report: dict[str, object] = {"ok": True, "errors": [], "warnings": []}
    errors: list[str] = report["errors"]  # type: ignore[assignment]
    warnings: list[str] = report["warnings"]  # type: ignore[assignment]

    csv_path = Path(value(env, "CSV_PATH", str(dataset_dir / "metadata.csv"))).resolve()
    audio_dir = Path(value(env, "AUDIO_DIR", "/")).resolve()
    cache_dir = Path(value(env, "CACHE_DIR", str(dataset_dir / "cache"))).resolve()
    config_path = Path(value(env, "CONFIG_PATH", str(dataset_dir / "config.json"))).resolve()
    checkpoint = Path(value(env, "CHECKPOINT", value(env, "DEFAULT_BASE_CKPT", "local/checkpoints/vi_VN-vais1000-medium.sanitized.ckpt"))).resolve()

    report["paths"] = {
        "csv_exists": csv_path.exists(),
        "audio_dir_exists": audio_dir.exists(),
        "cache_parent_exists": cache_dir.parent.exists(),
        "config_parent_exists": config_path.parent.exists(),
        "checkpoint_exists": checkpoint.exists(),
    }
    for path_name, exists in report["paths"].items():  # type: ignore[union-attr]
        if not exists:
            errors.append(f"{path_name}=false")

    import_results: dict[str, str] = {}
    for module in REQUIRED_IMPORTS:
        try:
            imported = __import__(module)
            import_results[module] = getattr(imported, "__version__", "ok")
        except Exception as exc:
            import_results[module] = f"missing:{exc.__class__.__name__}"
            errors.append(f"missing import: {module}")
    report["imports"] = import_results

    try:
        import piper.train.vits.monotonic_align  # noqa: F401

        report["monotonic_align"] = "ok"
    except Exception as exc:
        report["monotonic_align"] = f"missing:{exc.__class__.__name__}"
        errors.append("monotonic_align extension missing; run piper1-gpl/build_monotonic_align.sh")

    try:
        import piper.espeakbridge  # noqa: F401

        report["espeakbridge"] = "ok"
    except Exception as exc:
        report["espeakbridge"] = f"missing:{exc.__class__.__name__}"
        errors.append(
            "espeakbridge extension missing; run `python3 setup.py build_ext --inplace` in piper1-gpl"
        )

    try:
        import torch

        gpu_count = torch.cuda.device_count()
        gpus = []
        for idx in range(gpu_count):
            props = torch.cuda.get_device_properties(idx)
            gpus.append({"index": idx, "name": props.name, "memory_gb": round(props.total_memory / 1e9, 1)})
        report["cuda"] = {
            "available": torch.cuda.is_available(),
            "count": gpu_count,
            "gpus": gpus,
            "torch": torch.__version__,
        }
        if gpu_count < args.expect_gpus:
            errors.append(f"expected {args.expect_gpus} GPUs, found {gpu_count}")
        for gpu in gpus:
            if "A100" not in gpu["name"] and args.strict:
                errors.append(f"GPU {gpu['index']} is not A100: {gpu['name']}")
        if checkpoint.exists():
            try:
                ckpt_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
                report["checkpoint_load"] = "weights_only_ok"
                base_step = ckpt_data.get("global_step")
                report["checkpoint_global_step"] = base_step
                # A pre-v2 checkpoint loses its fit-loop state on upgrade, so
                # global_step resets to 0 even though the file says otherwise.
                # CKPT_STEP_RESETS=1 reflects that; the effective start is 0.
                if os.environ.get("CKPT_STEP_RESETS") == "1":
                    base_step = 0
                report["effective_start_step"] = base_step
                # max_steps is absolute, compared against the effective start.
                max_steps_env = os.environ.get("MAX_STEPS")
                if base_step is not None and max_steps_env:
                    if int(max_steps_env) <= int(base_step):
                        errors.append(
                            f"MAX_STEPS={max_steps_env} <= effective start global_step={base_step}; "
                            "trainer would stop immediately"
                        )
            except Exception as exc:
                report["checkpoint_load"] = f"failed:{exc.__class__.__name__}"
                errors.append("checkpoint is not weights_only-safe; run download_base_checkpoint.sh or sanitize_piper_checkpoint.py")
    except Exception as exc:
        report["cuda"] = f"failed:{exc.__class__.__name__}"
        errors.append("torch cuda probe failed")

    if csv_path.exists():
        rows = 0
        bad_columns = 0
        missing_audio = 0
        audio_errors = 0
        durations: list[float] = []
        with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.reader(csv_file, delimiter="|")
            for row in reader:
                rows += 1
                if len(row) != 2:
                    bad_columns += 1
                    continue
                if rows <= args.sample_rows:
                    audio_path = Path(row[0])
                    if not audio_path.is_absolute():
                        audio_path = audio_dir / audio_path
                    if not audio_path.exists():
                        missing_audio += 1
                        continue
                    duration = audio_duration(audio_path)
                    if duration is None:
                        audio_errors += 1
                    else:
                        durations.append(duration)
        report["metadata"] = {
            "rows": rows,
            "bad_columns": bad_columns,
            "sampled_rows": min(rows, args.sample_rows),
            "sample_missing_audio": missing_audio,
            "sample_audio_errors": audio_errors,
            "sample_min_s": round(min(durations), 3) if durations else None,
            "sample_max_s": round(max(durations), 3) if durations else None,
        }
        if rows == 0:
            errors.append("metadata has zero rows")
        if bad_columns:
            errors.append(f"metadata bad_columns={bad_columns}")
        if missing_audio:
            errors.append(f"sample missing_audio={missing_audio}")
        if audio_errors:
            errors.append(f"sample audio_errors={audio_errors}")

    free_gb = shutil.disk_usage(str(cache_dir.parent if cache_dir.parent.exists() else Path("."))).free / 1e9
    report["disk_free_gb"] = round(free_gb, 1)
    if free_gb < 50:
        warnings.append(f"cache disk free under 50GB: {free_gb:.1f}GB")

    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20, check=False)
        report["nvidia_smi_ok"] = result.returncode == 0
    except Exception:
        report["nvidia_smi_ok"] = False
        warnings.append("nvidia-smi failed")

    report["ok"] = not errors
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
