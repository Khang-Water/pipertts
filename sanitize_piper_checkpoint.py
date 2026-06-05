#!/usr/bin/env python3
"""Make an old trusted Piper checkpoint safer for modern Lightning/PyTorch.

Only run this on checkpoints from a trusted source. It uses weights_only=False
once, removes legacy/unneeded hparams, then verifies weights_only=True loading.
"""

from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
from typing import Any


DROP_HPARAMS = {"dataset", "vocoder_warmstart_ckpt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--piper-dir", type=Path, default=Path("piper1-gpl"))
    parser.add_argument("--in-place", action="store_true")
    return parser.parse_args()


def model_init_args(piper_dir: Path) -> set[str]:
    lightning_path = piper_dir / "src" / "piper" / "train" / "vits" / "lightning.py"
    tree = ast.parse(lightning_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "VitsModel":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    return {arg.arg for arg in item.args.args if arg.arg != "self"}
    raise RuntimeError(f"could not find VitsModel.__init__ in {lightning_path}")


def to_safe_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return tuple(to_safe_value(item) for item in value)
    if isinstance(value, list):
        return [to_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_safe_value(item) for key, item in value.items()}
    if hasattr(value, "ndim") and getattr(value, "ndim") == 0 and hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist") and callable(value.tolist):
        return value.tolist()
    return str(value)


def main() -> int:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    output_path = checkpoint_path if args.in_place else (args.output or checkpoint_path.with_suffix(".sanitized.ckpt"))
    allowed_hparams = model_init_args(args.piper_dir) - DROP_HPARAMS

    import torch

    try:
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if output_path != checkpoint_path:
            output_path.write_bytes(checkpoint_path.read_bytes())
        print(f"checkpoint already weights_only-safe: {output_path}")
        return 0
    except Exception as exc:
        print(f"sanitizing trusted checkpoint after weights_only failure: {exc.__class__.__name__}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = checkpoint.get("hyper_parameters") or {}
    sanitized_hparams = {
        key: to_safe_value(value)
        for key, value in hparams.items()
        if key in allowed_hparams
    }
    dropped = sorted(set(hparams) - set(sanitized_hparams))
    checkpoint["hyper_parameters"] = sanitized_hparams

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, output_path)
    torch.load(output_path, map_location="cpu", weights_only=True)
    print(f"sanitized checkpoint: {output_path}")
    if dropped:
        print(f"dropped hparams: {', '.join(dropped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

