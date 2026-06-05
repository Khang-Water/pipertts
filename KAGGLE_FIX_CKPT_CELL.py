"""
QUICK FIX CELL — paste vào Kaggle và chạy nếu train fail với:
  _pickle.UnpicklingError: Weights only load failed ... GLOBAL pathlib.PosixPath

Root cause (2 tầng):
  1. PyTorch 2.6+ đổi default torch.load → weights_only=True. Lightning CLI
     (_parse_ckpt_path) + resume load hard-code weights_only=True. Ckpt cũ từ
     rhasspy/piper-checkpoints (old piper + PL 1.x) chứa pathlib.PosixPath
     trong hyper_parameters → UnpicklingError.
  2. Kể cả fix (1): hparams cũ (sample_bytes, channels, num_workers, seed, ...)
     không có trong VitsModel của piper1-gpl → jsonargparse
     "Group 'model' does not accept option ..." → SystemExit.

Fix: load 1 lần weights_only=False (nguồn tin cậy: rhasspy official trên HF),
deep-convert về plain types, giữ CHỈ arch hparams, re-save in-place.

Sau khi chạy cell này → chạy lại cell Train.
"""

import os
import pathlib
import time

import torch

BASE_CKPT_LOCAL = "/kaggle/working/base_vi_VN.ckpt"  # khớp config notebook
dst = pathlib.Path(BASE_CKPT_LOCAL)
assert dst.exists(), f"Không thấy {dst} — chạy cell download ckpt trước"

# Arch hparams của piper1-gpl VitsModel (commit 2a60c2b). KHÔNG gồm:
# - batch_size, num_symbols: link targets (jsonargparse lấy từ data.*)
# - learning_rate/betas/c_mel/...: bỏ để CLI knobs của notebook có hiệu lực
#   (hparams trong ckpt OVERRIDE CLI args khi Lightning parse ckpt_path)
# - dataset: unused, chứa PosixPath
_ARCH_HPARAMS = {
    "sample_rate", "num_speakers",
    "resblock", "resblock_kernel_sizes", "resblock_dilation_sizes",
    "upsample_rates", "upsample_initial_channel", "upsample_kernel_sizes",
    "filter_length", "hop_length", "win_length",
    "mel_channels", "mel_fmin", "mel_fmax",
    "inter_channels", "hidden_channels", "filter_channels",
    "n_heads", "n_layers", "kernel_size", "p_dropout", "n_layers_q",
    "use_spectral_norm", "gin_channels", "use_sdp", "segment_size",
}


def _to_plain(o):
    """Convert mọi object về types mà torch.load(weights_only=True) chấp nhận."""
    if o is None or isinstance(o, torch.Tensor):
        return o
    if type(o) in (str, int, float, bool, bytes, complex):  # exact types (np.float64 là float subclass!)
        return o
    if isinstance(o, pathlib.PurePath):
        return str(o)
    if isinstance(o, dict):  # gồm AttributeDict / OrderedDict
        return {_to_plain(k): _to_plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        t = type(o) if type(o) in (list, tuple, set) else list
        return t(_to_plain(x) for x in o)
    for base in (bool, int, float, complex, str, bytes):  # scalar subclasses: np.float64, IntEnum, ...
        if isinstance(o, base):
            return base(o)
    if hasattr(o, "item") and callable(o.item) and getattr(o, "ndim", None) == 0:
        return o.item()   # 0-d numpy scalar (np.int64, ...)
    if hasattr(o, "tolist") and callable(o.tolist):
        return o.tolist()  # numpy array
    return str(o)          # object lạ → string


try:
    torch.load(dst, weights_only=True, map_location="cpu")
    print("✅ ckpt đã weights_only-safe — không cần sanitize")
except Exception as _e:
    print(f"ℹ️  Sanitizing ckpt ({type(_e).__name__}) ...")
    t0 = time.time()
    ckpt = torch.load(dst, map_location="cpu", weights_only=False)  # trusted: rhasspy official
    ckpt = _to_plain(ckpt)
    hp = ckpt.get("hyper_parameters") or {}
    dropped = sorted(set(hp) - _ARCH_HPARAMS)
    ckpt["hyper_parameters"] = {k: v for k, v in hp.items() if k in _ARCH_HPARAMS}
    if dropped:
        print(f"   dropped hparams: {dropped}")
    _tmp = str(dst) + ".tmp"
    torch.save(ckpt, _tmp)
    os.replace(_tmp, dst)
    torch.load(dst, weights_only=True, map_location="cpu")  # verify
    print(f"✅ Sanitized + verified in {time.time()-t0:.1f}s ({dst.stat().st_size/1e6:.1f} MB)")
    print("→ Chạy lại cell Train")
