"""
QUICK FIX CELL — paste vào Kaggle notebook và chạy nếu gặp:
  ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'

Root cause:
  - Kaggle preinstalls transformers mới → đòi huggingface_hub>=1.0
  - Notebook pin huggingface_hub<1 (bắt buộc, vì datasets<4 yêu cầu hub<1)
  - lightning → torchmetrics thấy transformers>=4.4 → import bert_score → crash

Fix: GỠ transformers (Piper train không cần). torchmetrics sẽ skip bert_score
nhờ guard _TRANSFORMERS_GREATER_EQUAL_4_4. KHÔNG reinstall transformers.
"""

import importlib
import importlib.util
import pathlib
import shutil
import subprocess
import sys

# 1. pip uninstall (show output — không nuốt lỗi)
r = subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "transformers"],
    capture_output=True, text=True,
)
print((r.stdout or r.stderr).strip()[-300:])

# 2. Fallback: pip fail silent → xóa thẳng package dir
spec = importlib.util.find_spec("transformers")
if spec is not None and spec.origin:
    pkg_dir = pathlib.Path(spec.origin).parent
    print(f"⚠️  pip uninstall không gỡ được → rm -rf {pkg_dir}")
    shutil.rmtree(pkg_dir, ignore_errors=True)
    for di in pkg_dir.parent.glob("transformers-*.dist-info"):
        shutil.rmtree(di, ignore_errors=True)

# 3. Purge module cache cả chain để guard _TRANSFORMERS_GREATER_EQUAL_4_4 re-evaluate
for m in [k for k in sys.modules
          if k.split(".")[0] in ("lightning", "torchmetrics", "transformers",
                                 "lightning_utilities", "pytorch_lightning")]:
    del sys.modules[m]
importlib.invalidate_caches()

# 4. Verify
assert importlib.util.find_spec("transformers") is None, \
    "transformers vẫn còn — Restart kernel (Run → Restart session) rồi chạy lại cell này"

import huggingface_hub
import lightning
import torchmetrics

print(f"✅ huggingface_hub: {huggingface_hub.__version__} (phải <1.0)")
print(f"✅ lightning:       {lightning.__version__}")
print(f"✅ torchmetrics:    {torchmetrics.__version__}")
print("✅ Fixed — chạy tiếp cell Pre-cache / Train")
