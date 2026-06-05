# Quick Fix Instructions for Kaggle Notebook

## 🔧 Cách fix lỗi ImportError trong notebook hiện tại:

### Bước 1: Thêm cell mới
Trong notebook Kaggle, thêm một cell mới **NGAY SAU** cell 3 (Install Piper) và **TRƯỚC** cell 7 (Pre-cache).

### Bước 2: Copy code này vào cell mới:

```python
# FIX: huggingface_hub dependency conflict
import subprocess
import sys

print("Fixing dependencies...")

# Uninstall conflicting packages
subprocess.run([
    sys.executable, "-m", "pip", "uninstall", "-y", 
    "transformers", "huggingface_hub", "tokenizers"
], capture_output=True, check=False)

# Install compatible versions
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "huggingface_hub>=0.24.0",
    "tokenizers>=0.19.0", 
    "transformers>=4.40.0",
], check=True)

print("✅ Fixed! Continue to next cell.")

# Verify
import huggingface_hub
import transformers
print(f"huggingface_hub: {huggingface_hub.__version__}")
print(f"transformers: {transformers.__version__}")
```

### Bước 3: Chạy lại
1. Chạy cell fix này
2. Tiếp tục chạy cell 7 (Pre-cache)

---

## 🔄 Alternative: Restart & Run All

Nếu vẫn lỗi, thử cách này:

1. **Runtime** → **Restart session**
2. Thêm cell fix vào vị trí đúng (sau cell 3)
3. **Run All** từ đầu

---

## 📝 Vị trí cell trong notebook:

```
Cell 1: Environment Check
Cell 2: Install System Dependencies  
Cell 3: Install Piper & Python Dependencies
Cell 3A: 👈 THÊM CELL FIX Ở ĐÂY
Cell 4: Patch Dataset
Cell 5: Configuration
Cell 6: Download Checkpoint
Cell 7: Load VIVOS Dataset
Cell 8: Pre-cache
Cell 9: Train
Cell 10: Export & Test
```

---

## ❓ Tại sao lỗi này xảy ra?

- Kaggle có sẵn `huggingface_hub` version cũ
- `lightning` → `torchmetrics` → `transformers` cần version mới
- Version cũ không có function `is_offline_mode`
- Cần upgrade trước khi import `piper.train.vits.dataset`

---

## ✅ Sau khi fix:

Cell 7 (Pre-cache) sẽ chạy thành công:
```python
from piper.train.vits.dataset import VitsDataModule  # ✅ No error
```
