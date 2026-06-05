# ✅ Notebook đã được sửa hoàn chỉnh!

## 📝 Những thay đổi đã thực hiện:

### 1. **Thêm Cell 3A: Fix Dependencies** (sau cell Install Piper)
- Fix lỗi `ImportError: cannot import name 'is_offline_mode'`
- Upgrade `huggingface_hub`, `transformers`, `tokenizers` lên version tương thích
- Vị trí: Ngay sau cell 5 (Install Piper)

### 2. **Cập nhật Cell Configuration** (Cell 11)
- Thêm option `USE_VIVOS_KAGGLE = True`
- Thêm `VIVOS_KAGGLE_ROOT` path
- Auto-detect và validate VIVOS dataset
- Tự động chọn voice name dựa trên dataset source

### 3. **Thay thế Cell Dataset Loading** (Cell 15)
- Thêm VIVOS loader từ Kaggle dataset
- Giữ nguyên HuggingFace loader (khi `USE_VIVOS_KAGGLE=False`)
- Tự động switch giữa 2 sources dựa trên config

## 🎯 Cách sử dụng:

### Trên Kaggle:

1. **Add VIVOS Dataset**:
   - Click "Add Data" → "Datasets"
   - Search: `vivos-vietnamese-speech-corpus-for-asr`
   - Add: `kynthesis/vivos-vietnamese-speech-corpus-for-asr`

2. **Run Notebook**:
   - Cell 1-10: Setup & config (giữ nguyên)
   - Cell 11: Config → `USE_VIVOS_KAGGLE = True` ✅ (đã set sẵn)
   - Cell 12-end: Run bình thường

3. **Notebook sẽ tự động**:
   - Load VIVOS từ Kaggle dataset
   - Process audio files
   - Train với VIVOS data

## 📊 Cấu trúc Notebook mới:

```
Cell 1:  Environment Check
Cell 2:  Install System Dependencies
Cell 3:  Install Piper
Cell 4:  👈 NEW: Fix Dependencies (huggingface_hub)
Cell 5:  Patch Dataset
Cell 6:  👈 UPDATED: Configuration (VIVOS support)
Cell 7:  Download Checkpoint
Cell 8:  👈 UPDATED: Dataset Loading (VIVOS + HF)
Cell 9:  Pre-cache
Cell 10: Train
Cell 11: Export & Test
```

## ✅ Các vấn đề đã fix:

1. ✅ **ImportError: is_offline_mode** → Fixed với cell 3A
2. ✅ **VIVOS path** → Configured: `/kaggle/input/vivos-vietnamese-speech-corpus-for-asr/vivos`
3. ✅ **Dataset loader** → Support cả VIVOS và HuggingFace
4. ✅ **Auto-detection** → Validate VIVOS structure tự động

## 🚀 Ready to use!

Notebook đã sẵn sàng chạy trên Kaggle với VIVOS dataset. Chỉ cần:
1. Upload notebook lên Kaggle
2. Add VIVOS dataset
3. Run All

## 📁 File location:

`/home/minhk/pipertts/piper_vi_finetune_kaggle.ipynb`

---

## 🔧 Nếu muốn dùng HuggingFace dataset thay vì VIVOS:

Trong cell Configuration, đổi:
```python
USE_VIVOS_KAGGLE = False  # Đổi thành False
DATASET_NAME = "doof-ferb/LSVSC"  # Chọn dataset HF
```

Notebook sẽ tự động switch sang HuggingFace loader!
