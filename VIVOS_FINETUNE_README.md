# PiperTTS VIVOS Finetuning Guide

Hướng dẫn finetune PiperTTS với dataset VIVOS tiếng Việt trên máy local.

## 📋 Yêu cầu

- Python 3.8+
- CUDA-capable GPU (khuyến nghị: 8GB+ VRAM)
- Dataset VIVOS (download từ https://ailab.hcmus.edu.vn/vivos)
- ~10GB disk space

## 🚀 Quick Start

### Bước 1: Tải VIVOS dataset

```bash
# Download VIVOS từ nguồn chính thức
# https://ailab.hcmus.edu.vn/vivos

# Giải nén và đảm bảo cấu trúc:
# vivos/
# ├── train/
# │   ├── waves/
# │   │   ├── VIVOSDEV01/
# │   │   │   ├── VIVOSDEV01_R001.wav
# │   │   │   └── ...
# │   │   └── ...
# │   └── prompts.txt
# └── test/
#     └── ...
```

### Bước 2: Chuẩn bị metadata

```bash
cd /home/minhk/pipertts

# Chỉnh VIVOS_ROOT trong script
python3 prepare_vivos_for_piper.py \
    --vivos_root /path/to/vivos \
    --output_csv ./output/vivos_train.csv \
    --max_samples 3000 \
    --split train
```

### Bước 3: Finetune

```bash
# Chỉnh VIVOS_ROOT trong script
export VIVOS_ROOT=/path/to/vivos
bash finetune_vivos.sh
```

## 📝 Chi tiết các file

### 1. `prepare_vivos_for_piper.py`

Convert VIVOS format sang PiperTTS CSV format:

```python
# VIVOS format (prompts.txt):
VIVOSDEV01_R001 Đây là câu văn mẫu

# PiperTTS CSV format:
/path/to/VIVOSDEV01_R001.wav|Đây là câu văn mẫu
```

**Tham số:**
- `--vivos_root`: Đường dẫn đến thư mục VIVOS
- `--output_csv`: File CSV output
- `--max_samples`: Số lượng mẫu tối đa (mặc định: all)
- `--split`: train hoặc test

### 2. `finetune_vivos.sh`

Script chạy training với các tham số tối ưu:

**Cấu hình quan trọng:**
```bash
VIVOS_ROOT="/path/to/vivos"          # Đường dẫn VIVOS
VOICE_NAME="vi_VN-vivos-medium"      # Tên voice
BATCH_SIZE=32                         # Batch size (giảm nếu OOM)
CHECKPOINT=""                         # Optional: pretrained checkpoint
```

### 3. `piper_vivos_local.py`

Module Python để load VIVOS từ local filesystem (thay vì HuggingFace).

## 🎛️ Cấu hình Training

### Tham số cơ bản

```bash
# Trong finetune_vivos.sh
BATCH_SIZE=32              # Giảm xuống 16 hoặc 8 nếu GPU OOM
MAX_SAMPLES=3000           # Số mẫu training (VIVOS train có ~11k)
SAMPLE_RATE=22050          # Sample rate (khớp với Piper medium checkpoint)
```

### Sử dụng pretrained checkpoint (khuyến nghị)

```bash
# Download checkpoint tiếng Việt từ HuggingFace
wget https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/vi/vi_VN/vais1000/medium/epoch=4769-step=919580.ckpt

# Thêm vào finetune_vivos.sh
CHECKPOINT="./epoch=4769-step=919580.ckpt"
```

Sử dụng checkpoint sẽ:
- Tăng tốc training 5-10x
- Cải thiện chất lượng giọng nói
- Cần ít data hơn để converge

## 📊 Training trên các GPU khác nhau

| GPU | VRAM | Batch Size | Training Time (3000 samples) |
|-----|------|------------|------------------------------|
| RTX 3060 | 12GB | 24-32 | ~2-3 hours |
| RTX 3070 | 8GB | 16-24 | ~3-4 hours |
| RTX 3080 | 10GB | 24-32 | ~2 hours |
| T4 | 16GB | 24-32 | ~3-4 hours |
| V100 | 16GB | 32-48 | ~1-2 hours |

**Nếu gặp OOM:**
```bash
# Giảm batch size
BATCH_SIZE=16  # hoặc 8

# Hoặc giảm số samples
MAX_SAMPLES=2000
```

## 🎯 Export ONNX

Sau khi training xong:

```bash
cd /home/minhk/pipertts/piper1-gpl

python3 -m piper.train.export_onnx \
    --checkpoint ./output/checkpoints/last.ckpt \
    --output-file ./output/vi_VN-vivos-medium.onnx
```

Output files:
- `vi_VN-vivos-medium.onnx` - Model ONNX
- `vi_VN-vivos-medium.onnx.json` - Config file

## 🧪 Test inference

```python
from piper import PiperVoice
import numpy as np
import soundfile as sf

# Load model
voice = PiperVoice.load(
    "vi_VN-vivos-medium.onnx",
    config_path="vi_VN-vivos-medium.onnx.json"
)

# Synthesize
text = "Xin chào, đây là giọng nói tiếng Việt."
audio_chunks = []
for chunk in voice.synthesize(text):
    audio_chunks.append(chunk.audio_int16_array.astype(np.float32) / 32768.0)

audio = np.concatenate(audio_chunks)
sf.write("output.wav", audio, voice.config.sample_rate)
```

## 🐛 Troubleshooting

### 1. `espeak-ng: voice 'vi' not found`

```bash
# Cài espeak-ng
sudo apt-get install espeak-ng

# Verify
espeak-ng --voices=vi
```

### 2. `CUDA out of memory`

```bash
# Giảm batch size
BATCH_SIZE=8

# Hoặc giảm số samples
MAX_SAMPLES=1500
```

### 3. `monotonic_align` import error

```bash
cd piper1-gpl
bash build_monotonic_align.sh
```

### 4. Training quá chậm

- Sử dụng pretrained checkpoint (tăng tốc 5-10x)
- Giảm `MAX_SAMPLES` xuống 2000-3000
- Tăng `NUM_WORKERS` (mặc định: 4)

### 5. VIVOS dataset không tìm thấy

```bash
# Kiểm tra cấu trúc
ls -la /path/to/vivos/train/
# Phải có: waves/ và prompts.txt

# Kiểm tra prompts.txt format
head -3 /path/to/vivos/train/prompts.txt
# Format: VIVOSDEV01_R001 Text here
```

## 📚 Tham khảo

- [Piper Training Docs](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/TRAINING.md)
- [VIVOS Dataset](https://ailab.hcmus.edu.vn/vivos)
- [Piper Checkpoints](https://huggingface.co/datasets/rhasspy/piper-checkpoints)

## 💡 Tips

1. **Sử dụng pretrained checkpoint** - Quan trọng nhất để tăng tốc và cải thiện chất lượng
2. **Start với 2000-3000 samples** - Đủ để finetune khi có checkpoint
3. **Monitor TensorBoard** - `tensorboard --logdir ./output/checkpoints/lightning_logs`
4. **Validate trên test set** - Dùng VIVOS test split để đánh giá
5. **Experiment với learning rate** - Mặc định 2e-4, có thể thử 1e-4 hoặc 5e-4

## 🔄 Workflow đầy đủ

```bash
# 1. Chuẩn bị môi trường
cd /home/minhk/pipertts
python3 -m venv .venv
source .venv/bin/activate

# 2. Cài dependencies
cd piper1-gpl
pip install -e '.[train]'
bash build_monotonic_align.sh

# 3. Download checkpoint (optional nhưng khuyến nghị)
wget https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/vi/vi_VN/vais1000/medium/epoch=4769-step=919580.ckpt

# 4. Chuẩn bị data
python3 ../prepare_vivos_for_piper.py \
    --vivos_root /path/to/vivos \
    --output_csv ./output/metadata.csv \
    --max_samples 3000

# 5. Train
export VIVOS_ROOT=/path/to/vivos
export CHECKPOINT=./epoch=4769-step=919580.ckpt
bash ../finetune_vivos.sh

# 6. Export ONNX
python3 -m piper.train.export_onnx \
    --checkpoint ./output/checkpoints/lightning_logs/version_0/checkpoints/last.ckpt \
    --output-file ./output/vi_VN-vivos-medium.onnx

# 7. Test
python3 -c "
from piper import PiperVoice
voice = PiperVoice.load('output/vi_VN-vivos-medium.onnx')
for chunk in voice.synthesize('Xin chào'):
    pass
print('✅ Model works!')
"
```
