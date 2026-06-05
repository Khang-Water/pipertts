# 🇻🇳 PiperTTS VIVOS Finetune - Complete Pipeline

Notebook hoàn chỉnh để finetune PiperTTS với VIVOS dataset trên Kaggle.

## 📋 Setup Instructions

### 1. Kaggle Settings
- **Accelerator**: GPU T4 x2 (hoặc P100)
- **Internet**: ON
- **Persistence**: Files only

### 2. Add VIVOS Dataset
1. Click "Add Data" → "Datasets"
2. Search: `vivos-vietnamese-speech-corpus-for-asr`
3. Add dataset: `kynthesis/vivos-vietnamese-speech-corpus-for-asr`

### 3. Run All Cells
Chạy tuần tự từ cell 1 → 10

---

## Cell 1: Environment Check
```python
import os, sys, subprocess, shutil, multiprocessing
print("Python:", sys.version.split()[0])
print("CPU cores:", multiprocessing.cpu_count())
try:
    import torch
    print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}:", torch.cuda.get_device_name(i),
              f"({torch.cuda.get_device_properties(i).total_memory/1e9:.1f} GB)")
except Exception as e:
    print("Torch chưa cài, sẽ cài ở cell sau:", e)

!free -h | head -2
!df -h /kaggle/working
```

---

## Cell 2: Install System Dependencies
```bash
%%bash
set -euo pipefail
apt-get -qq update
apt-get -qq install -y build-essential cmake ninja-build espeak-ng libsndfile1 sox > /dev/null
echo "✅ System packages installed"
espeak-ng --voices=vi || (echo "❌ espeak-ng không có voice vi!"; exit 1)
```

---

## Cell 3: Install Piper & Python Dependencies
```bash
%%bash
set -euo pipefail
cd /kaggle/working

# Clone Piper
PIPER_COMMIT="2a60c2bd152356b613673ecaa0e15cbd3c0e502c"
if [ ! -d piper1-gpl ]; then
    git clone https://github.com/OHF-Voice/piper1-gpl.git
    cd piper1-gpl && git checkout "$PIPER_COMMIT"
else
    cd piper1-gpl
    git checkout "$PIPER_COMMIT" 2>/dev/null || echo "⚠️  Using existing commit"
fi

# Install Python dependencies
pip install -q --upgrade pip
pip install -q \
    "lightning>=2,<3" "tensorboard>=2,<3" "tensorboardX>=2,<3" \
    "jsonargparse[signatures]>=4.27.7" "onnx>=1,<2" \
    "pysilero-vad>=2.1,<3" "cython>=3,<4" "librosa<1" \
    "scikit-build<1" "pathvalidate>=3,<4" "onnxruntime>=1,<2" \
    "soundfile>=0.12,<1"

# Install Piper
pip install -q -e .

# Build Cython extension
bash build_monotonic_align.sh

# Verify
python -c "import piper; print('piper:', piper.__file__)"
python -c "from piper.train.vits.monotonic_align import maximum_path; print('monotonic_align OK')"
echo "✅ Piper installed successfully"
```

---

## Cell 4: Patch Dataset for Performance
```python
from pathlib import Path

ds = Path("/kaggle/working/piper1-gpl/src/piper/train/vits/dataset.py")
src = ds.read_text()

pattern = "            num_workers=self.num_workers,\n        )"
replacement = (
    "            num_workers=self.num_workers,\n"
    "            pin_memory=True,\n"
    "            persistent_workers=(self.num_workers > 0),\n"
    "        )"
)

if "pin_memory=True" not in src:
    n = src.count(pattern)
    assert n == 3, f"Expected 3 DataLoaders, found {n}"
    new_src = src.replace(pattern, replacement)
    compile(new_src, str(ds), "exec")
    ds.write_text(new_src)
    print(f"✅ Patched {n} DataLoaders with pin_memory + persistent_workers")
else:
    print("ℹ️  Already patched")

import subprocess
print(subprocess.check_output(["grep", "-n", "pin_memory", str(ds)], text=True))
```

---

## Cell 5: Configuration
```python
# =============================================================================
# 🎛️  CONFIGURATION - Chỉnh ở đây
# =============================================================================

# --- VIVOS Kaggle Dataset ---
USE_VIVOS_KAGGLE = True
VIVOS_KAGGLE_ROOT = "/kaggle/input/vivos-vietnamese-speech-corpus-for-asr/vivos"
VIVOS_SPLIT = "train"  # "train" hoặc "test"

# --- Training Settings ---
MAX_SAMPLES = 3000          # Số mẫu training (VIVOS train có ~11k)
MIN_DURATION_S = 1.0
MAX_DURATION_S = 12.0
REBUILD_DATA = False

VOICE_NAME = "vi_VN-vivos-medium"
ESPEAK_VOICE = "vi"
SAMPLE_RATE = 22050

BATCH_SIZE = 24             # Giảm xuống 16 hoặc 8 nếu OOM
NUM_WORKERS = 4
ADDITIONAL_STEPS = 8000     # Số step train thêm
VAL_CHECK_INTERVAL = 1000
PRECISION = "16-mixed"      # "bf16-mixed" cho A100/H100
LEARNING_RATE = 2e-4
NUM_DEVICES = None          # None = auto-detect
STRATEGY = "ddp_find_unused_parameters_true"

# --- Paths ---
WORK_DIR = "/kaggle/working"
DATA_DIR = f"{WORK_DIR}/data"
AUDIO_DIR = f"{DATA_DIR}/wavs"
CSV_PATH = f"{DATA_DIR}/metadata.csv"
CACHE_DIR = f"{DATA_DIR}/cache"
CKPT_DIR = f"{WORK_DIR}/checkpoints"
CONFIG_PATH = f"{CKPT_DIR}/{VOICE_NAME}.json"
BASE_CKPT_LOCAL = f"{WORK_DIR}/base_vi_VN.ckpt"

# Create directories
import os, shutil
for d in (DATA_DIR, AUDIO_DIR, CACHE_DIR, CKPT_DIR):
    os.makedirs(d, exist_ok=True)

# Validate VIVOS
from pathlib import Path
if USE_VIVOS_KAGGLE:
    vivos_root = Path(VIVOS_KAGGLE_ROOT)
    if not (vivos_root / VIVOS_SPLIT / "waves").exists():
        raise FileNotFoundError(
            f"VIVOS not found at {VIVOS_KAGGLE_ROOT}\n"
            f"Please add dataset: kynthesis/vivos-vietnamese-speech-corpus-for-asr"
        )
    print(f"✅ Found VIVOS at: {vivos_root}")

# Disk check
free_gb = shutil.disk_usage(WORK_DIR).free / 1e9
print(f"Free disk: {free_gb:.1f} GB")
assert free_gb > 12, f"Need >12 GB free, only {free_gb:.1f} GB available"

print(f"\nConfiguration:")
print(f"  Dataset: VIVOS Kaggle ({VIVOS_KAGGLE_ROOT})")
print(f"  Max samples: {MAX_SAMPLES}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Additional steps: {ADDITIONAL_STEPS}")
print(f"  Precision: {PRECISION}")
```

---

## Cell 6: Download Base Checkpoint
```python
import os, time, pathlib
from huggingface_hub import hf_hub_download

dst = pathlib.Path(BASE_CKPT_LOCAL)
if dst.exists() and dst.stat().st_size > 800_000_000:
    print(f"✅ Checkpoint exists: {dst} ({dst.stat().st_size/1e6:.1f} MB)")
else:
    print("Downloading Vietnamese base checkpoint from HuggingFace...")
    t0 = time.time()
    p = hf_hub_download(
        repo_id="rhasspy/piper-checkpoints",
        repo_type="dataset",
        filename="vi/vi_VN/vais1000/medium/epoch=4769-step=919580.ckpt",
        local_dir=WORK_DIR,
    )
    if pathlib.Path(p) != dst:
        os.replace(p, dst)
    elapsed = time.time() - t0
    print(f"✅ Downloaded {dst} ({dst.stat().st_size/1e6:.1f} MB) in {elapsed:.1f}s")
```

---

## Cell 7: Load VIVOS Dataset
```python
import os, csv, time, shutil
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa

print("=" * 70)
print("🇻🇳 Loading VIVOS from Kaggle Dataset")
print("=" * 70)

t0 = time.time()

# Prepare directories
if REBUILD_DATA:
    print("REBUILD_DATA=True → removing old data")
    shutil.rmtree(AUDIO_DIR, ignore_errors=True)
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

vivos_root = Path(VIVOS_KAGGLE_ROOT)
split_dir = vivos_root / VIVOS_SPLIT
waves_dir = split_dir / "waves"
prompts_path = split_dir / "prompts.txt"

# Read prompts
print(f"\nReading prompts from {prompts_path}...")
prompts = {}
with open(prompts_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        utt_id, text = parts
        prompts[utt_id] = text

print(f"Found {len(prompts)} prompts")

# Process audio
print(f"\nProcessing audio (max {MAX_SAMPLES} samples)...")
n_written = 0
n_skip_missing = 0
n_skip_duration = 0
n_skip_error = 0
csv_rows = []

for utt_id, text in prompts.items():
    if n_written >= MAX_SAMPLES:
        break

    wav_matches = list(waves_dir.glob(f"**/{utt_id}.wav"))
    if not wav_matches:
        n_skip_missing += 1
        continue

    wav_path = wav_matches[0]

    try:
        audio, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
        duration = len(audio) / sr
        
        if duration < MIN_DURATION_S or duration > MAX_DURATION_S:
            n_skip_duration += 1
            continue

        wav_name = f"vi_{n_written:06d}.wav"
        output_path = Path(AUDIO_DIR) / wav_name
        sf.write(str(output_path), audio, sr, subtype="PCM_16")

        csv_rows.append((wav_name, text))
        n_written += 1

        if n_written % 500 == 0:
            print(f"  {n_written}/{MAX_SAMPLES} (skip: {n_skip_missing+n_skip_duration+n_skip_error}) "
                  f"elapsed {time.time()-t0:.1f}s")

    except Exception as e:
        n_skip_error += 1
        if n_skip_error <= 3:
            print(f"  Error: {e}")
        continue

# Write CSV
with open(CSV_PATH, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter='|', quoting=csv.QUOTE_NONE, escapechar='\\')
    for wav_name, text in csv_rows:
        clean_text = text.replace('|', ' ').replace('\n', ' ').replace('\r', ' ').strip()
        writer.writerow([wav_name, clean_text])

print(f"\n✅ Dataset prepared in {time.time()-t0:.1f}s")
print(f"   Written: {n_written}")
print(f"   Skipped: {n_skip_missing+n_skip_duration+n_skip_error}")
print(f"   CSV: {CSV_PATH}")
print(f"   Audio: {AUDIO_DIR}")

!head -3 "$CSV_PATH"
!ls "$AUDIO_DIR" | wc -l
!du -sh "$AUDIO_DIR"
```

---

## Cell 8: Pre-cache Phonemes
```python
import sys, time
sys.path.insert(0, "/kaggle/working/piper1-gpl/src")

# Reset module cache
for m in [k for k in sys.modules if k.startswith("piper")]:
    del sys.modules[m]

from piper.train.vits.dataset import VitsDataModule

print("Pre-caching phonemes and spectrograms...")
t0 = time.time()

dm = VitsDataModule(
    csv_path=CSV_PATH,
    audio_dir=AUDIO_DIR,
    cache_dir=CACHE_DIR,
    espeak_voice=ESPEAK_VOICE,
    config_path=CONFIG_PATH,
    voice_name=VOICE_NAME,
    sample_rate=SAMPLE_RATE,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)
dm.prepare_data()

print(f"✅ Pre-cache done in {time.time()-t0:.1f}s")
!ls "$CACHE_DIR" | head -5
!echo -n "Cache files: "; ls "$CACHE_DIR" | wc -l
```

---

## Cell 9: Train
```python
import torch, os, shlex, subprocess

# Read base checkpoint global_step
print("Reading base checkpoint...")
_meta = torch.load(BASE_CKPT_LOCAL, map_location="cpu", weights_only=False)
BASE_GLOBAL_STEP = int(_meta.get("global_step", 0))
del _meta

target_max_steps = BASE_GLOBAL_STEP + ADDITIONAL_STEPS
print(f"Base global_step: {BASE_GLOBAL_STEP}")
print(f"Additional steps: {ADDITIONAL_STEPS}")
print(f"Target max_steps: {target_max_steps}")

n_gpu = torch.cuda.device_count()
devices = NUM_DEVICES or max(1, n_gpu)
print(f"Using {devices} GPU(s)")

# Environment
env = os.environ.copy()
env["TOKENIZERS_PARALLELISM"] = "false"
env["PYTHONUNBUFFERED"] = "1"
env["TF_CPP_MIN_LOG_LEVEL"] = "2"
env["TORCH_CUDNN_V8_API_ENABLED"] = "1"

strategy_arg = STRATEGY if devices > 1 else "auto"

# Build command
cmd = [
    "python", "-m", "piper.train", "fit",
    f"--data.csv_path={CSV_PATH}",
    f"--data.audio_dir={AUDIO_DIR}",
    f"--data.cache_dir={CACHE_DIR}",
    f"--data.config_path={CONFIG_PATH}",
    f"--data.voice_name={VOICE_NAME}",
    f"--data.espeak_voice={ESPEAK_VOICE}",
    f"--data.batch_size={BATCH_SIZE}",
    f"--data.num_workers={NUM_WORKERS}",
    f"--data.validation_split=0.02",
    f"--data.num_test_examples=2",
    f"--model.sample_rate={SAMPLE_RATE}",
    f"--model.learning_rate={LEARNING_RATE}",
    f"--trainer.max_steps={target_max_steps}",
    f"--trainer.precision={PRECISION}",
    f"--trainer.accelerator=gpu",
    f"--trainer.devices={devices}",
    f"--trainer.strategy={strategy_arg}",
    f"--trainer.benchmark=true",
    f"--trainer.val_check_interval={VAL_CHECK_INTERVAL}",
    f"--trainer.limit_val_batches=4",
    f"--trainer.log_every_n_steps=25",
    f"--trainer.default_root_dir={CKPT_DIR}",
    f"--trainer.enable_progress_bar=true",
    f"--ckpt_path={BASE_CKPT_LOCAL}",
]

print("\n🏋️ Starting training...")
print("Command:", " ".join(cmd[:5]), "...")

ret = subprocess.run(cmd, cwd="/kaggle/working/piper1-gpl", env=env)
print(f"\nTrain exit code: {ret.returncode}")

# Fallback to 1 GPU if DDP fails
if ret.returncode != 0 and devices > 1:
    print(f"\n⚠️  Retrying with 1 GPU...")
    cmd_1gpu = [c.replace(f"--trainer.devices={devices}", "--trainer.devices=1")
                 .replace(f"--trainer.strategy={strategy_arg}", "--trainer.strategy=auto")
                 for c in cmd]
    ret = subprocess.run(cmd_1gpu, cwd="/kaggle/working/piper1-gpl", env=env)
    print(f"Retry exit code: {ret.returncode}")

assert ret.returncode == 0, "Training failed!"
print("\n✅ Training complete!")
```

---

## Cell 10: Export ONNX & Test
```python
import glob, os, subprocess
import numpy as np
import soundfile as sf
import IPython.display as ipd

# Find latest checkpoint
patterns = [f"{CKPT_DIR}/**/checkpoints/*.ckpt", f"{CKPT_DIR}/**/*.ckpt"]
ckpts = set()
for pat in patterns:
    ckpts.update(glob.glob(pat, recursive=True))
ckpts = sorted(ckpts, key=os.path.getmtime)
ckpts = [c for c in ckpts if os.path.abspath(c) != os.path.abspath(BASE_CKPT_LOCAL)]
assert ckpts, "No checkpoint found!"

latest = ckpts[-1]
print(f"Latest checkpoint: {latest} ({os.path.getsize(latest)/1e6:.1f} MB)")

# Export ONNX
onnx_out = f"{CKPT_DIR}/{VOICE_NAME}.onnx"
print(f"\nExporting to ONNX: {onnx_out}")

ret = subprocess.run([
    "python", "-m", "piper.train.export_onnx",
    "--checkpoint", latest,
    "--output-file", onnx_out,
], cwd="/kaggle/working/piper1-gpl")

assert ret.returncode == 0, "ONNX export failed!"

import shutil
shutil.copy(CONFIG_PATH, onnx_out + ".json")

print(f"✅ Exported:")
print(f"   {onnx_out} ({os.path.getsize(onnx_out)/1e6:.1f} MB)")
print(f"   {onnx_out}.json")

# Test inference
print("\n🎤 Testing inference...")
from piper import PiperVoice

voice = PiperVoice.load(onnx_out, config_path=onnx_out + ".json")
text = "Xin chào, đây là giọng nói tiếng Việt được tinh chỉnh với VIVOS dataset trên Kaggle."

audio_chunks = []
for chunk in voice.synthesize(text):
    arr = getattr(chunk, "audio_float_array", None)
    if arr is None:
        arr = chunk.audio_int16_array.astype(np.float32) / 32768.0
    audio_chunks.append(arr)

audio = np.concatenate(audio_chunks)
sr = voice.config.sample_rate

out_wav = f"{WORK_DIR}/demo_vivos.wav"
sf.write(out_wav, audio, sr, subtype="PCM_16")
print(f"✅ Saved: {out_wav} (duration={len(audio)/sr:.2f}s)")

ipd.display(ipd.Audio(audio, rate=sr))

print("\n" + "=" * 70)
print("🎉 FINETUNE COMPLETE!")
print("=" * 70)
print(f"Download files from /kaggle/working/checkpoints/")
```

---

## 📦 Output Files

Sau khi chạy xong:
- `checkpoints/vi_VN-vivos-medium.onnx` - Model ONNX
- `checkpoints/vi_VN-vivos-medium.onnx.json` - Config
- `demo_vivos.wav` - Audio demo
- `checkpoints/lightning_logs/` - Training logs

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM | Giảm `BATCH_SIZE` xuống 16 hoặc 8 |
| VIVOS not found | Add dataset `kynthesis/vivos-vietnamese-speech-corpus-for-asr` |
| Training slow | Giảm `MAX_SAMPLES` xuống 2000 |
| DDP error | Set `NUM_DEVICES=1` |
