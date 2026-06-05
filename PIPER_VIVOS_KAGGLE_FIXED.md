# 🇻🇳 PiperTTS VIVOS Finetune - FIXED VERSION

Complete pipeline với dependency fixes.

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

## Cell 2A: Fix Python Dependencies (IMPORTANT!)
```python
# Fix dependency conflicts BEFORE installing Piper
import subprocess
import sys

print("Fixing dependency version conflicts...")

# Uninstall potentially conflicting packages
subprocess.run([
    sys.executable, "-m", "pip", "uninstall", "-y", 
    "transformers", "huggingface_hub", "tokenizers"
], capture_output=True)

# Install compatible versions in correct order
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "huggingface_hub>=0.24,<1",
    "tokenizers>=0.19,<1",
    "transformers>=4.40,<5",
], check=True)

print("✅ Dependencies fixed")

# Verify
try:
    import huggingface_hub
    import transformers
    print(f"huggingface_hub: {huggingface_hub.__version__}")
    print(f"transformers: {transformers.__version__}")
except ImportError as e:
    print(f"⚠️  Import check: {e}")
```

---

## Cell 3: Install Piper & Dependencies
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
    git checkout "$PIPER_COMMIT" 2>/dev/null || echo "⚠️  Using existing"
fi

# Install Piper dependencies (without transformers - already installed)
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
echo "✅ Piper installed"
```

---

## Cell 4: Patch Dataset
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
    print(f"✅ Patched {n} DataLoaders")
else:
    print("ℹ️  Already patched")
```

---

## Cell 5: Configuration
```python
# =============================================================================
# 🎛️  CONFIGURATION
# =============================================================================

USE_VIVOS_KAGGLE = True
VIVOS_KAGGLE_ROOT = "/kaggle/input/vivos-vietnamese-speech-corpus-for-asr/vivos"
VIVOS_SPLIT = "train"

MAX_SAMPLES = 3000
MIN_DURATION_S = 1.0
MAX_DURATION_S = 12.0
REBUILD_DATA = False

VOICE_NAME = "vi_VN-vivos-medium"
ESPEAK_VOICE = "vi"
SAMPLE_RATE = 22050

BATCH_SIZE = 24
NUM_WORKERS = 4
ADDITIONAL_STEPS = 8000
VAL_CHECK_INTERVAL = 1000
PRECISION = "16-mixed"
LEARNING_RATE = 2e-4
NUM_DEVICES = None
STRATEGY = "ddp_find_unused_parameters_true"

WORK_DIR = "/kaggle/working"
DATA_DIR = f"{WORK_DIR}/data"
AUDIO_DIR = f"{DATA_DIR}/wavs"
CSV_PATH = f"{DATA_DIR}/metadata.csv"
CACHE_DIR = f"{DATA_DIR}/cache"
CKPT_DIR = f"{WORK_DIR}/checkpoints"
CONFIG_PATH = f"{CKPT_DIR}/{VOICE_NAME}.json"
BASE_CKPT_LOCAL = f"{WORK_DIR}/base_vi_VN.ckpt"

import os, shutil
for d in (DATA_DIR, AUDIO_DIR, CACHE_DIR, CKPT_DIR):
    os.makedirs(d, exist_ok=True)

from pathlib import Path
if USE_VIVOS_KAGGLE:
    vivos_root = Path(VIVOS_KAGGLE_ROOT)
    if not (vivos_root / VIVOS_SPLIT / "waves").exists():
        raise FileNotFoundError(f"VIVOS not found. Add dataset: kynthesis/vivos-vietnamese-speech-corpus-for-asr")
    print(f"✅ VIVOS: {vivos_root}")

free_gb = shutil.disk_usage(WORK_DIR).free / 1e9
print(f"Free disk: {free_gb:.1f} GB")
assert free_gb > 12, f"Need >12 GB"

print(f"\nConfig: {MAX_SAMPLES} samples, batch {BATCH_SIZE}, {ADDITIONAL_STEPS} steps")
```

---

## Cell 6: Download Checkpoint
```python
import os, time, pathlib
from huggingface_hub import hf_hub_download

dst = pathlib.Path(BASE_CKPT_LOCAL)
if dst.exists() and dst.stat().st_size > 800_000_000:
    print(f"✅ Checkpoint: {dst} ({dst.stat().st_size/1e6:.1f} MB)")
else:
    print("Downloading checkpoint...")
    t0 = time.time()
    p = hf_hub_download(
        repo_id="rhasspy/piper-checkpoints",
        repo_type="dataset",
        filename="vi/vi_VN/vais1000/medium/epoch=4769-step=919580.ckpt",
        local_dir=WORK_DIR,
    )
    if pathlib.Path(p) != dst:
        os.replace(p, dst)
    print(f"✅ Downloaded ({dst.stat().st_size/1e6:.1f} MB) in {time.time()-t0:.1f}s")
```

---

## Cell 7: Load VIVOS
```python
import os, csv, time, shutil
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa

print("🇻🇳 Loading VIVOS...")
t0 = time.time()

if REBUILD_DATA:
    shutil.rmtree(AUDIO_DIR, ignore_errors=True)
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

vivos_root = Path(VIVOS_KAGGLE_ROOT)
split_dir = vivos_root / VIVOS_SPLIT
waves_dir = split_dir / "waves"
prompts_path = split_dir / "prompts.txt"

prompts = {}
with open(prompts_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            prompts[parts[0]] = parts[1]

print(f"Found {len(prompts)} prompts")

n_written = n_skip = 0
csv_rows = []

for utt_id, text in prompts.items():
    if n_written >= MAX_SAMPLES:
        break
    
    wav_matches = list(waves_dir.glob(f"**/{utt_id}.wav"))
    if not wav_matches:
        n_skip += 1
        continue
    
    try:
        audio, sr = librosa.load(str(wav_matches[0]), sr=SAMPLE_RATE, mono=True)
        duration = len(audio) / sr
        
        if MIN_DURATION_S <= duration <= MAX_DURATION_S:
            wav_name = f"vi_{n_written:06d}.wav"
            sf.write(str(Path(AUDIO_DIR) / wav_name), audio, sr, subtype="PCM_16")
            csv_rows.append((wav_name, text))
            n_written += 1
            
            if n_written % 500 == 0:
                print(f"  {n_written}/{MAX_SAMPLES} ({time.time()-t0:.1f}s)")
        else:
            n_skip += 1
    except:
        n_skip += 1

with open(CSV_PATH, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter='|', quoting=csv.QUOTE_NONE, escapechar='\\')
    for wav_name, text in csv_rows:
        clean = text.replace('|', ' ').replace('\n', ' ').strip()
        writer.writerow([wav_name, clean])

print(f"✅ {n_written} samples in {time.time()-t0:.1f}s (skipped {n_skip})")
!head -3 "$CSV_PATH"
!ls "$AUDIO_DIR" | wc -l
```

---

## Cell 8: Pre-cache
```python
import sys, time
sys.path.insert(0, "/kaggle/working/piper1-gpl/src")

for m in [k for k in sys.modules if k.startswith("piper")]:
    del sys.modules[m]

from piper.train.vits.dataset import VitsDataModule

print("Pre-caching...")
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

print(f"✅ Done in {time.time()-t0:.1f}s")
!ls "$CACHE_DIR" | wc -l
```

---

## Cell 9: Train
```python
import torch, os, subprocess

_meta = torch.load(BASE_CKPT_LOCAL, map_location="cpu", weights_only=False)
BASE_GLOBAL_STEP = int(_meta.get("global_step", 0))
del _meta

target_max_steps = BASE_GLOBAL_STEP + ADDITIONAL_STEPS
print(f"Base: {BASE_GLOBAL_STEP}, Target: {target_max_steps}")

n_gpu = torch.cuda.device_count()
devices = NUM_DEVICES or max(1, n_gpu)
print(f"GPUs: {devices}")

env = os.environ.copy()
env.update({
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONUNBUFFERED": "1",
    "TF_CPP_MIN_LOG_LEVEL": "2",
})

strategy = STRATEGY if devices > 1 else "auto"

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
    f"--model.sample_rate={SAMPLE_RATE}",
    f"--model.learning_rate={LEARNING_RATE}",
    f"--trainer.max_steps={target_max_steps}",
    f"--trainer.precision={PRECISION}",
    f"--trainer.accelerator=gpu",
    f"--trainer.devices={devices}",
    f"--trainer.strategy={strategy}",
    f"--trainer.val_check_interval={VAL_CHECK_INTERVAL}",
    f"--trainer.default_root_dir={CKPT_DIR}",
    f"--ckpt_path={BASE_CKPT_LOCAL}",
]

print("🏋️ Training...")
ret = subprocess.run(cmd, cwd="/kaggle/working/piper1-gpl", env=env)

if ret.returncode != 0 and devices > 1:
    print("⚠️  Retry 1 GPU...")
    cmd = [c.replace(f"devices={devices}", "devices=1").replace(f"strategy={strategy}", "strategy=auto") for c in cmd]
    ret = subprocess.run(cmd, cwd="/kaggle/working/piper1-gpl", env=env)

assert ret.returncode == 0
print("✅ Training complete!")
```

---

## Cell 10: Export & Test
```python
import glob, os, subprocess, numpy as np, soundfile as sf
import IPython.display as ipd

patterns = [f"{CKPT_DIR}/**/checkpoints/*.ckpt"]
ckpts = sorted([c for p in patterns for c in glob.glob(p, recursive=True)], key=os.path.getmtime)
ckpts = [c for c in ckpts if os.path.abspath(c) != os.path.abspath(BASE_CKPT_LOCAL)]
assert ckpts, "No checkpoint!"

latest = ckpts[-1]
print(f"Checkpoint: {latest} ({os.path.getsize(latest)/1e6:.1f} MB)")

onnx_out = f"{CKPT_DIR}/{VOICE_NAME}.onnx"
ret = subprocess.run([
    "python", "-m", "piper.train.export_onnx",
    "--checkpoint", latest,
    "--output-file", onnx_out,
], cwd="/kaggle/working/piper1-gpl")
assert ret.returncode == 0

import shutil
shutil.copy(CONFIG_PATH, onnx_out + ".json")
print(f"✅ ONNX: {onnx_out} ({os.path.getsize(onnx_out)/1e6:.1f} MB)")

from piper import PiperVoice
voice = PiperVoice.load(onnx_out, config_path=onnx_out + ".json")
text = "Xin chào, đây là giọng nói tiếng Việt từ VIVOS dataset."

audio_chunks = []
for chunk in voice.synthesize(text):
    arr = getattr(chunk, "audio_float_array", None) or chunk.audio_int16_array.astype(np.float32) / 32768.0
    audio_chunks.append(arr)

audio = np.concatenate(audio_chunks)
out_wav = f"{WORK_DIR}/demo.wav"
sf.write(out_wav, audio, voice.config.sample_rate)
print(f"✅ Demo: {out_wav}")

ipd.display(ipd.Audio(audio, rate=voice.config.sample_rate))
print("\n🎉 COMPLETE!")
```
