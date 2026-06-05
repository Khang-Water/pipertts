"""Build the Kaggle Vietnamese Piper finetuning notebook (.ipynb).

Run: python build_notebook.py
Output: piper_vi_finetune_kaggle.ipynb
"""
import json
import uuid
from pathlib import Path

OUT = Path(__file__).parent / "piper_vi_finetune_kaggle.ipynb"


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


# ---------------------------------------------------------------------------
# CELLS
# ---------------------------------------------------------------------------
cells = []

cells.append(md(r"""# 🇻🇳 Piper TTS — Finetune cho tiếng Việt trên Kaggle (Fast)

Notebook tối ưu để **finetune Piper TTS** cho tiếng Việt nhanh nhất có thể trên Kaggle.

**Nguồn tham khảo (đã verify):**
- Code base: [OHF-Voice/piper1-gpl](https://github.com/OHF-Voice/piper1-gpl) v1.4.2
- Checkpoint warmstart: [rhasspy/piper-checkpoints](https://huggingface.co/datasets/rhasspy/piper-checkpoints) → `vi/vi_VN/vais1000/medium/epoch=4769-step=919580.ckpt` (846 MB, `global_step=919580`)
- Dataset mặc định: [doof-ferb/LSVSC](https://huggingface.co/datasets/doof-ferb/LSVSC) (100h, 24kHz, mở, multi-speaker → train như "voice trung bình")

## ⚙️ Cài đặt Kaggle (BẮT BUỘC)

1. **Accelerator**: chọn **GPU T4 x2** (Settings → Accelerator → GPU T4 x2). Gấp đôi tốc độ vs P100.
2. **Internet**: bật **Internet ON** (Settings → Internet).
3. **Persistence**: Settings → Persistence → **Files only** để giữ output giữa các session.
4. Nếu muốn dùng dataset gated (`capleaf/viVoice`, `thivux/phoaudiobook`) → thêm **Kaggle Secret** tên `HF_TOKEN`.

## 🚀 Các tối ưu tốc độ áp dụng

| Tối ưu | Cách | Tăng tốc |
|---|---|---|
| **Mixed precision** (`16-mixed` / `bf16-mixed`) | `--trainer.precision` | ~1.8x |
| **DDP 2 GPU** (T4 x2) | `--trainer.devices=2 --trainer.strategy=ddp_find_unused_parameters_true` | ~1.9x |
| **TF32 + cuDNN benchmark** | `--trainer.benchmark=true` + env | ~1.1x |
| **`pin_memory` + `persistent_workers`** | patch `dataset.py` (3 DataLoader) | ~1.2x |
| **Pre-cache `prepare_data()`** | gọi riêng cell, training subprocess re-iterate cache | tránh redo |
| **`num_workers` = 4** | match số CPU Kaggle | parallel I/O |
| **Sub-sample 3000 mẫu** | giảm thời gian/epoch | hội tụ nhanh khi warmstart cùng ngôn ngữ |
| **Warmstart vi_VN ckpt** | `--ckpt_path` resume + `+ADDITIONAL_STEPS` | hội tụ rất nhanh |
| **`max_steps` cap** | tránh chạy quá time-limit Kaggle 12h | dừng đúng lúc |

> ⚠️ `--ckpt_path` resume = restore cả `global_step`. Notebook tự đọc `global_step` từ base ckpt rồi cộng `ADDITIONAL_STEPS` để `max_steps` đúng nghĩa "train thêm N step".

> ⚠️ `ddp_find_unused_parameters_true` là **compatibility setting** cho manual-optimization GAN của VITS (không phải tăng tốc; chỉ để DDP không crash).
"""))

cells.append(md(r"""## 1. Kiểm tra môi trường (GPU / CPU / RAM)"""))

cells.append(code(r"""import os, sys, subprocess, shutil, multiprocessing
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
!df -h /kaggle/working 2>/dev/null || df -h /tmp
"""))

cells.append(md(r"""## 2. Cài system + Python dependencies

Bao gồm: `build-essential`, `cmake`, `ninja-build`, `espeak-ng`, các Python deps cho training + dataset.
"""))

cells.append(code(r"""%%bash
set -euo pipefail
# --- System packages ---
apt-get -qq update
apt-get -qq install -y build-essential cmake ninja-build espeak-ng libsndfile1 sox > /dev/null
echo "✅ apt deps installed"

# Verify espeak-ng có voice vi
espeak-ng --voices=vi || (echo "❌ espeak-ng không có voice vi!"; exit 1)
"""))

cells.append(code(r"""%%bash
set -euo pipefail
cd /kaggle/working
# Pin commit để reproducibility + tránh upstream bễ patch dataset.py (assert n==3 ở cell 7)
PIPER_COMMIT="2a60c2bd152356b613673ecaa0e15cbd3c0e502c"
if [ ! -d piper1-gpl ]; then
    git clone https://github.com/OHF-Voice/piper1-gpl.git
    cd piper1-gpl && git checkout "$PIPER_COMMIT"
else
    cd piper1-gpl
    git fetch --quiet origin "$PIPER_COMMIT" || true
    git checkout "$PIPER_COMMIT" 2>/dev/null || echo "⚠️  Could not checkout pinned commit; using existing."
fi

# Python deps — ép pin để khớp với piper1-gpl [train] extras + thêm deps notebook cần
pip install -q --upgrade pip
pip install -q \
    "lightning>=2,<3" "tensorboard>=2,<3" "tensorboardX>=2,<3" \
    "jsonargparse[signatures]>=4.27.7" "onnx>=1,<2" \
    "pysilero-vad>=2.1,<3" "cython>=3,<4" "librosa<1" \
    "scikit-build<1" "pathvalidate>=3,<4" "onnxruntime>=1,<2" \
    "datasets[audio]>=2.19,<4" "huggingface_hub>=0.23,<1" "soundfile>=0.12,<1"

# Cài piper editable (build cython extension monotonic_align + libpiper espeakbridge)
pip install -q -e .

# Build in-place để import trực tiếp từ src/ luôn có piper.espeakbridge
python setup.py build_ext --inplace

# Build monotonic_align Cython core
bash build_monotonic_align.sh

# Verify
python -c "import piper; print('piper:', piper.__file__)"
python -c "import sys; sys.path.insert(0, '/kaggle/working/piper1-gpl/src'); import piper.espeakbridge; print('espeakbridge OK')"
python -c "from piper.train.vits.monotonic_align import maximum_path; print('monotonic_align OK')"
python -c "import datasets, soundfile, huggingface_hub; print('dataset deps OK')"
echo "✅ piper1-gpl installed"
"""))

cells.append(md(r"""## 3. Patch `dataset.py` — bật `pin_memory` & `persistent_workers`

Hai tham số này **không có sẵn** trong code Piper. Patch tăng throughput ~20%.

Có assert đảm bảo đúng 3 DataLoader được patch (fail-fast nếu upstream đổi code).
"""))

cells.append(code(r"""from pathlib import Path

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
    assert n == 3, (
        f"Expected to patch 3 DataLoaders, found {n}. "
        f"Upstream dataset.py may have changed - inspect manually."
    )
    new_src = src.replace(pattern, replacement)
    compile(new_src, str(ds), "exec")  # syntax check
    ds.write_text(new_src)
    print(f"✅ Patched {n} DataLoaders with pin_memory + persistent_workers")
else:
    print("ℹ️  Already patched")

import subprocess
print(subprocess.check_output(["grep", "-n", "pin_memory", str(ds)], text=True))
"""))

cells.append(md(r"""## 4. ⚙️ Cấu hình training — chỉnh các knob ở đây

Đây là cell duy nhất bạn cần chỉnh cho hầu hết case.
"""))

cells.append(code(r"""# =============================================================================
# 🎛️  USER KNOBS — chỉnh ở đây
# =============================================================================

# --- Dataset (default: doof-ferb/LSVSC — open, standard HF datasets API, 24kHz, 100h) ---
# Các option khác:
#   "doof-ferb/LSVSC"        : 100h, mở, MIXED-SPEAKER (không có cột speaker thật).
#                              → train như single "voice trung bình". Tốt cho demo TTS tiếng Việt.
#   "capleaf/viVoice"        : 1016h, GATED (HF_TOKEN), có cột speaker → có thể single-clone.
#   "thivux/phoaudiobook"    : 1494h, GATED.
#   "ntt123/viet-tts-dataset": single-speaker nhưng **đóng gói trong tar.gz** — KHÔNG load qua HF
#                              datasets API thông thường. Nếu muốn dùng, cần code riêng tải tar.
DATASET_NAME      = "doof-ferb/LSVSC"
DATASET_SPLIT     = "train"
DATASET_TEXT_COL  = "transcription"   # auto-detect nếu không khớp
DATASET_AUDIO_COL = "audio"

# --- Optional pseudo-speaker filter ---
# LSVSC không có cột speaker thật nhưng có 'gender' / 'dialect' / 'emotion'. Có thể dùng làm pseudo-speaker
# để giọng nhất quán hơn. Mặc định AUTO_PSEUDO_SPEAKER=True sẽ tự pick value đông nhất từ 'gender'.
# Override: đặt PSEUDO_SPEAKER_FILTER = {"gender": "male", "dialect": "northern"} để tự chọn,
# hoặc đặt AUTO_PSEUDO_SPEAKER=False + PSEUDO_SPEAKER_FILTER=None để giữ tất cả.
PSEUDO_SPEAKER_FILTER = None
AUTO_PSEUDO_SPEAKER = True
AUTO_PSEUDO_SPEAKER_COLS = ("gender", "dialect", "emotion")

# --- Speaker filter (chỉ cho dataset CÓ cột speaker thật, vd: capleaf/viVoice) ---
# - SINGLE_SPEAKER=True : đếm speaker, giữ speaker đông nhất → single-voice clone chuẩn
# - SINGLE_SPEAKER=False: giữ tất cả → model học "voice trung bình" (default cho LSVSC)
# - SINGLE_SPEAKER=True nhưng không có cột speaker → notebook RAISE rõ ràng (không silent).
SINGLE_SPEAKER    = False
SPEAKER_COL       = None     # None => auto-detect (speaker_id, speaker, client_id, ...)

# --- Sub-sample ---
MAX_SAMPLES       = 3000     # 2000-5000 đủ cho finetune nhanh khi đã warmstart vi_VN
MIN_DURATION_S    = 1.0
MAX_DURATION_S    = 12.0
REBUILD_DATA      = False    # True => xoá AUDIO_DIR + CACHE_DIR trước khi rebuild

# --- Voice config ---
VOICE_NAME        = "vi_VN-finetune-medium"
ESPEAK_VOICE      = "vi"      # tên voice espeak-ng (verified: espeak-ng --voices=vi)
SAMPLE_RATE       = 22050     # match warmstart medium ckpt; librosa tự resample từ source

# --- Training ---
BATCH_SIZE        = 24        # T4 16GB + fp16: 24 OK; P100 → hạ 16; OOM → giảm tiếp
NUM_WORKERS       = 4         # số CPU Kaggle
ADDITIONAL_STEPS  = 8000      # số step train THÊM kể từ global_step của base ckpt
VAL_CHECK_INTERVAL = 1000
PRECISION         = "16-mixed"   # T4 dùng 16-mixed; A100/L4/H100 đổi "bf16-mixed"
LEARNING_RATE     = 2e-4
NUM_DEVICES       = None      # None => auto-detect; ép số bằng 1 hoặc 2 nếu cần
STRATEGY          = "ddp_find_unused_parameters_true"  # bắt buộc cho VITS manual-opt

# --- Paths ---
WORK_DIR          = "/kaggle/working"
DATA_DIR          = f"{WORK_DIR}/data"
AUDIO_DIR         = f"{DATA_DIR}/wavs"
CSV_PATH          = f"{DATA_DIR}/metadata.csv"
CACHE_DIR         = f"{DATA_DIR}/cache"
CKPT_DIR          = f"{WORK_DIR}/checkpoints"
CONFIG_PATH       = f"{CKPT_DIR}/{VOICE_NAME}.json"
BASE_CKPT_LOCAL   = f"{WORK_DIR}/base_vi_VN.ckpt"

# --- HF Token (cho dataset gated) ---
HF_TOKEN = None
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    print("✅ HF_TOKEN loaded from Kaggle Secret")
except Exception:
    print("ℹ️  No HF_TOKEN (chỉ cần cho dataset gated)")

import os, shutil
for d in (DATA_DIR, AUDIO_DIR, CACHE_DIR, CKPT_DIR):
    os.makedirs(d, exist_ok=True)

# Disk-space guard
free_gb = shutil.disk_usage(WORK_DIR).free / 1e9
print(f"Free disk on {WORK_DIR}: {free_gb:.1f} GB")
assert free_gb > 12, (
    f"Cần >12 GB free trên {WORK_DIR}. Hiện chỉ còn {free_gb:.1f} GB. "
    "Giảm MAX_SAMPLES, bật REBUILD_DATA để wipe cache cũ, hoặc dọn dẹp Kaggle output."
)

print(f"DATASET          : {DATASET_NAME}")
print(f"MAX_SAMPLES      : {MAX_SAMPLES}")
print(f"BATCH_SIZE       : {BATCH_SIZE}  (x{NUM_DEVICES or 'auto'} GPU)")
print(f"ADDITIONAL_STEPS : {ADDITIONAL_STEPS}")
print(f"PRECISION        : {PRECISION}")
"""))

cells.append(md(r"""## 5. Tải base checkpoint tiếng Việt từ HuggingFace

File ~846 MB, có resume nếu bị ngắt.
"""))

cells.append(code(r"""import os, time, pathlib

dst = pathlib.Path(BASE_CKPT_LOCAL)
if dst.exists() and dst.stat().st_size > 800_000_000:
    print(f"✅ Đã có {dst} ({dst.stat().st_size/1e6:.1f} MB)")
else:
    print("Downloading base ckpt từ HuggingFace...")
    t0 = time.time()
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(
        repo_id="rhasspy/piper-checkpoints",
        repo_type="dataset",
        filename="vi/vi_VN/vais1000/medium/epoch=4769-step=919580.ckpt",
        local_dir=WORK_DIR,
    )
    if pathlib.Path(p) != dst:
        os.replace(p, dst)
    print(f"✅ Saved {dst} ({dst.stat().st_size/1e6:.1f} MB) in {time.time()-t0:.1f}s")
"""))

cells.append(md(r"""## 6. Tải & tiền xử lý dataset → CSV + WAV files

- Stream dataset từ HF (không tải hết về RAM)
- Auto-detect cột text + cột speaker (nếu có)
- Single-speaker filter (raise nếu bật mà không có cột speaker)
- Lọc duration, ghi WAV mono 22050Hz PCM, sinh CSV `wav|text`
"""))

cells.append(code(r"""import os, csv, time, collections, shutil
import numpy as np
import soundfile as sf
from datasets import load_dataset, Audio

t0 = time.time()

if REBUILD_DATA:
    print("REBUILD_DATA=True → xoá AUDIO_DIR + CACHE_DIR")
    shutil.rmtree(AUDIO_DIR, ignore_errors=True)
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

def stream_ds(decode_audio: bool = True):
    # Stream dataset. decode_audio=False -> bỏ qua decode (nhanh hơn nhiều khi chỉ cần metadata).
    kw = dict(split=DATASET_SPLIT, streaming=True)
    if HF_TOKEN:
        kw["token"] = HF_TOKEN
    d = load_dataset(DATASET_NAME, **kw)
    if decode_audio:
        d = d.cast_column(DATASET_AUDIO_COL, Audio(sampling_rate=SAMPLE_RATE, mono=True))
    else:
        # Disable audio decoding → rất nhanh khi scan metadata
        d = d.cast_column(DATASET_AUDIO_COL, Audio(decode=False))
    return d

# Peek columns từ stream metadata-only (không decode audio → nhanh)
_peek_ds = stream_ds(decode_audio=False)
peek = next(iter(_peek_ds))
del _peek_ds
ds = stream_ds()  # full stream cho main loop
columns = list(peek.keys())
print("Dataset columns:", columns)

# Auto-detect text column
text_col = DATASET_TEXT_COL if DATASET_TEXT_COL in columns else None
if text_col is None:
    for c in ("transcription", "text", "sentence", "transcript", "normalized_text", "raw_transcription"):
        if c in columns:
            text_col = c; break
if text_col is None:
    raise RuntimeError(f"Không tìm thấy cột text trong: {columns}")
print("Using text column:", text_col)

# Auto-detect speaker column
speaker_candidates = [SPEAKER_COL, "speaker_id", "speaker", "speaker_name", "client_id", "reader"]
speaker_col = next((c for c in speaker_candidates if c and c in columns), None)

if SINGLE_SPEAKER:
    if speaker_col is None:
        raise RuntimeError(
            f"SINGLE_SPEAKER=True nhưng dataset {DATASET_NAME} không có cột speaker (columns={columns}). "
            "Hoặc dataset thực sự là single-speaker (như ntt123/viet-tts-dataset) thì set SINGLE_SPEAKER=False và bỏ qua filter, "
            "hoặc đổi DATASET_NAME sang dataset có speaker IDs."
        )
    print(f"Speaker column detected: {speaker_col}")
else:
    print("SINGLE_SPEAKER=False → giữ tất cả rows")

# Single-speaker filter: scan để chọn speaker đông nhất
chosen_speaker = None
if SINGLE_SPEAKER and speaker_col:
    counts = collections.Counter()
    ds_scan = stream_ds(decode_audio=False)  # metadata-only → fast
    for i, row in enumerate(ds_scan):
        sp = row.get(speaker_col)
        if sp is not None:
            counts[str(sp)] += 1
        if i >= MAX_SAMPLES * 10:
            break
    if not counts:
        raise RuntimeError(f"Không tìm thấy speaker nào trong {MAX_SAMPLES*10} rows đầu")
    chosen_speaker, n_for_chosen = counts.most_common(1)[0]
    print(f"Chosen speaker: {chosen_speaker} ({n_for_chosen} samples in scan window)")
    ds = stream_ds()  # fresh stream cho main loop

# Auto pseudo-speaker: nếu filter None + dataset có cột pseudo (gender/dialect), tự pick value đông nhất
# → voice consistency cao hơn nhiều so với toàn bộ mixed-speaker dataset.
if PSEUDO_SPEAKER_FILTER is None and AUTO_PSEUDO_SPEAKER and not SINGLE_SPEAKER:
    for auto_col in AUTO_PSEUDO_SPEAKER_COLS:
        if auto_col in columns:
            _counts = collections.Counter()
            _scan = stream_ds(decode_audio=False)  # metadata-only → fast
            for _i, _row in enumerate(_scan):
                _v = _row.get(auto_col)
                if _v is not None:
                    _counts[str(_v)] += 1
                if _i >= 800:
                    break
            if _counts:
                _top_v, _top_n = _counts.most_common(1)[0]
                PSEUDO_SPEAKER_FILTER = {auto_col: _top_v}
                print(f"Auto pseudo-speaker filter: {PSEUDO_SPEAKER_FILTER}  "
                      f"(top {_top_n}/{sum(_counts.values())} in scan; distribution: {dict(_counts.most_common(5))})")
                ds = stream_ds()
                break

# Validate PSEUDO_SPEAKER_FILTER keys exist in columns
if PSEUDO_SPEAKER_FILTER:
    missing = [k for k in PSEUDO_SPEAKER_FILTER if k not in columns]
    if missing:
        raise RuntimeError(
            f"PSEUDO_SPEAKER_FILTER keys {missing} không có trong dataset columns {columns}. "
            "Xóa key không tồn tại hoặc set PSEUDO_SPEAKER_FILTER=None."
        )
    print(f"Pseudo-speaker filter ACTIVE: {PSEUDO_SPEAKER_FILTER}")
else:
    print("Pseudo-speaker filter: None (giữ tất cả rows)")

# Main loop: ghi WAV + CSV
n_written, n_skip = 0, 0
csv_rows = []

for row in ds:
    if n_written >= MAX_SAMPLES:
        break
    txt = (row.get(text_col) or "").strip()
    if not txt:
        n_skip += 1; continue
    if chosen_speaker is not None and str(row.get(speaker_col)) != chosen_speaker:
        continue
    if PSEUDO_SPEAKER_FILTER:
        if any(str(row.get(k)) != str(v) for k, v in PSEUDO_SPEAKER_FILTER.items()):
            continue

    aud = row[DATASET_AUDIO_COL]
    arr = np.asarray(aud["array"], dtype=np.float32)
    sr  = int(aud["sampling_rate"])
    dur = len(arr) / sr
    if dur < MIN_DURATION_S or dur > MAX_DURATION_S:
        n_skip += 1; continue

    wav_name = f"vi_{n_written:06d}.wav"
    sf.write(os.path.join(AUDIO_DIR, wav_name), arr, sr, subtype="PCM_16")
    csv_rows.append((wav_name, txt))
    n_written += 1
    if n_written % 500 == 0:
        print(f"  {n_written}/{MAX_SAMPLES} ... (skip {n_skip})  elapsed {time.time()-t0:.1f}s")

# Ghi CSV với delimiter '|', strip newlines/pipes trong text
with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")
    for name, text in csv_rows:
        clean = text.replace("|", " ").replace("\n", " ").replace("\r", " ").strip()
        w.writerow([name, clean])

print(f"✅ Wrote {n_written} samples in {time.time()-t0:.1f}s")
print(f"   audio_dir = {AUDIO_DIR}")
print(f"   csv_path  = {CSV_PATH}")
!head -3 "$CSV_PATH"
!ls "$AUDIO_DIR" | wc -l
!du -sh "$AUDIO_DIR"
"""))

cells.append(md(r"""## 7. Pre-cache phoneme + spectrogram

`prepare_data()` của Piper sẽ:
- Phonemize qua espeak-ng (`vi`)
- Trim silence bằng Silero VAD
- Pre-compute spectrogram + lưu cache `.pt`

Chạy riêng để bước train không nghẽn ở data-prep (vốn single-thread). Subprocess train sau sẽ re-iterate cache (nhanh, vì các file đã tồn tại sẽ bị skip).
"""))

cells.append(code(r"""import sys, time, os, subprocess
sys.path.insert(0, "/kaggle/working/piper1-gpl/src")

# Self-heal: nếu kernel restart hoặc editable install chưa build extension,
# build lại trước khi import từ src/.
repo_dir = "/kaggle/working/piper1-gpl"
try:
    import piper.espeakbridge  # type: ignore[attr-defined]
except Exception:
    print("ℹ️  Building piper.espeakbridge in-place...")
    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=repo_dir,
        check=True,
    )
    import piper.espeakbridge  # type: ignore[attr-defined]

# Reset module cache để pick up patched dataset.py
for m in [k for k in sys.modules if k.startswith("piper")]:
    del sys.modules[m]

from piper.train.vits.dataset import VitsDataModule

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
print(f"✅ prepare_data() done in {time.time()-t0:.1f}s")
print(f"   cache_dir = {CACHE_DIR}")
!ls "$CACHE_DIR" | head -5
!echo -n "Total cache files: "; ls "$CACHE_DIR" | wc -l
"""))

cells.append(md(r"""## 8. 🏋️ Train

`python -m piper.train fit` với tất cả flag tối ưu.

### ⚠️ Quan trọng: `--ckpt_path` semantics

Lightning's `--ckpt_path` **resume** training, restore cả `global_step` từ base ckpt. Base ckpt có `global_step=919580`. Nếu set `--trainer.max_steps=8000` thì Lightning sẽ thấy `919580 > 8000` và DỪNG NGAY.

→ Notebook tự đọc `global_step` từ base ckpt rồi tính `target_max_steps = base_step + ADDITIONAL_STEPS`.
"""))

cells.append(code(r"""import torch, os, shlex, sys

# Đọc global_step từ base ckpt
print("Reading global_step từ base ckpt...")
_meta = torch.load(BASE_CKPT_LOCAL, map_location="cpu", weights_only=False)
BASE_GLOBAL_STEP = int(_meta.get("global_step", 0))
del _meta
target_max_steps = BASE_GLOBAL_STEP + ADDITIONAL_STEPS
print(f"Base global_step    : {BASE_GLOBAL_STEP}")
print(f"Additional steps    : {ADDITIONAL_STEPS}")
print(f"Target max_steps    : {target_max_steps}")

n_gpu = torch.cuda.device_count()
devices = NUM_DEVICES or max(1, n_gpu)
print(f"Using {devices} GPU(s)")

# Env tăng tốc
env = os.environ.copy()
env["TOKENIZERS_PARALLELISM"] = "false"
env["PYTHONUNBUFFERED"]      = "1"
env["TF_CPP_MIN_LOG_LEVEL"]  = "2"
env["TORCH_CUDNN_V8_API_ENABLED"] = "1"

strategy_arg = STRATEGY if devices > 1 else "auto"

cmd = [
    sys.executable, "-m", "piper.train", "fit",
    # data
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
    # model
    f"--model.sample_rate={SAMPLE_RATE}",
    f"--model.learning_rate={LEARNING_RATE}",
    # trainer
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
    # resume từ checkpoint Việt (Lightning restore weights + optimizer + global_step)
    f"--ckpt_path={BASE_CKPT_LOCAL}",
]
print("\nCommand:\n  " + " \\\n  ".join(shlex.quote(c) for c in cmd))
"""))

cells.append(code(r"""import subprocess
from pathlib import Path

repo_dir = "/kaggle/working/piper1-gpl"
train_log = Path(CKPT_DIR) / "train.log"
train_log.parent.mkdir(parents=True, exist_ok=True)


def run_and_log(args):
    with open(train_log, "a", encoding="utf-8") as log_f:
        log_f.write("\n" + "=" * 80 + "\n")
        log_f.write("COMMAND:\n")
        log_f.write(" ".join(args) + "\n")
        log_f.write("=" * 80 + "\n")
        log_f.flush()
        proc = subprocess.Popen(
            args,
            cwd=repo_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_f.write(line)
        return proc.wait()


retcode = run_and_log(cmd)
print("\nTrain exit code:", retcode)

# Auto-fallback: nếu DDP 2 GPU fail → retry với 1 GPU
if retcode != 0 and devices > 1:
    print(f"\n⚠️  DDP {devices}-GPU training failed (exit {retcode}). Retrying với 1 GPU + strategy=auto...")
    cmd_1gpu = []
    for arg in cmd:
        if arg.startswith("--trainer.devices="):
            cmd_1gpu.append("--trainer.devices=1")
        elif arg.startswith("--trainer.strategy="):
            cmd_1gpu.append("--trainer.strategy=auto")
        else:
            cmd_1gpu.append(arg)
    retcode = run_and_log(cmd_1gpu)
    print("\nTrain (1-GPU retry) exit code:", retcode)

if retcode != 0:
    tail_lines = train_log.read_text(encoding="utf-8", errors="ignore").splitlines()[-120:]
    print("\n===== train.log tail =====")
    print("\n".join(tail_lines))
    print("===== end tail =====")

    tail_text = "\n".join(tail_lines)
    hints = []
    if "CUDA out of memory" in tail_text:
        hints.append("CUDA OOM: giảm BATCH_SIZE xuống 16 hoặc 12, hoặc ép NUM_DEVICES=1.")
    if "global_step" in tail_text and "max_steps" in tail_text:
        hints.append("Checkpoint resume mismatch: tăng ADDITIONAL_STEPS để target max_steps > base global_step.")
    if "size mismatch" in tail_text or "shape mismatch" in tail_text:
        hints.append("Checkpoint/model mismatch: nếu dataset thành multi-speaker thì không dùng --ckpt_path base single-speaker; chuyển sang --model.vocoder_warmstart_ckpt.")
    if "No supported gpu backend found" in tail_text or "MisconfigurationException" in tail_text:
        hints.append("Trainer/GPU config lỗi: thử NUM_DEVICES=1 và PRECISION='16-mixed'.")
    if "ImportError" in tail_text or "ModuleNotFoundError" in tail_text:
        hints.append("Môi trường chưa đủ dependency: chạy lại cell cài đặt và prepare_data từ đầu.")

    if hints:
        print("\nPossible fixes:")
        for hint in hints:
            print(f"- {hint}")

    raise RuntimeError(
        f"Training failed. Xem log đầy đủ tại: {train_log}"
    )
"""))

cells.append(md(r"""## 9. Tìm checkpoint mới nhất + Export ONNX
"""))

cells.append(code(r"""import glob, os, subprocess, sys

# Tìm tất cả ckpt do Lightning ghi, lấy mới nhất theo mtime
patterns = [
    f"{CKPT_DIR}/**/checkpoints/*.ckpt",
    f"{CKPT_DIR}/**/*.ckpt",
]
ckpts = set()
for pat in patterns:
    ckpts.update(glob.glob(pat, recursive=True))
ckpts = sorted(ckpts, key=os.path.getmtime)
# Loại trừ base ckpt (nếu lỡ nằm trong CKPT_DIR)
ckpts = [c for c in ckpts if os.path.abspath(c) != os.path.abspath(BASE_CKPT_LOCAL)]
assert ckpts, f"Không tìm thấy checkpoint mới trong {CKPT_DIR}. Lightning có thể chưa lưu (cần ít nhất 1 validation pass)."
latest = ckpts[-1]
print(f"Latest checkpoint: {latest}  ({os.path.getsize(latest)/1e6:.1f} MB)")

onnx_out = f"{CKPT_DIR}/{VOICE_NAME}.onnx"
ret = subprocess.run([
    sys.executable, "-m", "piper.train.export_onnx",
    "--checkpoint", latest,
    "--output-file", onnx_out,
], cwd="/kaggle/working/piper1-gpl")
assert ret.returncode == 0, "ONNX export failed!"

# Copy voice config (sinh ra ở prepare_data)
import shutil
shutil.copy(CONFIG_PATH, onnx_out + ".json")

print("\n✅ Exported:")
print(f"   {onnx_out}        ({os.path.getsize(onnx_out)/1e6:.1f} MB)")
print(f"   {onnx_out}.json")
"""))

cells.append(md(r"""## 10. ✅ Test inference + nghe thử
"""))

cells.append(code(r"""import numpy as np, soundfile as sf, IPython.display as ipd
from piper import PiperVoice

voice = PiperVoice.load(onnx_out, config_path=onnx_out + ".json")
text = "Xin chào, đây là giọng nói tiếng Việt được tinh chỉnh trên Kaggle bằng Piper TTS."

audio_chunks = []
for chunk in voice.synthesize(text):
    arr = getattr(chunk, "audio_float_array", None)
    if arr is None:
        arr = chunk.audio_int16_array.astype(np.float32) / 32768.0
    audio_chunks.append(arr)
audio = np.concatenate(audio_chunks)
sr = voice.config.sample_rate

out_wav = f"{WORK_DIR}/demo_vi.wav"
sf.write(out_wav, audio, sr, subtype="PCM_16")
print(f"✅ Saved {out_wav}  duration={len(audio)/sr:.2f}s")
ipd.display(ipd.Audio(audio, rate=sr))
"""))

cells.append(md(r"""## 📦 Output files

Sau khi chạy xong, các file quan trọng tại `/kaggle/working/`:

| File | Mô tả |
|---|---|
| `checkpoints/<VOICE_NAME>.onnx` | Model ONNX cho inference |
| `checkpoints/<VOICE_NAME>.onnx.json` | Config (sample_rate, phoneme_id_map, ...) |
| `checkpoints/lightning_logs/version_*/checkpoints/*.ckpt` | PyTorch Lightning ckpt (để resume train) |
| `demo_vi.wav` | Audio demo |
| `checkpoints/lightning_logs/` | TensorBoard logs |

**Tải về local:**
- Kaggle tab **Output** → click file → Download
- Hoặc nén: `!cd /kaggle/working && zip -r piper_vi.zip checkpoints demo_vi.wav`

**Để tiếp tục train ở session sau:**
1. Save notebook + Output.
2. Session mới: REBUILD_DATA=False (giữ cache), `--ckpt_path=<ckpt vừa train>` thay vì base.
3. Tăng `ADDITIONAL_STEPS` cho lần tiếp.

---

## ⚠️ Troubleshooting

| Lỗi | Cách fix |
|---|---|
| `CUDA out of memory` | Giảm `BATCH_SIZE` (24→16→12) hoặc `NUM_DEVICES=1` |
| `espeak-ng: voice 'vi' not found` | Cell 2 chưa chạy xong |
| `monotonic_align` ImportError | Cell 2 chưa build cython; chạy lại `bash build_monotonic_align.sh` |
| `404 from HF (gated dataset)` | Cần `HF_TOKEN` Kaggle secret với quyền truy cập |
| `RuntimeError: SINGLE_SPEAKER=True nhưng ...` | Set `SINGLE_SPEAKER=False` (cho dataset không có speaker col) hoặc đổi dataset |
| Training dừng ngay sau khi load ckpt | `max_steps` < `BASE_GLOBAL_STEP`. Tăng `ADDITIONAL_STEPS` |
| `state_dict shape mismatch` khi load base | Đang train multi-speaker với base single-speaker. Dùng `--model.vocoder_warmstart_ckpt` thay `--ckpt_path` |
| Quá chậm | Giảm `MAX_SAMPLES`, tăng `NUM_WORKERS`, hoặc thử `bf16-mixed` (chỉ GPU support) |

**Tham khảo:**
- [Piper TRAINING.md](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/TRAINING.md)
- [Lightning Trainer flags](https://lightning.ai/docs/pytorch/stable/common/trainer.html)
- [rhasspy/piper-checkpoints](https://huggingface.co/datasets/rhasspy/piper-checkpoints)
"""))

# ---------------------------------------------------------------------------
# Build notebook
# ---------------------------------------------------------------------------
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
        "kaggle": {
            "accelerator": "nvidiaTeslaT4",
            "dataSources": [],
            "isGpuEnabled": True,
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
print(f"✅ Wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB, {len(cells)} cells)")
