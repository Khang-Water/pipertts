# Secret Single-Speaker Piper Fine-Tune

Production workflow for a private, long, single-speaker dataset on a remote server with 2x A100. Keep dataset, generated metadata, cache, checkpoints, logs, and ONNX files under `local/`; root `.gitignore` excludes them.

## 0. One-Command Mode (recommended)

Sections 1-7 below are automated by `run_pipeline.sh`. Configure once, run once
inside tmux, check back later:

```bash
cp pipeline.env.example pipeline.env   # edit MANIFEST + SRC_AUDIO_DIR
tmux new -s piper
bash run_pipeline.sh
# detach: Ctrl-b d — reattach later: tmux attach -t piper
```

Behavior:

- Every stage is idempotent; logs go to `local/logs/pipeline_<timestamp>.log`.
- The step target (`base global_step + EXTRA_STEPS`) is computed once and
  persisted to `local/secret_single_speaker/.pipeline/target_step`. Re-running
  after a crash resumes from the newest `last.ckpt` toward the SAME target —
  restarts never extend training. Delete that file to train further.
- If the target is already reached, training is skipped and the pipeline goes
  straight to ONNX export.
- A dry-run (`--print_config`) validates the trainer CLI before real training.

The manual stages below remain valid for debugging individual steps.

## 1. Remote Setup

`piper1-gpl` is not committed in this repo. Clone the patched fork (its `main`
already contains the DDP/shuffle/cache fixes):

```bash
cd /path/to/pipertts
git clone https://github.com/Khang-Water/piper1-gpl.git
```

If you must use upstream instead, pin `2a60c2bd` and apply
`piper1-gpl-local.patch` from this repo's root.

```bash
cd /path/to/pipertts
python3 -m venv .venv
source .venv/bin/activate
cd piper1-gpl
pip install -e '.[train]'
bash build_monotonic_align.sh
python3 setup.py build_ext --inplace
cd ..
```

System packages usually needed:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build espeak-ng
```

## 2. Prepare Dataset

Expected single-speaker manifest:

```text
relative/or/absolute/audio.wav|Transcript text
```

Reference mode is best for long datasets because it avoids copying/resampling the whole corpus up front:

```bash
source .venv/bin/activate
python3 prepare_single_speaker_dataset.py \
  --manifest /secret/path/manifest.csv \
  --audio-dir /secret/path/audio \
  --output-dir local/secret_single_speaker \
  --delimiter '|' \
  --path-column 0 \
  --text-column 1 \
  --sample-rate 22050 \
  --min-duration-s 0.5 \
  --max-duration-s 30 \
  --voice-name vi_VN-secret-medium \
  --espeak-voice vi
```

Use `--mode copy-resample` only if source storage/format causes runtime problems. It writes normalized WAVs under `local/secret_single_speaker/audio`.

## 3. Base Checkpoint

Download and sanitize old HF checkpoint:

```bash
source .venv/bin/activate
bash download_base_checkpoint.sh
```

Default checkpoint:

```text
https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/vi/vi_VN/vais1000/medium/epoch%3D4769-step%3D919580.ckpt
```

Override if needed:

```bash
export BASE_CKPT_URL='https://huggingface.co/datasets/.../your.ckpt'
export BASE_CKPT_PATH="$PWD/local/checkpoints/your-base.ckpt"
bash download_base_checkpoint.sh
```

## 4. Preflight

Run before expensive training:

```bash
source .venv/bin/activate
python3 preflight_training.py \
  --dataset-dir local/secret_single_speaker \
  --piper-dir piper1-gpl \
  --expect-gpus 2 \
  --sample-rows 500 \
  --strict
```

This checks GPU count/name, required packages, `monotonic_align`, metadata shape, sampled audio readability, checkpoint existence, and cache/config paths. It does not print transcript text.

## 5. Build Cache Once

Do this before DDP. Piper writes `.phonemes.pt`, `.audio.pt`, and `.spec.pt` cache files. If two DDP ranks build the same cache at the same time, cache writes can race.

```bash
source .venv/bin/activate
python3 prepare_piper_cache.py \
  --dataset-dir local/secret_single_speaker \
  --piper-dir piper1-gpl
```

## 6. Train on 2x A100

Validate the full CLI config first without touching the GPUs:

```bash
source .venv/bin/activate
DRY_RUN=1 NUM_GPUS=2 EXTRA_STEPS=20000 bash finetune_single_speaker.sh
```

Then train:

```bash
export NUM_GPUS=2
export STRATEGY=ddp
export PRECISION=bf16-mixed
export BATCH_SIZE=16
export NUM_WORKERS=16
export EXTRA_STEPS=20000
bash finetune_single_speaker.sh
```

Step semantics: `--ckpt_path` resumes trainer state, so the restored
`global_step` starts at the base checkpoint's value (e.g. 919580), and
`max_steps` is absolute, not relative. Use `EXTRA_STEPS` for "train N more
steps"; the script reads `global_step` from the checkpoint and computes
`MAX_STEPS = global_step + EXTRA_STEPS`. If you set `MAX_STEPS` directly and
it is <= the checkpoint's `global_step`, the script refuses to start instead
of silently training for 0 steps. Note: VITS uses manual optimization with
two optimizers (generator + discriminator), so `global_step` advances by 2
per batch — `EXTRA_STEPS=20000` is about 10000 batches.

Checkpointing: a `ModelCheckpoint` callback saves every `CKPT_EVERY_N_STEPS`
global steps (default 5000), keeps `CKPT_SAVE_TOP_K` step checkpoints
(default 1), and always refreshes `last.ckpt` for the export script.

`BATCH_SIZE` is per GPU under DDP. Effective batch:

```text
BATCH_SIZE * NUM_GPUS * ACCUMULATE_GRAD_BATCHES
```

Start conservative. If stable, increase `BATCH_SIZE`; if OOM, lower it. On A100, `bf16-mixed` is default because it is usually more stable than fp16 while keeping tensor-core speed.

## 7. Export

```bash
source .venv/bin/activate
bash export_single_speaker_onnx.sh
```

Set `CHECKPOINT=/path/to/last.ckpt` if auto-discovery does not find the checkpoint.

## Debug Checklist

- `metadata.csv` must contain exactly `audio_path|text` rows.
- `dataset_stats.json` should show nonzero `accepted`, low `missing_audio`, and realistic duration percentiles.
- `AUDIO_DIR=/` is correct for reference mode because metadata contains absolute audio paths.
- `AUDIO_DIR=local/secret_single_speaker/audio` is correct for copy-resample mode.
- `CACHE_DIR` can grow large because Piper stores phoneme/audio/spectrogram tensors.
- `--ckpt_path` resumes model and optimizer state; `MAX_STEPS` is absolute trainer steps, not extra steps. Prefer `EXTRA_STEPS`.
- Resuming also restores optimizer LR and the per-epoch ExponentialLR schedulers, so the effective LR is roughly `2e-4 * 0.999875^4769 ~= 1.1e-4`, and `--model.learning_rate` has no effect under full resume.
- `piper1-gpl` carries local patches (DDP rank!=0 `prepare_data`, train shuffle, cache `.exists()` checks, atomic config write). Clone the `Khang-Water/piper1-gpl` fork (already patched); `piper1-gpl-local.patch` is the fallback for an upstream clone at pin `2a60c2bd`.
- Old HF checkpoints may contain legacy hparams that fail modern `torch.load(..., weights_only=True)`; sanitizer handles trusted HF checkpoint once.
- DDP should run after `prepare_piper_cache.py`, otherwise ranks can race while writing cache files.

