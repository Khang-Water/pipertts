# Teaching Checklist — Piper TTS Finetune Pipeline (Kaggle)

Trạng thái hiểu bài qua quiz. ✓ = đã verify qua quiz/restate, ◐ = đã giải thích chưa quiz, ☐ = chưa cover.

## 1. Problem — chuỗi 5 lỗi

- [x] **Lỗi #1: hub/transformers triangle** — Kaggle transformers mới cần hub>=1, datasets<4 cần hub<1, torchmetrics import transformers chỉ vì thấy nó tồn tại. (Quiz lần 1 sai → re-teach → re-quiz đúng: triage "X có được dùng thật không?")
- [x] **Lỗi #2: PosixPath weights_only** — PyTorch 2.6 đổi default, Lightning `_parse_ckpt_path` hard-code `weights_only=True`, ckpt cũ (old piper + PL 1.7) chứa PosixPath trong hparams. (Quiz 2 đúng)
- [x] **Lỗi #3 (ẩn): hparams schema mismatch** — keys cũ (sample_bytes, channels...) không có trong VitsModel mới → jsonargparse reject; hparams ckpt override CLI args. (Quiz 2 đúng)
- [ ] **Lỗi #4: DDP prepare_data/setup contract** — prepare_data chỉ chạy rank 0, setup mọi rank, piper lưu state in-memory trong prepare_data → rank 1 chết, rank 0 treo NCCL barrier. *(chưa quiz)*
- [ ] **Lỗi #5: val_check_interval semantics** — mặc định = interval trong epoch, dataset nhỏ → 62 batches < 1000; `check_val_every_n_epoch=null` đổi nghĩa thành global steps. *(chưa quiz)*

## 2. Solution — design decisions

- [x] Gỡ transformers vs pin version — triage "có dùng thật không" trước (re-quiz đúng)
- [x] Sanitize lọc arch hparams: schema mismatch + hparams override CLI → giữ LR knob hiệu lực (quiz đúng)
- [ ] Tại sao sanitize ckpt thay vì patch `add_safe_globals` vào piper source *(chưa quiz — đáp: sanitize cover MỌI load site: parse_ckpt_path, resume, DDP ranks; file fix 1 lần vĩnh viễn, không phụ thuộc re-clone repo)*
- [ ] Patch shuffle=True train_dataloader — upstream thiếu, SequentialSampler → data cùng thứ tự mỗi epoch *(chưa quiz)*
- [ ] Edge case: np.float64 là float subclass → lọt exact-type check — bắt được nhờ test local *(chưa quiz)*

## 3. Data format (tầng input)

- [x] espeak-ng = text → phonemes lúc pre-cache (quiz đúng)
- [◐] metadata.csv: `wav_name|text` delimiter `|` không header; multi-speaker `wav|speaker|text`; WAV 22050Hz mono PCM16
- [◐] 3 tầng: HF dataset → CSV+WAV → cache (.phonemes.pt / .audio.pt / .spec.pt)

## 4. Broader context

- [◐] Resume semantics: `--ckpt_path` restore weights + optimizer states + global_step; max_steps = base_step + ADDITIONAL_STEPS
- [ ] global_step đếm optimizer steps (VITS 2 optimizers → +2/batch) *(chưa quiz)*
- [ ] Export ONNX + inference flow *(chưa tới — train đang chạy)*

## Session log

- 2026-06-04: fix chain 5 lỗi, train chạy thành công từ step 919580. Quiz: 1 sai → re-teach → đúng; 3 đúng.
## Secret single-speaker remote branch
- [ ] Human can explain why single-speaker metadata is `wav|text`, not `wav|speaker|text`.
- [ ] Human can explain why secret data/artifacts must live under `local/` and never be committed.
- [ ] Human can choose `reference` vs `copy-resample` mode and state the tradeoff.
- [ ] Human can explain why long datasets need duration filtering, batch-size control, and isolated cache dirs.
- [ ] Human can explain why `--ckpt_path` resumes state and why `MAX_STEPS` is absolute, not "additional steps".
- [ ] Human can explain why old HF checkpoints need trusted one-time sanitize before modern PyTorch/Lightning resume.
- [ ] Human can debug these first: bad manifest columns, missing audio, too-long clips, CUDA OOM, missing `monotonic_align`.
- [ ] Verify by restate/quiz before final session close.
## 2x A100 serious-run understanding
- [ ] Human can explain why DDP needs cache prebuild before multi-rank training.
- [ ] Human can calculate effective batch: `BATCH_SIZE * NUM_GPUS * ACCUMULATE_GRAD_BATCHES`.
- [ ] Human can explain why `bf16-mixed` is preferred on A100.
- [ ] Human can explain what `preflight_training.py` checks and why each check prevents expensive failure.
- [ ] Human can explain why `MAX_STEPS` is absolute when resuming from `--ckpt_path`.
- [ ] Human can explain why checkpoint sanitizer is allowed only for trusted HF checkpoint.

## No-sudo remote dependency branch

- [ ] Problem: `sudo apt-get install -y build-essential cmake ninja-build espeak-ng` needs root because it writes system packages into `/usr`/system paths.
- [ ] Why it matters: Piper/phonemization builds often need a C/C++ compiler, CMake, Ninja, and the `espeak-ng` runtime/library.
- [ ] Branch A: Ask admin for system packages when the remote is managed and long-lived.
- [ ] Branch B: Use Conda/Mamba packages in user space when no sudo is allowed.
- [ ] Conda edge case: `espeak-ng` may not exist as a conda package; current Piper embeds `espeak-ng` inside the Python wheel, so do not block on the standalone binary unless a script directly calls `espeak-ng`.
- [ ] Branch C: Build missing tools under `$HOME/.local` only if Conda cannot provide them.
- [ ] Branch D: Use Docker/Apptainer only if the server policy allows containers.
- [ ] Verification: confirm `gcc`, `cmake`, `ninja`, and `espeak-ng` are visible on `PATH`.
- [ ] Pipeline launch condition: run `./run_pipeline.sh` only from an activated env that exposes `gcc`, `g++`, `cmake`, `ninja`, and Python packages; create `pipeline.env` from `pipeline.env.example` before expecting dataset prep to work.
