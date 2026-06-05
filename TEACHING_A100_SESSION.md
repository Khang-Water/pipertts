# Teaching Checklist — Single-Speaker A100 Fine-tune (2026-06-05)

Đánh dấu ✅ khi đã giải thích lại được bằng lời của mình + trả lời đúng quiz.

## A. Bài toán & bối cảnh
- [x] Vì sao fine-tune từ checkpoint vais1000 thay vì train từ đầu
- [ ] Vì sao dataset để NGOÀI git repo, artifacts trong `local/` (gitignored)
- [ ] Pipeline 9 stage làm gì, vì sao mỗi stage phải idempotent

## B. Resume semantics (lõi của mọi thứ)
- [ ] `--ckpt_path` restore những gì (weights, optimizer, scheduler, global_step, epoch)
- [x] Vì sao `MAX_STEPS=20000` → train 0 step; `EXTRA_STEPS` fix thế nào
- [x] Vì sao KHÔNG reset global_step về 0 (warm-start vs resume, LR 2e-4 vs ~1.1e-4)
- [x] global_step tăng 2/batch (manual optimization, 2 optimizer G+D)
- [x] Target step ghi file 1 lần — vì sao restart không được "moving target"

## C. DDP & dataset
- [ ] `prepare_data()` chạy rank 0 only, `setup()` chạy mọi rank — vì sao crash, patch fallback + barrier
- [x] Vì sao build cache 1 process TRƯỚC khi DDP (race)
- [ ] Bug `if not path:` vs `path.exists()` — Path luôn truthy
- [ ] `shuffle=True` thiếu thì sao (SequentialSampler → học theo thứ tự file)

## D. Mixed precision & CUDA stack
- [ ] bf16 vs fp16 vs fp32 — vì sao A100 dùng bf16
- [x] cuFFT không có kernel bf16 → patch autocast(enabled=False) + .float() quanh mel/STFT
- [x] Driver CUDA (12.2) vs runtime torch (cu130/cu128) — major phải khớp, minor compatibility
- [ ] V100/T4 không có bf16 → PRECISION=16-mixed

## E. Chuỗi 6 lỗi server bring-up (thứ tự gặp)
- [ ] 1. pip `-e /abs/path[extras]` → cd + `-e '.[train]'`
- [ ] 2. piper1-gpl chưa clone → lỗi pip đánh lạc hướng → auto-clone
- [ ] 3. `~/.local` leak vào conda → PYTHONNOUSERSITE=1
- [ ] 4. torch cu130 vs driver 12.2 → cài cu128
- [ ] 5. cuFFT bf16 → patch mel fp32
- [ ] 6. SLURM logger version = job ID → config.yaml đụng → save_config overwrite

## F. Vận hành
- [ ] tmux vs nohup — SIGHUP, container lifetime, vì sao nohup không cứu container
- [ ] Đọc nvidia-smi: memory.used trung bình vs spike (padding theo câu dài nhất batch)
- [ ] Batch 32→40GB ⇒ 48 an toàn, 60 rủi ro OOM — cách ngoại suy tuyến tính + vì sao phải chừa headroom
- [x] Target theo STEP ⇒ batch to hơn = lâu hơn + nhiều data hơn, KHÔNG nhanh hơn
- [ ] OOM cùng seed → chết lặp cùng chỗ

## G. Bảo mật & git hygiene
- [x] .env chứa token → gitignore trước khi init; scan secrets trước push
- [ ] Repo private; fork piper1-gpl chứa patch; pipertts không vendor fork (clone riêng)

---

**Phiên 2026-06-05:** 10/12 quiz đúng. Đã dạy lại (kiểm tra lại phiên sau):
- Chuỗi 6 lỗi server (kể lại được 1/6 — học lại theo 'truyện củ hành')
- cuFFT bf16: chọn 'đổi fp16 toàn cục' thay vì cách ly cục bộ → nguyên tắc: cách ly phép tính lỗi, đừng hạ cấp cả hệ thống
