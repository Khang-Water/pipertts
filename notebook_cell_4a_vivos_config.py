# =============================================================================
# 🎛️  VIVOS KAGGLE DATASET CONFIGURATION
# =============================================================================
# Thêm cell này vào sau cell 4 (configuration) để sử dụng VIVOS từ Kaggle dataset

USE_VIVOS_KAGGLE = True  # Set True để dùng VIVOS từ Kaggle dataset thay vì HuggingFace

# VIVOS Kaggle dataset path
VIVOS_KAGGLE_ROOT = "/kaggle/input/datasets/kynthesis/vivos-vietnamese-speech-corpus-for-asr/vivos"
VIVOS_SPLIT = "train"  # "train" hoặc "test"

if USE_VIVOS_KAGGLE:
    print("=" * 70)
    print("🇻🇳 Using VIVOS from Kaggle Dataset")
    print("=" * 70)

    from pathlib import Path
    import json

    # Validate VIVOS path
    vivos_root = Path(VIVOS_KAGGLE_ROOT)
    train_dir = vivos_root / VIVOS_SPLIT
    waves_dir = train_dir / "waves"
    prompts_path = train_dir / "prompts.txt"

    if not waves_dir.exists():
        raise FileNotFoundError(
            f"VIVOS waves directory not found at {waves_dir}\n"
            f"Please add VIVOS dataset to Kaggle:\n"
            f"  1. Go to: https://www.kaggle.com/datasets/kynthesis/vivos-vietnamese-speech-corpus-for-asr\n"
            f"  2. Click 'Add Data' in your notebook\n"
            f"  3. Search for 'vivos-vietnamese-speech-corpus-for-asr'"
        )

    if not prompts_path.exists():
        raise FileNotFoundError(f"prompts.txt not found at {prompts_path}")

    print(f"✅ Found VIVOS dataset:")
    print(f"   Root: {vivos_root}")
    print(f"   Waves: {waves_dir}")
    print(f"   Prompts: {prompts_path}")

    # Count total samples
    total_prompts = sum(1 for line in prompts_path.read_text(encoding="utf-8").splitlines() if line.strip())
    print(f"   Total prompts: {total_prompts}")

    # Override dataset settings
    print(f"\n⚠️  Overriding dataset settings for VIVOS:")
    print(f"   MAX_SAMPLES: {MAX_SAMPLES} (from {total_prompts} available)")
    print(f"   SAMPLE_RATE: {SAMPLE_RATE} Hz")
    print(f"   Duration filter: {MIN_DURATION_S}s - {MAX_DURATION_S}s")

    # Skip cell 6 warning
    print(f"\n⚠️  IMPORTANT: SKIP CELL 6 (HuggingFace dataset loading)")
    print(f"   Run CELL 6A (VIVOS Kaggle loader) instead")
else:
    print("USE_VIVOS_KAGGLE=False → will use HuggingFace dataset (run cell 6)")
