# =============================================================================
# CELL 6A: Load VIVOS from Kaggle Dataset (ALTERNATIVE TO CELL 6)
# =============================================================================
# Run this cell INSTEAD OF cell 6 if USE_VIVOS_KAGGLE=True

import os, csv, time, shutil
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa

if not USE_VIVOS_KAGGLE:
    print("⚠️  USE_VIVOS_KAGGLE=False → Skip this cell, run cell 6 instead")
else:
    print("=" * 70)
    print("Loading VIVOS from Kaggle Dataset")
    print("=" * 70)

    t0 = time.time()

    # Prepare directories
    if REBUILD_DATA:
        print("REBUILD_DATA=True → removing AUDIO_DIR + CACHE_DIR")
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

    print(f"Found {len(prompts)} prompts in {VIVOS_SPLIT} split")

    # Process audio files
    print(f"\nProcessing audio files (max {MAX_SAMPLES} samples)...")
    n_written = 0
    n_skip_missing = 0
    n_skip_duration = 0
    n_skip_error = 0
    csv_rows = []

    for utt_id, text in prompts.items():
        if n_written >= MAX_SAMPLES:
            break

        # Find WAV file (usually in waves/SPEAKER_ID/utt_id.wav)
        wav_matches = list(waves_dir.glob(f"**/{utt_id}.wav"))
        if not wav_matches:
            n_skip_missing += 1
            continue

        wav_path = wav_matches[0]

        try:
            # Load and resample audio
            audio, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)

            # Check duration
            duration = len(audio) / sr
            if duration < MIN_DURATION_S or duration > MAX_DURATION_S:
                n_skip_duration += 1
                continue

            # Save processed audio
            wav_name = f"vi_{n_written:06d}.wav"
            output_path = Path(AUDIO_DIR) / wav_name
            sf.write(str(output_path), audio, sr, subtype="PCM_16")

            # Add to CSV
            csv_rows.append((wav_name, text))
            n_written += 1

            if n_written % 500 == 0:
                elapsed = time.time() - t0
                print(f"  {n_written}/{MAX_SAMPLES} processed "
                      f"(skip: {n_skip_missing} missing, {n_skip_duration} duration, {n_skip_error} error) "
                      f"elapsed {elapsed:.1f}s")

        except Exception as e:
            n_skip_error += 1
            if n_skip_error <= 5:  # Only print first 5 errors
                print(f"  Error processing {wav_path}: {e}")
            continue

    # Write CSV
    print(f"\nWriting CSV to {CSV_PATH}...")
    with open(CSV_PATH, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='|', quoting=csv.QUOTE_NONE, escapechar='\\')
        for wav_name, text in csv_rows:
            # Clean text: remove pipes, newlines
            clean_text = text.replace('|', ' ').replace('\n', ' ').replace('\r', ' ').strip()
            writer.writerow([wav_name, clean_text])

    elapsed = time.time() - t0
    print(f"\n✅ VIVOS dataset prepared in {elapsed:.1f}s")
    print(f"   Samples written: {n_written}")
    print(f"   Skipped (missing WAV): {n_skip_missing}")
    print(f"   Skipped (duration filter): {n_skip_duration}")
    print(f"   Skipped (errors): {n_skip_error}")
    print(f"   Audio dir: {AUDIO_DIR}")
    print(f"   CSV path: {CSV_PATH}")

    # Show sample
    print("\nFirst 3 lines of CSV:")
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            print(f"  {line.rstrip()}")

    # Stats
    import subprocess
    wav_count = subprocess.check_output(f"ls {AUDIO_DIR} | wc -l", shell=True, text=True).strip()
    audio_size = subprocess.check_output(f"du -sh {AUDIO_DIR}", shell=True, text=True).split()[0]
    print(f"\nTotal WAV files: {wav_count}")
    print(f"Audio directory size: {audio_size}")

    print("\n" + "=" * 70)
    print("✅ Ready for training! Continue to cell 7 (Pre-cache)")
    print("=" * 70)
