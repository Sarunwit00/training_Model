"""
แก้ปัญหา audio file ที่ sample rate ไม่ตรง 16kHz หรือชื่อไฟล์ไม่ standard

ตรวจสอบและแก้ไข:
1. ไฟล์ original ที่ไม่ใช่ 16kHz -> resample ผ่าน ffmpeg
2. ชื่อไฟล์ที่ไม่ standard (เช่น voice_X.wav แทน voice_X_1.wav) -> rename
3. Backup ไฟล์เดิมไว้ที่ audio_data_backup/ ก่อนแก้

ต้องมี ffmpeg ติดตั้งบนเครื่อง:
  Windows: choco install ffmpeg  (หรือ download จาก ffmpeg.org)
  Mac:     brew install ffmpeg
  Linux:   apt install ffmpeg

Usage:
    python scripts/fix_audio.py
    python scripts/fix_audio.py --dry_run  (ดูก่อนว่าจะแก้อะไรบ้าง)
"""

import argparse
import shutil
import subprocess
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "audio_data"
BACKUP_ROOT = ROOT / "audio_data_backup"
TARGET_SR = 16000


def wav_sample_rate(path):
    try:
        with wave.open(str(path), "rb") as w:
            return w.getframerate()
    except Exception:
        return None


def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def resample_to_16k(src, dst):
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-i", str(src),
           "-ar", str(TARGET_SR),
           "-ac", "1",
           "-c:a", "pcm_s16le",
           str(dst)]
    subprocess.run(cmd, check=True)


def find_problems():
    problems = []
    for region_dir in sorted(p for p in DATA_ROOT.iterdir() if p.is_dir()):
        for voice_dir in sorted(p for p in region_dir.iterdir() if p.is_dir()):
            for wav in sorted(voice_dir.glob("*.wav")):
                if "_aug_" in wav.name:
                    continue  # only check originals
                sr = wav_sample_rate(wav)
                if sr is None:
                    continue
                expected_name = f"{voice_dir.name}_1.wav"
                needs_fix = sr != TARGET_SR or wav.name != expected_name
                if needs_fix:
                    target = voice_dir / expected_name
                    problems.append((wav, target, sr))
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true",
                    help="Show what would be fixed without making changes")
    args = ap.parse_args()

    if not DATA_ROOT.exists():
        sys.exit(f"ERROR: {DATA_ROOT} not found")

    if not check_ffmpeg():
        sys.exit("ERROR: ffmpeg not found.\n"
                 "  Install: https://ffmpeg.org/download.html\n"
                 "  Or: choco install ffmpeg (Windows)\n"
                 "      brew install ffmpeg (Mac)\n"
                 "      apt install ffmpeg (Linux)")

    problems = find_problems()
    if not problems:
        print("OK: all original files are 16kHz and properly named")
        return

    print(f"Found {len(problems)} files to fix:\n")
    for src, target, sr in problems:
        action = []
        if sr != TARGET_SR:
            action.append(f"resample {sr}->16k")
        if src.name != target.name:
            action.append(f"rename to {target.name}")
        print(f"  {src.relative_to(DATA_ROOT)} : {' + '.join(action)}")

    if args.dry_run:
        print("\n(dry run - no changes made)")
        return

    print(f"\nBackup directory: {BACKUP_ROOT}")
    n_done = 0
    for src, target, sr in problems:
        rel = src.relative_to(DATA_ROOT)
        backup_path = BACKUP_ROOT / rel
        backup_path.parent.mkdir(parents=True, exist_ok=True)

        if not backup_path.exists():
            shutil.copy2(src, backup_path)

        if target.exists() and target != src:
            tsr = wav_sample_rate(target)
            if tsr == TARGET_SR:
                src.unlink()
                print(f"  [skip] {rel} -> target already 16k, removed misnamed source")
                continue

        if sr != TARGET_SR:
            tmp = target.with_suffix(".tmp.wav")
            resample_to_16k(src, tmp)
            tmp.replace(target)
            if src.exists() and src != target:
                src.unlink()
        elif src != target:
            src.rename(target)
        n_done += 1
        print(f"  [fixed] {rel}")

    print(f"\nDone: {n_done} files fixed")


if __name__ == "__main__":
    main()
