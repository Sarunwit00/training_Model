"""
Package โมเดลที่ train เสร็จเป็นไฟล์ zip พร้อมใช้งาน

ฟีเจอร์:
1. ตรวจไฟล์ครบและ trained จริง (ไม่ใช่แค่ checkpoint)
2. Copy ไฟล์โมเดลหลัก (ทิ้ง checkpoint ที่ใช้พื้นที่เยอะ)
3. รวม dialect_to_central.json
4. รวม inference scripts (transcribe_file.py, transcribe_mic.py)
5. รวม README + requirements.txt
6. Zip เป็นไฟล์เดียวพร้อมแจกจ่าย

Usage:
    python scripts/export_model.py
    python scripts/export_model.py --model_dir models/wav2vec2-south-th-v2
    python scripts/export_model.py --output exports/my_model.zip
    python scripts/export_model.py --no_zip
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ESSENTIAL_FILES = [
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "special_tokens_map.json",
    "added_tokens.json",
]


def check_model_files(model_dir):
    missing = [f for f in ESSENTIAL_FILES if not (model_dir / f).exists()]
    if missing:
        print(f"ERROR: missing files in {model_dir}: {missing}")
        # Hint about checkpoints
        ckpts = sorted(model_dir.glob("checkpoint-*"))
        if ckpts:
            print(f"  Hint: found {len(ckpts)} checkpoint(s):")
            for c in ckpts[-3:]:
                ok = (c / "config.json").exists() and (c / "model.safetensors").exists()
                print(f"    {c.name}  {'(complete)' if ok else '(incomplete)'}")
            print(f"  Use --model_dir {ckpts[-1]} to export from a checkpoint")
        return False
    return True


def get_size_mb(path):
    if path.is_file():
        return path.stat().st_size / 1024 / 1024
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024 / 1024


def write_inference_app(target_dir):
    """สร้างไฟล์ inference app ใน target_dir"""

    # transcribe_file.py
    transcribe_file_code = '''"""ถอดเสียงไฟล์ .wav แล้วแปลงเป็นไทยกลาง"""
import argparse
import json
import sys
from difflib import get_close_matches
from pathlib import Path
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"
LOOKUP_PATH = HERE / "dialect_to_central.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="ไฟล์ .wav หรือ folder")
    args = ap.parse_args()

    if not MODEL_DIR.exists():
        sys.exit(f"ERROR: model folder not found: {MODEL_DIR}")
    if not LOOKUP_PATH.exists():
        print(f"WARN: lookup not found, central translation disabled")
        lookup = {}
    else:
        lookup = json.load(open(LOOKUP_PATH, encoding="utf-8"))

    target = Path(args.input)
    if not target.exists():
        sys.exit(f"ERROR: input not found: {target}")
    files = [target] if target.is_file() else sorted(target.rglob("*.wav"))
    if not files:
        sys.exit(f"ERROR: no .wav files at {target}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR).to(device).eval()
    print(f"Model loaded on {device}")

    for wav in files:
        audio, _ = librosa.load(str(wav), sr=16000, mono=True)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values.to(device)).logits
        text = processor.batch_decode(torch.argmax(logits, -1))[0]

        if text in lookup:
            central, how = lookup[text]["text_central"], "exact"
        else:
            cand = get_close_matches(text, list(lookup.keys()), n=1, cutoff=0.6)
            central, how = (lookup[cand[0]]["text_central"], f"fuzzy: {cand[0]}") if cand else (None, "no match")

        print(f"\\nFile: {wav}")
        print(f"  Dialect : {text}")
        print(f"  Central : {central}  [{how}]")


if __name__ == "__main__":
    main()
'''
    (target_dir / "transcribe_file.py").write_text(transcribe_file_code, encoding="utf-8")

    # transcribe_mic.py
    transcribe_mic_code = '''"""อัดเสียงจากไมค์แล้วถอดเสียง"""
import argparse
import json
import sys
import time
from difflib import get_close_matches
from pathlib import Path
import numpy as np
import sounddevice as sd
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"
LOOKUP_PATH = HERE / "dialect_to_central.json"
SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--loop", action="store_true")
    args = ap.parse_args()

    if not MODEL_DIR.exists():
        sys.exit(f"ERROR: model folder not found: {MODEL_DIR}")
    lookup = json.load(open(LOOKUP_PATH, encoding="utf-8")) if LOOKUP_PATH.exists() else {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR).to(device).eval()
    print(f"Model loaded on {device}\\n")

    def once():
        print(f"Recording {args.duration}s ...")
        audio = sd.rec(int(args.duration * SR), samplerate=SR, channels=1, dtype="float32")
        sd.wait()
        inputs = processor(audio.flatten(), sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values.to(device)).logits
        text = processor.batch_decode(torch.argmax(logits, -1))[0]

        if text in lookup:
            central = lookup[text]["text_central"]
        else:
            cand = get_close_matches(text, list(lookup.keys()), n=1, cutoff=0.6)
            central = lookup[cand[0]]["text_central"] if cand else None

        print(f"  Dialect : {text}")
        print(f"  Central : {central}\\n")

    if args.loop:
        try:
            while True:
                once()
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("bye")
    else:
        once()


if __name__ == "__main__":
    main()
'''
    (target_dir / "transcribe_mic.py").write_text(transcribe_mic_code, encoding="utf-8")

    # README.md
    readme = '''# Thai Dialect Speech Recognition

โมเดล Wav2Vec2 ที่ fine-tune สำหรับถอดเสียงภาษาไทยถิ่นใต้

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# ถอดเสียงไฟล์
python transcribe_file.py path/to/audio.wav

# อัดจากไมค์
python transcribe_mic.py
python transcribe_mic.py --loop  # โหมดต่อเนื่อง
```

## Files

```
.
├── model/                    7 model files
├── dialect_to_central.json   lookup table
├── transcribe_file.py
├── transcribe_mic.py
├── requirements.txt
└── README.md
```
'''
    (target_dir / "README.md").write_text(readme, encoding="utf-8")

    # requirements.txt
    reqs = '''torch>=2.1.0
transformers>=4.40.0
librosa>=0.10.0
soundfile>=0.12.0
sounddevice>=0.4.6
numpy>=1.24.0
'''
    (target_dir / "requirements.txt").write_text(reqs, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="models/wav2vec2-south-th-v2",
                    help="Path to trained model directory")
    ap.add_argument("--lookup", default="manifests/dialect_to_central.json",
                    help="Path to lookup table")
    ap.add_argument("--output", default=None,
                    help="Output zip path (default: exports/<model_name>.zip)")
    ap.add_argument("--no_zip", action="store_true",
                    help="Skip zipping (keep folder only)")
    args = ap.parse_args()

    model_dir = (ROOT / args.model_dir).resolve() if not Path(args.model_dir).is_absolute() else Path(args.model_dir).resolve()
    lookup_path = (ROOT / args.lookup).resolve() if not Path(args.lookup).is_absolute() else Path(args.lookup).resolve()

    if not model_dir.exists():
        sys.exit(f"ERROR: model dir not found: {model_dir}")
    if not check_model_files(model_dir):
        sys.exit(1)
    if not lookup_path.exists():
        sys.exit(f"ERROR: lookup not found: {lookup_path}\n"
                 f"  Run: python scripts/build_manifest_v2.py first")

    model_name = model_dir.name
    exports_dir = ROOT / "exports"
    exports_dir.mkdir(exist_ok=True)
    staging_dir = exports_dir / model_name
    output_zip = Path(args.output) if args.output else exports_dir / f"{model_name}.zip"

    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    # 1. Copy model files
    print(f"\nCopying model files from {model_name} ...")
    target_model = staging_dir / "model"
    target_model.mkdir()
    for f in ESSENTIAL_FILES:
        src = model_dir / f
        dst = target_model / f
        shutil.copy2(src, dst)
        print(f"  OK {f}  ({get_size_mb(src):.2f} MB)")

    # 2. Copy lookup
    shutil.copy2(lookup_path, staging_dir / "dialect_to_central.json")
    print(f"  OK dialect_to_central.json")

    # 3. Inference app
    print(f"\nWriting inference app ...")
    write_inference_app(staging_dir)
    print(f"  OK transcribe_file.py")
    print(f"  OK transcribe_mic.py")
    print(f"  OK requirements.txt")
    print(f"  OK README.md")

    total_mb = get_size_mb(staging_dir)
    print(f"\nStaging dir: {staging_dir}")
    print(f"Total size:  {total_mb:.2f} MB")

    if args.no_zip:
        print(f"\nDone (folder only): {staging_dir}")
        return

    # 4. Zip
    print(f"\nZipping -> {output_zip} ...")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in staging_dir.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(staging_dir.parent))

    zip_mb = output_zip.stat().st_size / 1024 / 1024
    print(f"\nDone!")
    print(f"  zip:  {output_zip}")
    print(f"  size: {zip_mb:.2f} MB")
    print(f"\nUsage on target machine:")
    print(f"  unzip {output_zip.name}")
    print(f"  cd {model_name}")
    print(f"  pip install -r requirements.txt")
    print(f"  python transcribe_file.py audio.wav")


if __name__ == "__main__":
    main()
