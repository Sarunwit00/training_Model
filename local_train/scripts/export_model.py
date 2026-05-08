"""
Package โมเดลที่ train เสร็จเป็นไฟล์ zip พร้อมใช้งาน

ฟีเจอร์:
1. Copy เฉพาะไฟล์โมเดลหลัก (ทิ้ง checkpoint ที่ใช้พื้นที่เยอะ)
2. รวม dialect_to_central.json (lookup table)
3. รวม inference scripts (transcribe_file.py, transcribe_mic.py, gui_app.py)
4. รวม README + requirements.txt สำหรับเครื่องปลายทาง
5. Zip ทั้งหมดให้เป็นไฟล์เดียวพร้อมแจกจ่าย

Usage:
    python scripts/export_model.py
    python scripts/export_model.py --model_dir models/wav2vec2-south-th-v2
    python scripts/export_model.py --output exports/my_model.zip
"""

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ไฟล์โมเดล 7 ตัวที่จำเป็นสำหรับ inference
ESSENTIAL_FILES = [
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "special_tokens_map.json",
    "added_tokens.json",
]


def check_model_files(model_dir: Path):
    """ตรวจว่าไฟล์โมเดลครบหรือไม่"""
    missing = [f for f in ESSENTIAL_FILES if not (model_dir / f).exists()]
    if missing:
        print(f"❌ ขาดไฟล์ในโมเดล: {missing}")
        print(f"   Path: {model_dir}")
        return False
    return True


def get_size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / 1024 / 1024
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024 / 1024


def write_inference_app(target_dir: Path):
    """สร้าง inference app ใน folder ปลายทาง"""

    # transcribe_file.py — ถอดเสียงจากไฟล์
    (target_dir / "transcribe_file.py").write_text("""\
\"\"\"ถอดเสียงไฟล์ .wav แล้วแปลงเป็นไทยกลาง\"\"\"
import argparse
import json
from difflib import get_close_matches
from pathlib import Path
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"
LOOKUP = json.load(open(HERE / "dialect_to_central.json", encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="ไฟล์ .wav หรือ folder")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR).to(device).eval()
    print(f"✅ Model loaded on {device}")

    target = Path(args.input)
    files = [target] if target.is_file() else sorted(target.rglob("*.wav"))

    for wav in files:
        audio, _ = librosa.load(str(wav), sr=16000, mono=True)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values.to(device)).logits
        text = processor.batch_decode(torch.argmax(logits, -1))[0]

        if text in LOOKUP:
            central, how = LOOKUP[text]["text_central"], "exact"
        else:
            cand = get_close_matches(text, list(LOOKUP.keys()), n=1, cutoff=0.6)
            central, how = (LOOKUP[cand[0]]["text_central"], f"fuzzy: {cand[0]}") if cand else (None, "no match")

        print(f"\\n📁 {wav}")
        print(f"   Dialect : {text}")
        print(f"   Central : {central}  [{how}]")


if __name__ == "__main__":
    main()
""", encoding="utf-8")

    # transcribe_mic.py — อัดจากไมค์
    (target_dir / "transcribe_mic.py").write_text("""\
\"\"\"อัดเสียงจากไมค์แล้วถอดเสียง\"\"\"
import argparse
import json
import time
from difflib import get_close_matches
from pathlib import Path
import numpy as np
import sounddevice as sd
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"
LOOKUP = json.load(open(HERE / "dialect_to_central.json", encoding="utf-8"))
SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--loop", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR).to(device).eval()
    print(f"✅ Model loaded on {device}\\n")

    def once():
        print(f"🎙️  อัด {args.duration} วิ ...")
        audio = sd.rec(int(args.duration * SR), samplerate=SR, channels=1, dtype="float32")
        sd.wait()
        inputs = processor(audio.flatten(), sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values.to(device)).logits
        text = processor.batch_decode(torch.argmax(logits, -1))[0]

        if text in LOOKUP:
            central = LOOKUP[text]["text_central"]
        else:
            cand = get_close_matches(text, list(LOOKUP.keys()), n=1, cutoff=0.6)
            central = LOOKUP[cand[0]]["text_central"] if cand else None

        print(f"📝 Dialect : {text}")
        print(f"📝 Central : {central}\\n")

    if args.loop:
        try:
            while True:
                once()
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("👋 bye")
    else:
        once()


if __name__ == "__main__":
    main()
""", encoding="utf-8")

    # README.md
    (target_dir / "README.md").write_text("""\
# Thai Dialect Speech Recognition — Inference Package

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### ถอดเสียงไฟล์
```bash
python transcribe_file.py path/to/audio.wav
```

### อัดจากไมค์
```bash
python transcribe_mic.py --duration 4
python transcribe_mic.py --loop   # โหมดต่อเนื่อง
```

## โครงสร้าง

```
.
├── model/                    ← ไฟล์โมเดล 7 ตัว
├── dialect_to_central.json   ← Lookup ใต้→กลาง
├── transcribe_file.py        ← ถอดไฟล์ .wav
├── transcribe_mic.py         ← อัดจากไมค์
├── requirements.txt
└── README.md
```
""", encoding="utf-8")

    # requirements.txt
    (target_dir / "requirements.txt").write_text("""\
torch>=2.1.0
transformers>=4.40.0
librosa>=0.10.0
soundfile>=0.12.0
sounddevice>=0.4.6
numpy>=1.24.0
""", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Export trained model for distribution")
    ap.add_argument(
        "--model_dir",
        default="models/wav2vec2-south-th-v2",
        help="Path ไปยังโฟลเดอร์โมเดลที่ train เสร็จ",
    )
    ap.add_argument(
        "--lookup",
        default="manifests/dialect_to_central.json",
        help="Path ไปยัง lookup table",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Path ของไฟล์ zip ที่จะสร้าง (default: exports/<model_name>.zip)",
    )
    ap.add_argument(
        "--no_zip",
        action="store_true",
        help="สร้างแค่ folder ไม่ zip (เร็วกว่า สำหรับใช้ทันที)",
    )
    args = ap.parse_args()

    model_dir = (ROOT / args.model_dir).resolve()
    lookup_path = (ROOT / args.lookup).resolve()

    # ตรวจ input
    if not model_dir.exists():
        sys.exit(f"❌ ไม่พบโฟลเดอร์โมเดล: {model_dir}")
    if not check_model_files(model_dir):
        sys.exit(1)
    if not lookup_path.exists():
        sys.exit(f"❌ ไม่พบ lookup table: {lookup_path}")

    # ตั้งชื่อ output
    model_name = model_dir.name
    exports_dir = ROOT / "exports"
    exports_dir.mkdir(exist_ok=True)
    staging_dir = exports_dir / model_name
    output_zip = Path(args.output) if args.output else exports_dir / f"{model_name}.zip"

    # ลบ staging เก่า
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    # 1. Copy ไฟล์โมเดล (เฉพาะ 7 ไฟล์หลัก ไม่เอา checkpoint)
    print(f"\n📦 Copy ไฟล์โมเดล {model_name} ...")
    target_model = staging_dir / "model"
    target_model.mkdir()
    for f in ESSENTIAL_FILES:
        src = model_dir / f
        dst = target_model / f
        shutil.copy2(src, dst)
        print(f"   ✅ {f}  ({get_size_mb(src):.2f} MB)")

    # 2. Copy lookup table
    shutil.copy2(lookup_path, staging_dir / "dialect_to_central.json")
    print(f"   ✅ dialect_to_central.json")

    # 3. สร้าง inference app
    print(f"\n📝 สร้าง inference app ...")
    write_inference_app(staging_dir)
    print(f"   ✅ transcribe_file.py")
    print(f"   ✅ transcribe_mic.py")
    print(f"   ✅ requirements.txt")
    print(f"   ✅ README.md")

    # 4. แสดงสรุป
    total_mb = get_size_mb(staging_dir)
    print(f"\n📁 Folder ปลายทาง: {staging_dir}")
    print(f"   ขนาดรวม: {total_mb:.2f} MB")

    # 5. Zip (ถ้าต้องการ)
    if args.no_zip:
        print(f"\n✅ เสร็จสิ้น — folder พร้อมใช้ที่ {staging_dir}")
        return

    print(f"\n🗜️  กำลัง zip → {output_zip} ...")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in staging_dir.rglob("*"):
            if f.is_file():
                arcname = f.relative_to(staging_dir.parent)
                zf.write(f, arcname)

    zip_mb = output_zip.stat().st_size / 1024 / 1024
    print(f"\n✅ เสร็จสิ้น!")
    print(f"   📦 ไฟล์ zip:  {output_zip}")
    print(f"   📊 ขนาด:      {zip_mb:.2f} MB")
    print(f"\n💡 วิธีใช้: ส่งไฟล์ zip ให้คนอื่น แตก zip → pip install -r requirements.txt → python transcribe_file.py audio.wav")


if __name__ == "__main__":
    main()
