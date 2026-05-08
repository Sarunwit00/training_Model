# Local Training (รันบนเครื่องตัวเอง)

โปรเจคนี้ทำให้คุณสามารถ fine-tune โมเดล Wav2Vec2 บนเครื่องตัวเอง — เหมือน Colab notebook แต่ไม่ต้องพึ่ง cloud และมีฟังก์ชัน **Export โมเดลเป็น zip พร้อมใช้** หลังเทรนเสร็จ

## ⚙️ ความต้องการ

- **GPU**: NVIDIA GPU + CUDA (แนะนำ 12 GB+ VRAM)
- **Python**: 3.10 หรือ 3.11
- **OS**: Windows / Mac / Linux
- **Disk**: ~15 GB ว่าง (โมเดล + checkpoint + dependencies)

> **CPU เทรนได้ไหม?** ได้ แต่ช้ามาก (10–20 ชั่วโมงสำหรับ 30 epochs) ใช้ Colab T4 ฟรีจะเร็วกว่า

## 📁 โครงสร้างโฟลเดอร์

```
local_train/
├── README.md                 ← คุณอยู่ที่นี่
├── TRAINING_GUIDE.md         ← คู่มือเทรนละเอียด (ตอน setup)
├── requirements.txt
├── .vscode/                  ← ตั้งค่า VS Code (Run/Debug ครบ)
├── scripts/
│   ├── build_manifest_v2.py  ← Step 1: สร้าง manifest
│   ├── prepare_vocab.py      ← Step 2: สร้าง vocab
│   ├── train_wav2vec2.py     ← Step 3: เทรนโมเดล
│   ├── inference.py          ← Step 4: ทดสอบ
│   └── export_model.py       ← Step 5: 📦 Pack เป็น zip ⭐
├── audio_data/               ← ⚠️ ใส่ data ที่นี่ (หรือ symlink)
├── manifests/                ← (auto-generated)
├── models/                   ← (auto-generated, output ของการเทรน)
└── exports/                  ← (auto-generated, zip พร้อมแจก)
```

## 🚀 ขั้นตอน Quick Start

### 1. ติดตั้ง dependencies

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1     # Windows
# source venv/bin/activate      # Mac/Linux
pip install -r requirements.txt
```

### 2. ใส่ Dataset

Copy หรือ symlink โฟลเดอร์ `audio_data/` ของคุณมาไว้ใน `local_train/audio_data/`:

```
local_train/audio_data/
└── south/
    ├── voice_1/
    │   ├── ReadMe.txt
    │   ├── voice_1_1.wav
    │   └── voice_1_1_aug_1.wav  ... voice_1_1_aug_100.wav
    ├── voice_2/
    └── ...
```

หรือใช้ symlink (Windows ใช้ Junction):
```bash
# Windows (PowerShell, run as admin)
New-Item -ItemType SymbolicLink -Path audio_data -Target "C:\path\to\Data_Project\audio_data"

# Mac/Linux
ln -s /path/to/Data_Project/audio_data audio_data
```

### 3. รันตามลำดับ

```bash
# Step 1: สร้าง manifest (~1 นาที)
python scripts/build_manifest_v2.py

# Step 2: สร้าง vocabulary (~5 วินาที)
python scripts/prepare_vocab.py --target text_dialect

# Step 3: Sanity check (~10–15 นาที — ทดสอบว่า pipeline ใช้ได้)
python scripts/train_wav2vec2.py \
    --skip_augmented \
    --epochs 5 \
    --batch_size 4 \
    --output_dir models/sanity-check

# Step 4: เทรนเต็ม (~3–5 ชั่วโมงบน RTX 3060+)
python scripts/train_wav2vec2.py \
    --base_model airesearch/wav2vec2-large-xlsr-53-th \
    --target text_dialect \
    --epochs 30 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr 3e-4 \
    --output_dir models/wav2vec2-south-th-v2

# Step 5: ทดสอบ inference (optional)
python scripts/inference.py \
    --model models/wav2vec2-south-th-v2 \
    --translate \
    audio_data/south/voice_1/voice_1_1.wav

# Step 6: 📦 Export โมเดลเป็น zip พร้อมใช้
python scripts/export_model.py
```

หลัง Step 6 จะได้ไฟล์ `exports/wav2vec2-south-th-v2.zip` ที่บรรจุทุกอย่างพร้อมแจก

## 📦 ฟังก์ชัน Export Model (`export_model.py`)

หลังเทรนเสร็จ รันคำสั่งเดียวจะได้ zip พร้อมใช้:

```bash
python scripts/export_model.py
```

zip ที่ได้จะมี:
- ✅ ไฟล์โมเดลหลัก 7 ตัว (config.json, model.safetensors, ฯลฯ)
- ✅ Lookup table (dialect_to_central.json)
- ✅ Inference scripts (transcribe_file.py, transcribe_mic.py)
- ✅ requirements.txt + README.md
- ❌ ไม่รวม checkpoint (ลดขนาดจาก ~5 GB → ~1.2 GB)

ผู้รับ zip แค่:
```bash
unzip wav2vec2-south-th-v2.zip
cd wav2vec2-south-th-v2
pip install -r requirements.txt
python transcribe_file.py audio.wav
```

### Options ของ export_model.py

```bash
# เลือก model อื่น
python scripts/export_model.py --model_dir models/sanity-check

# กำหนดชื่อ zip
python scripts/export_model.py --output exports/my_model_v1.zip

# ไม่ zip (เก็บเป็น folder)
python scripts/export_model.py --no_zip
```

## 🎯 ตัวเลือก: ใช้ VS Code

ดู `TRAINING_GUIDE.md` สำหรับวิธี setup VS Code ครบทุก step

VS Code มี Run/Debug configs สำเร็จ:
- 🔧 Step 1: Build Manifest
- 📚 Step 2: Prepare Vocab
- 🏋️ Step 3a: Sanity Check
- 🏋️ Step 3b: Full Training
- 🎤 Step 4: Test Inference
- 📦 Step 5: Export Model

แค่กด F5 แล้วเลือก step ที่ต้องการ

## ⚠️ ความแตกต่างจาก Colab

| ประเด็น | Colab | Local |
|---|---|---|
| GPU | ฟรี (T4 16GB) | ใช้ของตัวเอง |
| Disconnect | บ่อย (90 นาที) | ไม่ disconnect |
| Setup | ติดตั้งใหม่ทุกครั้ง | ติดตั้งครั้งเดียว |
| ไฟล์ persistence | หายเมื่อปิด | อยู่ถาวร |
| Export model | ต้อง copy กลับ Drive | มี script ทำให้ ⭐ |

## 🆘 ถ้าเจอปัญหา

ดู [TRAINING_GUIDE.md](TRAINING_GUIDE.md) ส่วน Troubleshooting
