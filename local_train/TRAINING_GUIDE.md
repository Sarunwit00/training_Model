# Training Guide (ฉบับละเอียด)

## เนื้อหา

1. ติดตั้ง Python + CUDA
2. Setup โปรเจค
3. รันแต่ละ Step
4. ใช้ VS Code
5. Export โมเดล
6. Troubleshooting

---

## 1. ติดตั้ง Python + CUDA

### Python

ติดตั้ง [Python 3.10 หรือ 3.11](https://www.python.org/downloads/)
- ✅ ติ๊ก **"Add Python to PATH"** ตอนติดตั้ง

### CUDA (สำหรับ NVIDIA GPU)

ตรวจ GPU มี CUDA หรือไม่:
```bash
nvidia-smi
```
ถ้าออกผลตาราง GPU = ใช้ได้ ถ้า command not found = ต้องติดตั้ง driver

ติดตั้ง [NVIDIA CUDA Toolkit 12.1+](https://developer.nvidia.com/cuda-downloads)

### PyTorch (สำคัญ!)

ติดตั้ง PyTorch ที่ตรงกับ CUDA version ของคุณ — ดูที่ [pytorch.org](https://pytorch.org/get-started/locally/)

ตัวอย่าง CUDA 12.1:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

ตรวจสอบ:
```python
import torch
print(torch.cuda.is_available())   # ควรได้ True
print(torch.cuda.get_device_name(0))
```

---

## 2. Setup โปรเจค

### สร้าง venv

```bash
cd local_train
python -m venv venv
```

### Activate venv

```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Windows CMD
venv\Scripts\activate.bat

# Mac/Linux
source venv/bin/activate
```

⚠️ ถ้า PowerShell error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### ติดตั้ง dependencies

```bash
pip install -r requirements.txt
```

### ใส่ data

Option A — Copy เข้ามาทั้งหมด:
```bash
cp -r /path/to/Data_Project/audio_data ./audio_data
```

Option B — Symlink (ดีกว่า ไม่กินพื้นที่ซ้ำ):

Windows (PowerShell as Admin):
```powershell
New-Item -ItemType SymbolicLink -Path audio_data -Target "C:\Users\YourName\Desktop\Data_Project\audio_data"
```

Mac/Linux:
```bash
ln -s /Users/yourname/Desktop/Data_Project/audio_data audio_data
```

ตรวจสอบ:
```bash
ls audio_data/south/  # ควรเห็น voice_1, voice_2, ..., voice_50
```

---

## 3. รันแต่ละ Step

### Step 1: สร้าง Manifest

```bash
python scripts/build_manifest_v2.py
```

**ทำอะไร:** สแกน `audio_data/` แล้วสร้าง:
- `manifests/train.jsonl` (~4,000 คลิป)
- `manifests/val.jsonl` (~500 คลิป)
- `manifests/test.jsonl` (~550 คลิป)
- `manifests/dialect_to_central.json` (50 lookup entries)

**Output ที่คาดหวัง:**
```
train: 4000 clips | 50 unique transcripts
val  :  500 clips | 50 unique transcripts
test :  550 clips | 50 unique transcripts
```

### Step 2: สร้าง Vocabulary

```bash
python scripts/prepare_vocab.py --target text_dialect
```

**ทำอะไร:** อ่านตัวอักษรไทยจาก train set แล้วสร้าง `models/vocab/vocab.json`

**Output:** ตัวอักษรไทย ~48 ตัว + special tokens 3 ตัว = 51 tokens

### Step 3a: Sanity Check (แนะนำมาก!)

```bash
python scripts/train_wav2vec2.py \
    --skip_augmented \
    --epochs 5 \
    --batch_size 4 \
    --output_dir models/sanity-check
```

**ใช้เวลา:** 10–15 นาที (RTX 3060) / 30 นาที (RTX 3050)

**ผ่าน sanity check = pipeline ใช้ได้** ค่อยเทรนเต็ม

### Step 3b: Full Training

```bash
python scripts/train_wav2vec2.py \
    --base_model airesearch/wav2vec2-large-xlsr-53-th \
    --target text_dialect \
    --epochs 30 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr 3e-4 \
    --warmup_steps 500 \
    --eval_steps 200 \
    --save_steps 200 \
    --output_dir models/wav2vec2-south-th-v2
```

**ใช้เวลา:**
- RTX 3060 (12GB): ~5 ชั่วโมง
- RTX 3090 (24GB): ~2.5 ชั่วโมง
- RTX 4090: ~1.5 ชั่วโมง

**Hyperparameters ที่ปรับได้:**

| Flag | ค่าเริ่มต้น | เพิ่ม batch ถ้า | ลด batch ถ้า |
|---|---|---|---|
| `--batch_size` | 4 | GPU 24GB+ → 8 | GPU 8GB → 2 |
| `--gradient_accumulation_steps` | 4 | (เพิ่มเมื่อลด batch_size) | (ลดเมื่อเพิ่ม batch) |
| `--epochs` | 30 | ลองดูว่า val loss ยังลดอยู่ → 50 | overfitting → 20 |
| `--lr` | 3e-4 | loss ลดช้า → 5e-4 | loss ระเบิด → 1e-4 |

### Step 4: Test Inference

```bash
python scripts/inference.py \
    --model models/wav2vec2-south-th-v2 \
    --translate \
    audio_data/south/voice_1/voice_1_1.wav
```

### Step 5: 📦 Export Model

```bash
python scripts/export_model.py
```

ได้ไฟล์ `exports/wav2vec2-south-th-v2.zip` พร้อมแจกจ่าย

---

## 4. ใช้ VS Code

### Setup ครั้งแรก

1. ติดตั้ง [VS Code](https://code.visualstudio.com/)
2. ติดตั้ง Extension: **Python** (Microsoft)
3. `File → Open Folder` → เลือก `local_train/`
4. Terminal → New Terminal → setup venv (ดูข้อ 2)
5. กด `Ctrl+Shift+P` → `Python: Select Interpreter` → เลือก `./venv/Scripts/python.exe`

### Run/Debug

กด `Ctrl+Shift+D` เปิด Run panel แล้วเลือก:

| Configuration | ทำอะไร |
|---|---|
| 🔧 Step 1: Build Manifest | รัน `build_manifest_v2.py` |
| 📚 Step 2: Prepare Vocab | สร้าง vocab |
| 🏋️ Step 3a: Sanity Check | Train 5 epoch |
| 🏋️ Step 3b: Full Training | Train 30 epoch |
| 🎤 Step 4: Test Inference | ทดสอบไฟล์เสียง |
| 📦 Step 5: Export Model | สร้าง zip |

กด **F5** เพื่อรัน

### Tasks (ติดตั้ง deps)

`Ctrl+Shift+P` → `Tasks: Run Task`:
- 🔧 Setup: Install dependencies
- 🧪 Test: Verify setup

---

## 5. Export Model — รายละเอียด

### `export_model.py` ทำอะไร

1. ตรวจไฟล์ 7 ตัวใน `model_dir` ครบหรือไม่
2. Copy เฉพาะไฟล์หลัก (ไม่เอา `checkpoint-*` ที่ใช้พื้นที่ ~3 GB ต่อ checkpoint)
3. Copy `dialect_to_central.json`
4. สร้าง `transcribe_file.py` + `transcribe_mic.py` ใน folder ปลายทาง
5. สร้าง `requirements.txt` + `README.md`
6. Zip ทุกอย่างเป็น `exports/<model_name>.zip`

### โครงสร้างของ zip ที่ได้

```
wav2vec2-south-th-v2/
├── model/
│   ├── config.json
│   ├── model.safetensors           (~1.18 GB)
│   ├── preprocessor_config.json
│   ├── tokenizer_config.json
│   ├── vocab.json
│   ├── special_tokens_map.json
│   └── added_tokens.json
├── dialect_to_central.json
├── transcribe_file.py
├── transcribe_mic.py
├── requirements.txt
└── README.md
```

### วิธีใช้ zip บนเครื่องอื่น

```bash
# unzip
unzip wav2vec2-south-th-v2.zip
cd wav2vec2-south-th-v2

# install deps
pip install -r requirements.txt

# ใช้งาน
python transcribe_file.py audio.wav
python transcribe_mic.py --duration 4
```

ขนาด zip ประมาณ **~1 GB** (compressed)

---

## 6. Troubleshooting

### `CUDA out of memory`

```bash
# ลด batch size
--batch_size 2 --gradient_accumulation_steps 8

# หรือ enable gradient checkpointing (ช้าลง 20-30% แต่ประหยัด VRAM)
# (เปิดอยู่แล้วใน train_wav2vec2.py)
```

### `Loss = NaN`

```bash
# ลด learning rate
--lr 1e-4
```

### `WER ไม่ลด`

- ตรวจ vocab มีตัวอักษรครบ: `cat models/vocab/vocab.json`
- ลอง sanity check ก่อน
- ตรวจว่า manifest ใช้ split ใหม่ (build_manifest_v2.py)

### `Disk เต็ม`

```bash
# ลบ checkpoint เก่า
rm -rf models/wav2vec2-south-th-v2/checkpoint-*
```

### `ModuleNotFoundError: torch`

ตรวจว่า venv activate แล้ว → `which python` ควรชี้ไปที่ venv

### `RuntimeError: ... CUDA mismatch ...`

PyTorch CUDA version ไม่ตรงกับ driver — reinstall:
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Model ไม่ converge บน CPU

CPU เทรนได้แต่ช้ามาก (~10x). แนะนำใช้ Colab T4 ฟรีดีกว่า

### Train เสร็จแล้ว export ไม่ได้

ตรวจไฟล์โมเดลครบ:
```bash
ls models/wav2vec2-south-th-v2/
# ต้องมี: config.json, model.safetensors, vocab.json, ...
```

ถ้าขาด `config.json` แสดงว่าเทรนยังไม่เสร็จสมบูรณ์ — ดู checkpoint:
```bash
ls models/wav2vec2-south-th-v2/checkpoint-*/
# resume training:
python scripts/train_wav2vec2.py ... --resume_from_checkpoint models/wav2vec2-south-th-v2/checkpoint-XXXX
```
