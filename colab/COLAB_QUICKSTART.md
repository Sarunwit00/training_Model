# Colab Quickstart

วิธีเตรียมไฟล์และเริ่มเทรนบน Google Colab (multi-dialect: ใต้/เหนือ/อีสาน)

---

## ภาพรวม Workflow

```
[1] เครื่องคุณ           [2] Google Drive          [3] Colab
─────────────────      ────────────────         ──────────────
zip_for_colab.py   →   Data_Project.zip    →   notebook auto-extract
(สร้าง zip)            (อัพโหลด ~3-5 GB)         + auto-resume
```

---

## ขั้นตอน

### 1. สร้างไฟล์ zip บนเครื่อง

เปิด PowerShell (หรือ cmd):

```powershell
cd C:\Users\Sarunwit\Desktop\Data_Project
python zip_for_colab.py
```

Script จะใส่เฉพาะ `audio_data/` + `colab/` (ข้าม `venv`, `models`, `manifests`, `.git`)
ได้ไฟล์ `C:\Users\Sarunwit\Desktop\Data_Project.zip` ขนาดประมาณ **3-5 GB**

### 2. อัพโหลดขึ้น Google Drive

1. เปิด [drive.google.com](https://drive.google.com)
2. ลาก `Data_Project.zip` วางใน **"ไดรฟ์ของฉัน" (My Drive)** ตรงๆ (อย่าใส่ใน sub-folder)
3. รอจน upload จบ (~10-30 นาที ขึ้นกับเน็ต)

> 💡 **ทำไมต้อง zip?** Upload zip ก้อนเดียวเร็วกว่า upload โฟลเดอร์ที่มีไฟล์หลายแสนไฟล์ ~10 เท่า

### 3. เปิด Notebook บน Colab

มี 2 วิธี:

**วิธี A — Upload notebook โดยตรง**
1. เปิด [colab.research.google.com](https://colab.research.google.com)
2. `File → Upload notebook` → เลือก `colab/wav2vec2_finetune_colab.ipynb`

**วิธี B — เปิดจาก Drive**
1. อัพ notebook ขึ้น Drive ด้วย แล้วคลิกขวา → `Open with → Google Colaboratory`

### 4. ตั้ง GPU Runtime

`Runtime → Change runtime type → Hardware accelerator: T4 GPU`

### 5. รัน Cells ตามลำดับ

กด **Shift+Enter** ทีละ cell:

| Cell | ทำอะไร | เวลา (T4) |
|---|---|---|
| 1.1 | Mount Google Drive | ~30 วินาที |
| 1.2 | Extract zip + สร้าง Drive cache | **ครั้งแรก: 30-50 นาที**, รอบหลัง: 3-5 นาที |
| 1.3 | ตรวจ GPU | ทันที |
| 1.4 | ติดตั้ง libraries | ~3 นาที |
| 2.1-2.4 | Build manifest + vocab | ~5 นาที |
| 3.1 | **Sanity check** (แนะนำให้ทำก่อน) | 10-15 นาที |
| 3.2 | **Full training** (8 epochs) | **5-8 ชั่วโมง** |

---

## Features ที่ช่วยให้รันสะดวก (built-in)

### 💾 Drive Cache สำหรับ Dataset
ครั้งแรกที่รัน Cell 1.2 จะ copy dataset ที่ extract แล้วไปไว้บน Drive ด้วย
รอบถัดไปหลัง disconnect จะ copy จาก cache (เร็วกว่า extract zip ~5-10 เท่า)
ใช้พื้นที่ Drive เพิ่ม ~3-5 GB — ตั้ง `USE_CACHE = False` ใน Cell 1.2 ถ้าไม่ต้องการ

### 📦 Checkpoint บน Drive + Auto-Resume
Cell 3.2 บันทึก checkpoint ไปที่ `/content/drive/MyDrive/Data_Project/models/wav2vec2-thai-dialects-v3/` โดยตรง
ไม่หายเมื่อ Colab disconnect — และ Cell 3.2 จะ **detect checkpoint ล่าสุดอัตโนมัติ** แล้ว resume ให้

### 🛑 Early Stopping
หยุดเทรนเองถ้า CER ไม่ดีขึ้น 3 evals ติด (ไม่ต้องเทรนครบ 8 epochs ถ้า converge เร็ว)

### 📊 Clean Progress Display
แสดงบรรทัดเดียวต่อ logging step: step/total, %, epoch, loss + trend, lr, ETA
ตอน eval: CER, WER, loss, best CER (มี ⭐ ถ้าเป็น best ใหม่)

### 🎯 CER เป็น Primary Metric
ภาษาไทยไม่มี space แบ่งคำ — WER จึงเข้มเกินไป Trainer ใช้ CER ตัดสิน best checkpoint แทน

---

## หลัง Disconnect (เกิดบ่อยบน Colab ฟรี)

**ไม่ต้องกังวล** — แค่ทำตามนี้:

1. กด `Connect` ใหม่ (Colab จะให้ runtime ใหม่)
2. รัน Cell 1.1 (mount Drive)
3. รัน Cell 1.2 (จะ copy จาก Drive cache ~3-5 นาที)
4. รัน Cell 1.3, 1.4 (ตรวจ GPU + libraries)
5. ข้ามไปรัน Cell 3.2 — จะ resume จาก checkpoint ล่าสุดบน Drive อัตโนมัติ

ไม่ต้องรัน manifest/vocab ใหม่ เพราะ Drive cache มีอยู่แล้ว

---

## Tips

- **Sanity check ก่อนเสมอ** (Cell 3.1) — 10-15 นาที ถ้าไม่ผ่าน อย่าเสียเวลาเทรนเต็ม
- **Keep-alive** (optional) ใส่ใน browser console (กด F12):
  ```javascript
  setInterval(()=>document.querySelector('colab-toolbar-button#connect')?.click(), 60000)
  ```
- **Save model กลับ Drive** หลังเทรนเสร็จ (Cell 5.1)
- ถ้า **OOM** ลด `--batch_size` ใน Cell 3.2 เหลือ 8 หรือ 4
- ถ้า **loss = NaN** ลด `--lr` เหลือ `5e-5`

---

## ทางเลือก: Kaggle Notebooks

ถ้าไม่อยากเสี่ยง disconnect ของ Colab ฟรี ลอง [kaggle.com/notebooks](https://www.kaggle.com/notebooks)

- ฟรี GPU T4 หรือ P100 30 ชั่วโมง/สัปดาห์
- ไม่ disconnect 90 นาที (ใช้ได้ยาวสุด ~9 ชม. ต่อรอบ)
- Upload dataset ผ่าน Kaggle Datasets

ใช้ notebook ตัวเดียวกันได้ แค่เปลี่ยน Drive mount เป็น Kaggle dataset path

---

## ไฟล์ที่เกี่ยวข้อง

| ไฟล์ | หน้าที่ |
|---|---|
| `zip_for_colab.py` (root) | สร้าง Data_Project.zip สำหรับอัพ Drive |
| `colab/wav2vec2_finetune_colab.ipynb` | Notebook หลักที่รันบน Colab |
| `colab/scripts/train_wav2vec2.py` | Training script (CleanProgressCallback, EarlyStopping, CER metric) |
| `colab/scripts/build_manifest_v3.py` | สร้าง train/val/test manifests (Plan B รองรับ 3 ภาค) |
| `colab/scripts/prepare_vocab.py` | สร้าง vocab.json จาก transcripts |
| `colab/scripts/inference.py` | ใช้โมเดลที่เทรนเสร็จกับไฟล์ใหม่ |
