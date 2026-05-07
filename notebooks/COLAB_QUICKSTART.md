# Colab Quickstart

วิธีเริ่มเทรนบน Google Colab

## ขั้นตอน

### 1. เตรียม Google Drive

1. เปิด [drive.google.com](https://drive.google.com)
2. สร้าง folder `Data_Project` ที่ My Drive
3. อัพโหลด **ทั้งโฟลเดอร์ Data_Project** ขึ้นไป (จะมี `audio_data/`, `manifests/`, `scripts/`, ฯลฯ)
4. รอจนอัพโหลดเสร็จ — ขนาดประมาณ 250–300 MB อาจใช้เวลา 5–15 นาที

### 2. เปิด Notebook

มี 2 วิธี:

**วิธี A — Upload notebook ขึ้น Colab โดยตรง**

1. เปิด [colab.research.google.com](https://colab.research.google.com)
2. กด `File → Upload notebook`
3. เลือกไฟล์ `wav2vec2_finetune_colab.ipynb`

**วิธี B — เปิดจาก Drive (ถ้าอัพ notebook ขึ้น Drive ด้วย)**

1. ไปที่ folder ใน Drive
2. คลิกขวาที่ `wav2vec2_finetune_colab.ipynb` → `Open with → Google Colaboratory`

### 3. ตั้ง GPU Runtime

`Runtime → Change runtime type → Hardware accelerator: T4 GPU`

### 4. รัน Cell ตามลำดับ

- กด **Shift+Enter** หรือเครื่องหมาย ▶ ที่แต่ละ cell
- Cell แรกจะถาม permission ให้เข้าถึง Google Drive — กด `Connect to Google Drive`
- ที่ cell #2 ให้ตรวจว่า `DRIVE_PATH` ตรงกับโฟลเดอร์ที่อัพไว้

### 5. Tips

- **ทำ sanity check ก่อน** (cell #6) ใช้เวลาเพียง 10–15 นาที — ถ้าไม่ผ่าน อย่าเสียเวลาเทรนเต็ม
- **Colab ฟรีอาจ disconnect** ระหว่างเทรนเต็ม — ตั้ง `--save_steps 200` ไว้ จะ resume ได้
- **Save model กลับ Drive** ก่อนปิด session เสมอ (cell #11)

## ทางเลือก: ใช้ Kaggle Notebooks

ถ้าไม่อยากใช้ Colab ฟรีเพราะ disconnect บ่อย ลอง [kaggle.com/notebooks](https://www.kaggle.com/notebooks)

- ฟรี GPU T4 หรือ P100 30 ชั่วโมง/สัปดาห์
- ไม่ disconnect 90 นาที
- Upload dataset ผ่าน Kaggle Datasets

ส่วน notebook ใช้ตัวเดียวกันได้ แค่เปลี่ยน Drive mount เป็น Kaggle dataset path
