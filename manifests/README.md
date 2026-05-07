# Dataset Manifests

โครงสร้าง dataset สำหรับ fine-tune ASR ภาษาไทยถิ่น → ภาษาไทยกลาง

## ไฟล์ในโฟลเดอร์นี้

| ไฟล์ | คำอธิบาย |
|---|---|
| `metadata.csv` | Master list ทุกคลิป (5,050 แถว) |
| `train.jsonl` | ชุดเทรน — 4,040 คลิป จาก 40 speakers |
| `val.jsonl` | ชุด validation — 505 คลิป จาก 5 speakers |
| `test.jsonl` | ชุด test — 505 คลิป จาก 5 speakers |
| `dialect_to_central.json` | Lookup table ภาษาถิ่น → ภาษากลาง (50 entries) |

## Schema (ทุกแถวใน .jsonl)

```json
{
  "audio_filepath": "audio_data/south/voice_1/voice_1_1.wav",
  "duration": 3.2,
  "sample_rate": 16000,
  "text_dialect": "พันพรือบ้างช่วงนี้",
  "text_central": "เป็นอย่างไรบ้างช่วงนี้",
  "region": "south",
  "speaker_id": "voice_1",
  "utterance_id": "voice_1_1",
  "is_augmented": false
}
```

## วิธีการแบ่ง Split

แบ่งที่ระดับ `speaker_id` (= utterance_id) ด้วย seed=42 เพื่อให้:

- 80% (40 speakers) → train
- 10% (5 speakers) → val
- 10% (5 speakers) → test

augmented version ของ utterance เดียวกันจะอยู่ split เดียวกันเสมอ ป้องกัน data leakage

## การใช้งาน

### Train แบบ Two-stage (แนะนำ)

1. ใช้ `text_dialect` เป็น label เทรน ASR model
2. ตอน inference ใช้ `dialect_to_central.json` แปลงเป็นภาษากลาง

### Train แบบ End-to-end

ใช้ `text_central` เป็น label โดยตรง (โมเดลเรียน ASR + translation พร้อมกัน)

## ข้อควรระวัง

- `voice_23/ReadMe.txt` มีไบต์ truncated — `text_central` ของคลิปนี้ไม่สมบูรณ์ ("อยากกินผัดส") ควรแก้ไขก่อนเทรน
- Dataset มีแค่ 50 utterance ต่างๆ กัน → เหมาะกับ keyword/command recognition มากกว่า ASR ทั่วไป

## สร้างใหม่

```bash
python scripts/build_manifest.py
```
