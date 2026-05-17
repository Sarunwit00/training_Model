"""
สร้างไฟล์ zip ของ Data_Project สำหรับอัพขึ้น Google Drive → ใช้ใน Colab

ใส่เฉพาะ folder ที่จำเป็น (audio_data + colab) และข้าม venv, .git, local_train ฯลฯ
output: C:\\Users\\Sarunwit\\Desktop\\Data_Project.zip

วิธีรัน (เปิด PowerShell หรือ cmd):
    cd C:\\Users\\Sarunwit\\Desktop\\Data_Project
    python zip_for_colab.py
"""

import os
import sys
import time
import zipfile
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────
SOURCE_DIR = Path(r'C:\Users\Sarunwit\Desktop\Data_Project')
OUTPUT_ZIP = Path(r'C:\Users\Sarunwit\Desktop\Data_Project.zip')

# folders ที่จะใส่เข้า zip (skip local_train, venv, .git, manifests, models)
INCLUDE = ['audio_data', 'colab']

# folders/files ที่จะ skip ภายใน INCLUDE (เผื่อมี cache)
SKIP_DIRS = {'__pycache__', '.ipynb_checkpoints', '.git', 'venv', '.venv', 'node_modules'}
SKIP_SUFFIX = {'.pyc', '.tmp'}

# compression: ZIP_DEFLATED level 1 = เร็ว + ลดขนาดได้นิดหน่อย (wav บีบอัดไม่ค่อยได้อยู่แล้ว)
COMPRESSION = zipfile.ZIP_DEFLATED
COMPRESS_LEVEL = 1
# ───────────────────────────────────────────────────────────────────────────


def human_size(n: float) -> str:
    """แปลง bytes → string อ่านง่าย (KB / MB / GB)"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if n < 1024:
            return f'{n:.1f} {unit}'
        n /= 1024
    return f'{n:.1f} TB'


def main() -> None:
    # 1. ตรวจว่ามี source folder อยู่จริง
    if not SOURCE_DIR.exists():
        print(f'X ไม่พบ source folder: {SOURCE_DIR}')
        sys.exit(1)

    for name in INCLUDE:
        d = SOURCE_DIR / name
        if not d.is_dir():
            print(f'X ไม่พบ {d}')
            print(f'   ตรวจว่ามี folder {name}/ ใน {SOURCE_DIR} หรือไม่')
            sys.exit(1)

    print(f'Source : {SOURCE_DIR}')
    print(f'Output : {OUTPUT_ZIP}')
    print(f'Include: {INCLUDE}')
    print(f'Skip   : {sorted(SKIP_DIRS)}')
    print()

    # 2. สแกนไฟล์ทั้งหมด + ขนาดรวม
    print('[1/4] กำลังสแกนไฟล์ ...')
    files_to_zip: list[Path] = []
    total_bytes = 0
    for folder in INCLUDE:
        root_dir = SOURCE_DIR / folder
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for f in files:
                if any(f.endswith(suf) for suf in SKIP_SUFFIX):
                    continue
                fp = Path(root) / f
                try:
                    sz = fp.stat().st_size
                except OSError:
                    continue
                files_to_zip.append(fp)
                total_bytes += sz

    if not files_to_zip:
        print('X ไม่พบไฟล์ใน folders ที่ระบุ')
        sys.exit(1)

    print(f'      พบ {len(files_to_zip):,} ไฟล์  รวม {human_size(total_bytes)}')
    print()

    # 3. ลบไฟล์ zip เก่า (ถ้ามี)
    if OUTPUT_ZIP.exists():
        print(f'[2/4] ลบไฟล์ zip เก่า: {OUTPUT_ZIP}')
        OUTPUT_ZIP.unlink()
    else:
        print('[2/4] ไม่มีไฟล์ zip เก่า — ข้าม')
    print()

    # 4. สร้างไฟล์ zip
    print(f'[3/4] เริ่ม zip ... (compression level {COMPRESS_LEVEL})')
    t0 = time.time()
    bytes_done = 0
    last_print = 0.0

    try:
        with zipfile.ZipFile(
            OUTPUT_ZIP, 'w',
            compression=COMPRESSION,
            compresslevel=COMPRESS_LEVEL,
            allowZip64=True,
        ) as zf:
            for i, fp in enumerate(files_to_zip):
                arcname = fp.relative_to(SOURCE_DIR).as_posix()
                try:
                    zf.write(fp, arcname)
                    bytes_done += fp.stat().st_size
                except (OSError, zipfile.BadZipFile) as e:
                    print(f'   ! ข้าม {fp}: {e}')
                    continue

                # print progress ทุก 2 วินาที หรือไฟล์สุดท้าย
                now = time.time()
                if now - last_print >= 2.0 or i == len(files_to_zip) - 1:
                    pct = 100 * bytes_done / total_bytes if total_bytes else 100
                    elapsed = now - t0
                    speed = bytes_done / elapsed if elapsed > 0 else 0
                    print(
                        f'      {i+1:>7,}/{len(files_to_zip):,}  '
                        f'{pct:>5.1f}%  '
                        f'{human_size(bytes_done):>10}  '
                        f'@ {human_size(speed)}/s'
                    )
                    last_print = now
    except KeyboardInterrupt:
        print('\nX ผู้ใช้ยกเลิก — ลบไฟล์ zip ที่สร้างไม่เสร็จ')
        if OUTPUT_ZIP.exists():
            OUTPUT_ZIP.unlink()
        sys.exit(1)

    elapsed = time.time() - t0
    zip_size = OUTPUT_ZIP.stat().st_size

    print()
    print(f'      zip เสร็จใน {elapsed/60:.1f} นาที')
    print(f'      ขนาด zip    : {human_size(zip_size)}')
    print(f'      ขนาดต้นฉบับ : {human_size(total_bytes)}')
    if total_bytes > 0:
        print(f'      อัตราบีบอัด : {100*zip_size/total_bytes:.1f}% ของต้นฉบับ')
    print()

    # 5. Verify zip
    print('[4/4] ตรวจสอบความถูกต้องของ zip ...')
    with zipfile.ZipFile(OUTPUT_ZIP, 'r') as zf:
        bad = zf.testzip()
        if bad:
            print(f'X พบไฟล์เสียใน zip: {bad}')
            sys.exit(1)
        n_entries = len(zf.namelist())
    print(f'      OK — zip ใช้ได้ ({n_entries:,} entries)')
    print()

    # 6. สรุป + คำแนะนำขั้นต่อไป
    print('=' * 64)
    print(f'เสร็จแล้ว! ไฟล์ zip อยู่ที่:')
    print(f'   {OUTPUT_ZIP}')
    print()
    print('ขั้นต่อไป:')
    print('   1. ไปที่ https://drive.google.com → "ไดรฟ์ของฉัน"')
    print('      (สำคัญ: ต้องวางใน "ไดรฟ์ของฉัน" ไม่ใช่ "คอมพิวเตอร์")')
    print('   2. ลากไฟล์ Data_Project.zip เข้าไปวาง')
    print('   3. รอ upload จบ (~2-10 นาที ขึ้นกับ internet)')
    print('   4. ใน Colab notebook (Cell 1.2) ตั้ง:')
    print("      DRIVE_PATH = '/content/drive/MyDrive/Data_Project.zip'")
    print('=' * 64)


if __name__ == '__main__':
    main()
