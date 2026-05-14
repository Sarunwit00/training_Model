"""One-shot patcher: update Cell 1.2 to auto-detect folder vs zip upload.

Already applied. Safe to delete.
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "wav2vec2_finetune_colab.ipynb"
nb = json.load(open(NB_PATH, "r", encoding="utf-8"))


def set_cell(idx, lines):
    nb["cells"][idx]["source"] = lines


# ── Cell 4 (markdown for Cell 1.2) ─────────────────────────────────────────
set_cell(4, [
    "## Cell 1.2 — กำหนด Path และเตรียม Data บน Colab Disk\n",
    "\n",
    "⚠️ **ต้องแก้:** ตั้ง `DRIVE_PATH` ให้ตรงกับสิ่งที่อัพไว้ใน Drive\n",
    "\n",
    "Cell นี้ **auto-detect** ได้ทั้ง 2 แบบ:\n",
    "\n",
    "| รูปแบบ | DRIVE_PATH | คำแนะนำ |\n",
    "|---|---|---|\n",
    "| 📁 **โฟลเดอร์** (audio_data + colab) | `/content/drive/MyDrive/Data_Project` | ง่าย แต่ upload ช้า (มีไฟล์เยอะ) |\n",
    "| 🗜️ **ไฟล์ zip** (แนะนำ) | `/content/drive/MyDrive/Data_Project.zip` | upload เร็วกว่า 10–30 เท่า |\n",
    "\n",
    "### โครงสร้างที่ Cell นี้คาดหวัง\n",
    "\n",
    "ไม่ว่า DRIVE_PATH จะเป็นโฟลเดอร์หรือ zip ภายในต้องมี:\n",
    "```\n",
    "<root>/\n",
    "├── audio_data/     ← จำเป็น (ไฟล์เสียง)\n",
    "└── colab/          ← จำเป็น (scripts + notebook)\n",
    "```\n",
    "ส่วน `manifests/` และ `models/` ไม่ต้องอัพ — script จะสร้างให้เอง\n",
    "\n",
    "### ทำไมต้อง copy/extract มาที่ Colab disk?\n",
    "\n",
    "Colab อ่านไฟล์จาก Drive ช้ามาก (ช้ากว่า local disk ~10 เท่า) การคัดลอกหรือแตก zip มาที่ `/content/` ก่อนทำให้เทรนเร็วขึ้นมาก"
])


# ── Cell 5 (code for Cell 1.2) ─────────────────────────────────────────────
set_cell(5, [
    "# ⚠️ แก้ DRIVE_PATH ให้ตรงกับสิ่งที่อัพไว้ใน Drive\n",
    "# - ถ้าอัพเป็น folder: '/content/drive/MyDrive/Data_Project'\n",
    "# - ถ้าอัพเป็น zip   : '/content/drive/MyDrive/Data_Project.zip'\n",
    "DRIVE_PATH = '/content/drive/MyDrive/Data_Project'\n",
    "\n",
    "# โฟลเดอร์ทำงานบน Colab disk (เร็วกว่า Drive มาก)\n",
    "WORK_DIR = '/content/Data_Project'\n",
    "\n",
    "import os, shutil, zipfile, time\n",
    "\n",
    "def _find_project_root(start_dir):\n",
    "    \"\"\"หาโฟลเดอร์ที่มี audio_data/ + colab/ อยู่ภายใน (สำหรับกรณี zip มี nested folder)\"\"\"\n",
    "    from pathlib import Path\n",
    "    root = Path(start_dir)\n",
    "    if (root / 'audio_data').is_dir() and (root / 'colab').is_dir():\n",
    "        return str(root)\n",
    "    for sub in root.iterdir():\n",
    "        if sub.is_dir() and (sub / 'audio_data').is_dir() and (sub / 'colab').is_dir():\n",
    "            return str(sub)\n",
    "    return None\n",
    "\n",
    "if os.path.exists(WORK_DIR) and os.path.isdir(WORK_DIR) and os.listdir(WORK_DIR):\n",
    "    print(f'✅ Dataset อยู่ที่ {WORK_DIR} แล้ว (ข้าม copy/extract)')\n",
    "elif DRIVE_PATH.endswith('.zip'):\n",
    "    # ── โหมด zip: extract มาที่ Colab disk ────────────────────────────────\n",
    "    if not os.path.exists(DRIVE_PATH):\n",
    "        raise FileNotFoundError(f'ไม่พบไฟล์ zip ที่ {DRIVE_PATH} — แก้ DRIVE_PATH ให้ถูก')\n",
    "    size_mb = os.path.getsize(DRIVE_PATH) / 1e6\n",
    "    print(f'🗜️  ตรวจพบไฟล์ zip ({size_mb:,.0f} MB) — กำลังแตก ...')\n",
    "    t0 = time.time()\n",
    "    extract_to = '/content/_extract_tmp'\n",
    "    if os.path.exists(extract_to):\n",
    "        shutil.rmtree(extract_to)\n",
    "    os.makedirs(extract_to)\n",
    "    with zipfile.ZipFile(DRIVE_PATH, 'r') as zf:\n",
    "        zf.extractall(extract_to)\n",
    "    # หา project root ใน zip (zip อาจมี Data_Project/ ครอบ หรือไม่ก็ได้)\n",
    "    root = _find_project_root(extract_to)\n",
    "    if root is None:\n",
    "        raise RuntimeError(\n",
    "            f'แตก zip แล้วไม่พบ audio_data/ + colab/ — ตรวจโครงสร้าง zip:\\n'\n",
    "            f'  zip ต้องมี audio_data/ และ colab/ อยู่ใน root หรือใน subfolder ชั้นเดียว'\n",
    "        )\n",
    "    if root != WORK_DIR:\n",
    "        shutil.move(root, WORK_DIR)\n",
    "    # ลบ tmp ที่เหลือ\n",
    "    if os.path.exists(extract_to):\n",
    "        shutil.rmtree(extract_to, ignore_errors=True)\n",
    "    print(f'✅ แตก zip เสร็จใน {time.time()-t0:.0f} วินาที')\n",
    "elif os.path.isdir(DRIVE_PATH):\n",
    "    # ── โหมด folder: copy จาก Drive ──────────────────────────────────────\n",
    "    if _find_project_root(DRIVE_PATH) is None:\n",
    "        raise RuntimeError(\n",
    "            f'{DRIVE_PATH} ไม่มี audio_data/ + colab/ — ตรวจว่าอัพถูกตำแหน่ง'\n",
    "        )\n",
    "    print(f'📁 ตรวจพบโฟลเดอร์ — กำลัง copy {DRIVE_PATH} → {WORK_DIR}')\n",
    "    print('   (อาจใช้เวลา 15–30 นาที สำหรับ data ~3–5 GB)')\n",
    "    t0 = time.time()\n",
    "    shutil.copytree(DRIVE_PATH, WORK_DIR)\n",
    "    print(f'✅ Copy เสร็จใน {(time.time()-t0)/60:.1f} นาที')\n",
    "else:\n",
    "    raise FileNotFoundError(\n",
    "        f'ไม่พบ {DRIVE_PATH}\\n'\n",
    "        f'  • ถ้าใช้ folder: ตรวจว่าโฟลเดอร์มีอยู่จริงและมี audio_data/ + colab/\\n'\n",
    "        f'  • ถ้าใช้ zip:    ตรวจว่าไฟล์ .zip มีอยู่และ path ลงท้ายด้วย .zip'\n",
    "    )\n",
    "\n",
    "os.chdir(WORK_DIR)\n",
    "print(f'\\n📂 ตำแหน่งงาน: {os.getcwd()}')\n",
    "print(f'\\nไฟล์ภายใน:')\n",
    "!ls\n",
    "\n",
    "# ตรวจขั้นต้น\n",
    "for required in ['audio_data', 'colab']:\n",
    "    if not os.path.isdir(required):\n",
    "        raise RuntimeError(f'ขาดโฟลเดอร์ {required}/ — pipeline ใช้ไม่ได้')\n",
    "print('\\n✅ พบ audio_data/ + colab/ พร้อมเริ่ม')"
])


with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Patched Cell 1.2: {NB_PATH}")
