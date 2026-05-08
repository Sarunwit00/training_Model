"""
Build dataset manifests with AUGMENTATION-BASED split (Path C strategy).

ต่างจาก build_manifest.py ที่ split โดย speaker:
  - speaker split: model เห็น utterance N ใน train เท่านั้น
                   → ไม่เคยเห็น utterance ใน test ได้ CER สูง
  - augmentation split: model เห็นทุก utterance ทุก speaker
                        แต่ test ใช้ augmented version ที่ไม่เคยเห็น
                        → CER ต่ำ เหมาะสำหรับ closed-set evaluation

Output (overwrites manifests/*.jsonl):
  - manifests/train.jsonl
  - manifests/val.jsonl
  - manifests/test.jsonl
  - manifests/metadata.csv
  - manifests/dialect_to_central.json

Usage:
    python scripts/build_manifest_v2.py
"""

import csv
import json
import random
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "audio_data"
OUT_ROOT = ROOT / "manifests"
OUT_ROOT.mkdir(exist_ok=True)

SEED = 42
TRAIN_RATIO = 0.80   # 80 of 101 clips per voice → train
VAL_RATIO = 0.10     # 10 → val
# test = remaining 10 ~


def parse_readme(path: Path):
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    if "�" in text:
        print(f"[warn] {path} contains undecodable bytes")
    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        left, right = [p.strip() for p in line.split("=", 1)]
        if left.lower() == "transcript" and right.lower() == "translation":
            continue
        return left, right
    return None, None


def wav_meta(path: Path):
    with wave.open(str(path), "rb") as w:
        return {
            "duration": round(w.getnframes() / w.getframerate(), 3),
            "sample_rate": w.getframerate(),
        }


def collect_samples():
    samples = []
    for region_dir in sorted(p for p in DATA_ROOT.iterdir() if p.is_dir()):
        region = region_dir.name
        for voice_dir in sorted(p for p in region_dir.iterdir() if p.is_dir()):
            speaker_id = voice_dir.name
            readme = voice_dir / "ReadMe.txt"
            if not readme.exists():
                continue
            text_dialect, text_central = parse_readme(readme)
            if not text_dialect:
                continue

            for wav in sorted(voice_dir.glob("*.wav")):
                meta = wav_meta(wav)
                is_aug = "_aug_" in wav.name
                utt_id = wav.stem.split("_aug_")[0]
                rel_path = wav.relative_to(ROOT).as_posix()
                samples.append(
                    {
                        "audio_filepath": rel_path,
                        "duration": meta["duration"],
                        "sample_rate": meta["sample_rate"],
                        "text_dialect": text_dialect,
                        "text_central": text_central,
                        "region": region,
                        "speaker_id": speaker_id,
                        "utterance_id": utt_id,
                        "is_augmented": is_aug,
                    }
                )
    return samples


def split_by_augmentation(samples):
    """Split each speaker's clips into train/val/test independently.

    For each speaker (voice_X) we have 1 original + 100 augmented = 101 clips.
    We shuffle them with a fixed seed and take 80/10/10.
    The original (non-augmented) clip always goes to TEST so the final
    evaluation reflects performance on the cleanest version of the audio.
    """
    rng = random.Random(SEED)

    # Group samples by speaker
    by_speaker = {}
    for s in samples:
        by_speaker.setdefault((s["region"], s["speaker_id"]), []).append(s)

    splits = {"train": [], "val": [], "test": []}
    for key, items in by_speaker.items():
        # Separate originals and augmented
        originals = [s for s in items if not s["is_augmented"]]
        augmented = [s for s in items if s["is_augmented"]]

        # Shuffle augmented deterministically
        rng.shuffle(augmented)

        n = len(augmented)
        n_train = int(round(n * TRAIN_RATIO))
        n_val = int(round(n * VAL_RATIO))

        splits["train"].extend(augmented[:n_train])
        splits["val"].extend(augmented[n_train : n_train + n_val])
        splits["test"].extend(augmented[n_train + n_val :])

        # Originals → test (clean audio for fair eval)
        splits["test"].extend(originals)

    return splits


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_lookup(path, samples):
    lookup = {}
    for s in samples:
        if not s["is_augmented"]:
            lookup[s["text_dialect"]] = {
                "text_central": s["text_central"],
                "region": s["region"],
                "utterance_id": s["utterance_id"],
            }
    path.write_text(json.dumps(lookup, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(lookup)


def main():
    print(f"Scanning {DATA_ROOT}")
    samples = collect_samples()
    print(f"  -> {len(samples)} clips collected")

    if not samples:
        raise SystemExit("No samples found.")

    write_csv(OUT_ROOT / "metadata.csv", samples)

    splits = split_by_augmentation(samples)
    for name, rows in splits.items():
        write_jsonl(OUT_ROOT / f"{name}.jsonl", rows)
        utts = sorted({(r["region"], r["utterance_id"]) for r in rows})
        n_aug = sum(1 for r in rows if r["is_augmented"])
        n_orig = len(rows) - n_aug
        print(
            f"  {name:5s}: {len(rows):>5} clips | "
            f"{len(utts):>3} unique utterances | "
            f"orig={n_orig:>3}, aug={n_aug:>4}"
        )

    n_lookup = write_lookup(OUT_ROOT / "dialect_to_central.json", samples)
    print(f"  lookup entries: {n_lookup}")

    # Sanity check: ทุก split ควรมีทั้ง 50 utterances
    train_utts = {(r["region"], r["utterance_id"]) for r in splits["train"]}
    test_utts = {(r["region"], r["utterance_id"]) for r in splits["test"]}
    print(f"\nSplit overlap (utterances ที่อยู่ทั้ง train และ test):")
    print(f"  {len(train_utts & test_utts)} (ควรเท่ากับ 50 — โมเดลเห็นทุก utterance)")
    print(f"\nThis is intentional for closed-set evaluation (Path C strategy).")


if __name__ == "__main__":
    main()
