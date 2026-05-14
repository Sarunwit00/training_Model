"""
Build dataset manifests for multi-dialect Thai ASR fine-tuning
(supports south + north + isan).

Strategy (Path B, uniform across all regions):
  - For each speaker:
      ・ files ending in "_original.wav" (and the canonical "voice_X_X.wav"
        in south) are reserved for the TEST split — so evaluation is always
        performed on the cleanest version of the audio.
      ・ all remaining files (variations: child/female/male/elderly_*,
        plus south's _aug_*) are shuffled deterministically and split
        80 % train / 10 % val / 10 % test.

Why uniform?
  Previously v2 split only "_aug_" files and dumped every variation into
  test. That worked for south (which has 100 aug files) but breaks for
  north/isan (which have NO _aug_ files — everything would land in test).
  v3 treats child/female/male/elderly_* as natural augmentation, which is
  what they effectively are.

Output (overwrites manifests/*.jsonl):
  - manifests/train.jsonl
  - manifests/val.jsonl
  - manifests/test.jsonl
  - manifests/metadata.csv
  - manifests/dialect_to_central.json

Usage:
    python colab/scripts/build_manifest_v3.py
"""

import csv
import json
import random
import re
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = ROOT / "audio_data"
OUT_ROOT = ROOT / "manifests"
OUT_ROOT.mkdir(exist_ok=True)

SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
# test = remainder (+ all originals)

# Categories detected from filename suffix.
CATEGORY_PATTERNS = [
    ("elderly_female", re.compile(r"_elderly_female\.wav$")),
    ("elderly_male",   re.compile(r"_elderly_male\.wav$")),
    ("child",          re.compile(r"_child\.wav$")),
    ("female",         re.compile(r"_female\.wav$")),
    ("male",           re.compile(r"_male\.wav$")),
    ("original",       re.compile(r"_original\.wav$")),
    ("aug",            re.compile(r"_aug_\d+\.wav$")),
]


def categorize(filename: str) -> str:
    """Return the variation category of an audio file.

    "canonical" is the south-only base file like voice_1_1.wav that has no
    suffix — we treat it as another flavour of original for test reservation.
    """
    for name, pat in CATEGORY_PATTERNS:
        if pat.search(filename):
            return name
    # south canonical: voice_<spk>_<utt>.wav (no descriptive suffix)
    return "canonical"


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
                print(f"[skip] no ReadMe in {voice_dir}")
                continue
            text_dialect, text_central = parse_readme(readme)
            if not text_dialect:
                print(f"[skip] could not parse {readme}")
                continue

            for wav in sorted(voice_dir.glob("*.wav")):
                try:
                    meta = wav_meta(wav)
                except wave.Error as e:
                    print(f"[skip] bad wav {wav}: {e}")
                    continue
                category = categorize(wav.name)
                # is_augmented retained for backward compat with train script
                # (--skip_augmented filter). True for anything that is not
                # the clean original recording.
                is_aug = category not in ("original", "canonical")
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
                        "utterance_id": speaker_id,  # 1 sentence per speaker
                        "category": category,
                        "is_augmented": is_aug,
                    }
                )
    return samples


def split_per_speaker(samples):
    """Plan B: originals → test, shuffle rest 80/10/10 per speaker."""
    rng = random.Random(SEED)

    by_speaker = {}
    for s in samples:
        by_speaker.setdefault((s["region"], s["speaker_id"]), []).append(s)

    splits = {"train": [], "val": [], "test": []}
    for key, items in by_speaker.items():
        originals = [s for s in items if s["category"] in ("original", "canonical")]
        rest = [s for s in items if s["category"] not in ("original", "canonical")]

        # Deterministic shuffle (per-speaker seed via key hash for stability)
        local_rng = random.Random((SEED, key))
        local_rng.shuffle(rest)

        n = len(rest)
        n_train = int(round(n * TRAIN_RATIO))
        n_val = int(round(n * VAL_RATIO))

        splits["train"].extend(rest[:n_train])
        splits["val"].extend(rest[n_train : n_train + n_val])
        splits["test"].extend(rest[n_train + n_val :])

        # Originals always go to test (clean audio for fair evaluation)
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
    """Build dialect→central lookup from clean original samples only."""
    lookup = {}
    for s in samples:
        if s["category"] in ("original", "canonical"):
            # keep first occurrence per dialect string
            key = s["text_dialect"]
            if key not in lookup:
                lookup[key] = {
                    "text_central": s["text_central"],
                    "region": s["region"],
                    "speaker_id": s["speaker_id"],
                }
    path.write_text(json.dumps(lookup, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(lookup)


def main():
    print(f"Scanning {DATA_ROOT}")
    samples = collect_samples()
    print(f"  -> {len(samples)} clips collected")
    if not samples:
        raise SystemExit("No samples found.")

    # Region & category breakdown
    from collections import Counter
    by_region = Counter(s["region"] for s in samples)
    by_cat = Counter(s["category"] for s in samples)
    print(f"  Per-region clip counts: {dict(by_region)}")
    print(f"  Per-category counts:    {dict(by_cat)}")

    write_csv(OUT_ROOT / "metadata.csv", samples)

    splits = split_per_speaker(samples)
    for name, rows in splits.items():
        write_jsonl(OUT_ROOT / f"{name}.jsonl", rows)
        regions = Counter(r["region"] for r in rows)
        cats = Counter(r["category"] for r in rows)
        speakers = len({(r["region"], r["speaker_id"]) for r in rows})
        print(
            f"  {name:5s}: {len(rows):>6} clips | "
            f"speakers={speakers:>3} | "
            f"regions={dict(regions)} | "
            f"categories={dict(cats)}"
        )

    n_lookup = write_lookup(OUT_ROOT / "dialect_to_central.json", samples)
    print(f"  lookup entries: {n_lookup}")

    # Sanity: each speaker should appear in all 3 splits
    train_spk = {(r["region"], r["speaker_id"]) for r in splits["train"]}
    val_spk = {(r["region"], r["speaker_id"]) for r in splits["val"]}
    test_spk = {(r["region"], r["speaker_id"]) for r in splits["test"]}
    print(
        f"\nSpeaker coverage check (closed-set, ควรเท่ากันทั้ง 3):"
        f"\n  train={len(train_spk)}  val={len(val_spk)}  test={len(test_spk)}"
    )

    # Test-set composition
    test_cats = Counter(r["category"] for r in splits["test"])
    print(f"\nTest set category mix: {dict(test_cats)}")
    print("  (originals are reserved → test gets a clean baseline for eval)")


if __name__ == "__main__":
    main()
