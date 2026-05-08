"""
Build dataset manifests for Thai dialect ASR fine-tuning.

Scans audio_data/<region>/voice_*/, reads ReadMe.txt for transcript pairs,
and outputs:
  - manifests/metadata.csv          (master list of all clips)
  - manifests/train.jsonl           (80% of speakers/utterances)
  - manifests/val.jsonl             (10%)
  - manifests/test.jsonl            (10%)
  - manifests/dialect_to_central.json (lookup table for stage-2 translation)

Split is done at the speaker_id (= utterance_id) level so augmented
versions of the same source never leak across splits.
"""

import csv
import json
import random
import wave
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "audio_data"
OUT_ROOT = ROOT / "manifests"
OUT_ROOT.mkdir(exist_ok=True)

SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
# test = remainder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_readme(path: Path):
    """Return (dialect_text, central_text) from a ReadMe.txt file.

    Some files in the dataset may be truncated mid-UTF-8 character, so we
    decode with errors='replace'. Lines that contain the U+FFFD replacement
    character are flagged for the user to inspect.
    """
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    if "�" in text:
        print(f"[warn] {path} contains undecodable bytes (truncated file?)")
    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        left, right = [p.strip() for p in line.split("=", 1)]
        # skip the header line "transcript = translation"
        if left.lower() == "transcript" and right.lower() == "translation":
            continue
        return left, right
    return None, None


def wav_meta(path: Path):
    with wave.open(str(path), "rb") as w:
        return {
            "duration": round(w.getnframes() / w.getframerate(), 3),
            "sample_rate": w.getframerate(),
            "channels": w.getnchannels(),
            "sample_width_bits": w.getsampwidth() * 8,
        }


# ---------------------------------------------------------------------------
# Scan dataset
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def split_by_speaker(samples):
    """Split at speaker_id level so augments don't leak across splits."""
    speakers_by_region = {}
    for s in samples:
        speakers_by_region.setdefault(s["region"], set()).add(s["speaker_id"])

    rng = random.Random(SEED)
    train_keys, val_keys, test_keys = set(), set(), set()
    for region, speakers in speakers_by_region.items():
        speakers = sorted(speakers)
        rng.shuffle(speakers)
        n = len(speakers)
        n_train = int(round(n * TRAIN_RATIO))
        n_val = int(round(n * VAL_RATIO))
        train_keys.update((region, s) for s in speakers[:n_train])
        val_keys.update((region, s) for s in speakers[n_train : n_train + n_val])
        test_keys.update((region, s) for s in speakers[n_train + n_val :])

    splits = {"train": [], "val": [], "test": []}
    for s in samples:
        key = (s["region"], s["speaker_id"])
        if key in train_keys:
            splits["train"].append(s)
        elif key in val_keys:
            splits["val"].append(s)
        else:
            splits["test"].append(s)
    return splits


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows):
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_lookup(path: Path, samples):
    """Build dialect -> central lookup from non-augmented (canonical) samples."""
    lookup = {}
    for s in samples:
        if not s["is_augmented"]:
            lookup[s["text_dialect"]] = {
                "text_central": s["text_central"],
                "region": s["region"],
                "utterance_id": s["utterance_id"],
            }
    path.write_text(
        json.dumps(lookup, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return len(lookup)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Scanning {DATA_ROOT}")
    samples = collect_samples()
    print(f"  -> {len(samples)} clips collected")

    if not samples:
        raise SystemExit("No samples found. Check audio_data path.")

    write_csv(OUT_ROOT / "metadata.csv", samples)

    splits = split_by_speaker(samples)
    for name, rows in splits.items():
        write_jsonl(OUT_ROOT / f"{name}.jsonl", rows)
        speakers = sorted({(r["region"], r["speaker_id"]) for r in rows})
        print(f"  {name}: {len(rows):>5} clips | {len(speakers):>3} speakers")

    n_lookup = write_lookup(OUT_ROOT / "dialect_to_central.json", samples)
    print(f"  lookup entries: {n_lookup}")

    # Sanity check: no speaker overlap across splits
    train_keys = {(r["region"], r["speaker_id"]) for r in splits["train"]}
    val_keys = {(r["region"], r["speaker_id"]) for r in splits["val"]}
    test_keys = {(r["region"], r["speaker_id"]) for r in splits["test"]}
    assert not (train_keys & val_keys), "leakage train<->val"
    assert not (train_keys & test_keys), "leakage train<->test"
    assert not (val_keys & test_keys), "leakage val<->test"
    print("Leakage check: OK (no shared speakers across splits)")

    # Region summary
    by_region = {}
    for s in samples:
        by_region.setdefault(s["region"], 0)
        by_region[s["region"]] += 1
    print("Per-region totals:", by_region)


if __name__ == "__main__":
    main()
