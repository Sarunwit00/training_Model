"""
Build a character-level vocabulary for Wav2Vec2 CTC training.

Reads manifests/train.jsonl, collects every unique character that appears
in the chosen text column, then writes models/vocab/vocab.json.

Usage:
    python scripts/prepare_vocab.py --target text_dialect
    python scripts/prepare_vocab.py --target text_central
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFESTS = ROOT / "manifests"
OUT_DIR = ROOT / "models" / "vocab"


def collect_chars(jsonl_path, text_key):
    chars = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if text_key not in row:
                sys.exit(f"ERROR: column '{text_key}' not found in manifest.\n"
                         f"  Available columns: {list(row.keys())}")
            chars.update(row[text_key])
    return chars


def build_vocab(chars):
    """Wav2Vec2CTCTokenizer expects char -> id mapping with special tokens."""
    # Drop literal space, replace with "|"
    chars = sorted(c for c in chars if c != " ")
    vocab = {c: i for i, c in enumerate(chars)}
    vocab["|"] = len(vocab)
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)
    return vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="text_dialect",
                    choices=["text_dialect", "text_central"])
    ap.add_argument("--manifest", default=str(MANIFESTS / "train.jsonl"))
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        sys.exit(f"ERROR: manifest not found: {manifest_path}\n"
                 f"  Run: python scripts/build_manifest_v2.py first")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    chars = collect_chars(manifest_path, args.target)
    if not chars:
        sys.exit(f"ERROR: no characters collected from '{args.target}' column.\n"
                 f"  Manifest may be empty or column may be wrong.")

    print(f"Unique characters from '{args.target}': {len(chars)}")
    print(f"  {''.join(sorted(c for c in chars if c != ' '))}")

    vocab = build_vocab(chars)
    out_path = OUT_DIR / "vocab.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out_path} with {len(vocab)} tokens")


if __name__ == "__main__":
    main()
