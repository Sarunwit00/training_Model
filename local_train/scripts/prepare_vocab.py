"""
Build a character-level vocabulary for Wav2Vec2 CTC training.

Reads `manifests/train.jsonl`, collects every unique character that appears
in the chosen text column, then writes `models/vocab/vocab.json` in the
format Wav2Vec2CTCTokenizer expects.

Usage:
    python scripts/prepare_vocab.py --target text_dialect
    python scripts/prepare_vocab.py --target text_central
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFESTS = ROOT / "manifests"
OUT_DIR = ROOT / "models" / "vocab"


def collect_chars(jsonl_path: Path, text_key: str) -> set[str]:
    chars: set[str] = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            chars.update(row[text_key])
    return chars


def build_vocab(chars: set[str]) -> dict:
    """
    Wav2Vec2CTCTokenizer expects a JSON dict mapping char → integer id.
    By convention:
      - "|" replaces space (word delimiter)
      - "[UNK]" for unknown char
      - "[PAD]" for CTC blank
    """
    # Drop literal space, we replace with "|"
    chars = sorted(c for c in chars if c != " ")
    vocab = {c: i for i, c in enumerate(chars)}
    vocab["|"] = len(vocab)
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)
    return vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--target",
        default="text_dialect",
        choices=["text_dialect", "text_central"],
        help="Which transcript column to build vocab from",
    )
    ap.add_argument(
        "--manifest",
        default=str(MANIFESTS / "train.jsonl"),
        help="Path to train manifest (jsonl)",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    chars = collect_chars(Path(args.manifest), args.target)
    print(f"Unique characters from '{args.target}': {len(chars)}")
    print(f"  {''.join(sorted(c for c in chars if c != ' '))}")

    vocab = build_vocab(chars)
    out_path = OUT_DIR / "vocab.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out_path} with {len(vocab)} tokens")


if __name__ == "__main__":
    main()
