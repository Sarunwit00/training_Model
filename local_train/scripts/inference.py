"""
Run a fine-tuned Wav2Vec2 model on WAV files and translate dialect → central Thai.

Usage:
    # transcribe single file
    python scripts/inference.py audio.wav

    # batch over folder + translate
    python scripts/inference.py path/to/folder/ --translate

    # specify model dir
    python scripts/inference.py audio.wav --model models/sanity-check
"""

import argparse
import json
import sys
from difflib import get_close_matches
from pathlib import Path

import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

ROOT = Path(__file__).resolve().parent.parent
LOOKUP_PATH = ROOT / "manifests" / "dialect_to_central.json"
DEFAULT_MODEL = "models/wav2vec2-south-th-v2"


def load_lookup():
    if not LOOKUP_PATH.exists():
        print(f"WARN: lookup not found at {LOOKUP_PATH}")
        return {}
    return json.loads(LOOKUP_PATH.read_text(encoding="utf-8"))


def translate(transcript, lookup):
    if transcript in lookup:
        return transcript, lookup[transcript]["text_central"]
    keys = list(lookup.keys())
    candidates = get_close_matches(transcript, keys, n=1, cutoff=0.6)
    if candidates:
        return candidates[0], lookup[candidates[0]]["text_central"]
    return None, None


def transcribe(model, processor, wav_path, device):
    audio, _ = librosa.load(str(wav_path), sr=16000, mono=True)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0]


def collect_files(target):
    if target.is_file():
        return [target]
    if target.is_dir():
        return sorted(target.rglob("*.wav"))
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="A .wav file or directory of .wav files")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Path to fine-tuned model dir")
    ap.add_argument("--translate", action="store_true", help="Translate dialect to central Thai")
    args = ap.parse_args()

    # Validate model path
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / args.model
    if not model_path.exists():
        sys.exit(f"ERROR: model directory not found: {model_path}\n"
                 f"  Train first: python scripts/train_wav2vec2.py --epochs 30")
    if not (model_path / "config.json").exists():
        sys.exit(f"ERROR: config.json missing in {model_path}\n"
                 f"  Training may have been interrupted before saving the model.\n"
                 f"  Look for checkpoint: ls {model_path}/checkpoint-*")

    # Validate input path
    target = Path(args.input)
    if not target.is_absolute() and not target.exists():
        # try relative to project root
        alt = ROOT / args.input
        if alt.exists():
            target = alt
    if not target.exists():
        sys.exit(f"ERROR: input not found: {args.input}")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}: {model_path}")
    processor = Wav2Vec2Processor.from_pretrained(str(model_path))
    model = Wav2Vec2ForCTC.from_pretrained(str(model_path)).to(device)
    model.eval()

    lookup = load_lookup() if args.translate else {}
    files = collect_files(target)
    if not files:
        sys.exit(f"ERROR: no .wav files found at {target}")

    print(f"Found {len(files)} file(s)\n")
    for wav_path in files:
        transcript = transcribe(model, processor, wav_path, device)
        print(f"File: {wav_path}")
        print(f"  Dialect : {transcript}")
        if args.translate:
            key, central = translate(transcript, lookup)
            if central is None:
                print(f"  Central : <no match>")
            elif key == transcript:
                print(f"  Central : {central}  [exact]")
            else:
                print(f"  Central : {central}  [fuzzy: {key}]")
        print()


if __name__ == "__main__":
    main()
