"""
Run a fine-tuned Wav2Vec2 model on one or more WAV files and (optionally)
translate the dialect transcript to Standard Thai using the lookup table.

Usage:
    # transcribe a single file
    python scripts/inference.py --model models/wav2vec2-south-th \\
        audio_data/south/voice_1/voice_1_1.wav

    # transcribe and translate to central Thai
    python scripts/inference.py --model models/wav2vec2-south-th --translate \\
        path/to/recording.wav

    # batch over a folder
    python scripts/inference.py --model models/wav2vec2-south-th --translate \\
        audio_data/south/voice_3/

Behaviour:
    - Loads model + processor from --model
    - Reads each .wav at 16 kHz mono (resamples if needed via librosa)
    - Outputs the raw CTC transcript
    - When --translate is set, looks the transcript up in
      manifests/dialect_to_central.json. If the exact match is not found,
      it falls back to a fuzzy lookup (closest entry by edit distance)
      and prints both the matched key and the translation.
"""

import argparse
import json
from difflib import get_close_matches
from pathlib import Path

import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

ROOT = Path(__file__).resolve().parent.parent
LOOKUP_PATH = ROOT / "manifests" / "dialect_to_central.json"


def load_lookup():
    if not LOOKUP_PATH.exists():
        return {}
    return json.loads(LOOKUP_PATH.read_text(encoding="utf-8"))


def translate(transcript: str, lookup: dict):
    """Return (matched_key, central_text) or (None, None)."""
    if transcript in lookup:
        return transcript, lookup[transcript]["text_central"]

    # Fuzzy fallback: find the closest known dialect phrase
    keys = list(lookup.keys())
    candidates = get_close_matches(transcript, keys, n=1, cutoff=0.6)
    if candidates:
        k = candidates[0]
        return k, lookup[k]["text_central"]
    return None, None


def transcribe(model, processor, wav_path: Path, device: str) -> str:
    audio, sr = librosa.load(str(wav_path), sr=16000, mono=True)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0]


def collect_files(target: Path):
    if target.is_file():
        return [target]
    return sorted(target.rglob("*.wav"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to fine-tuned model dir")
    ap.add_argument("--translate", action="store_true",
                    help="Also translate dialect → central Thai using lookup")
    ap.add_argument("input", help="A .wav file or a directory of .wav files")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}: {args.model}")
    processor = Wav2Vec2Processor.from_pretrained(args.model)
    model = Wav2Vec2ForCTC.from_pretrained(args.model).to(device)
    model.eval()

    lookup = load_lookup() if args.translate else {}
    files = collect_files(Path(args.input))
    if not files:
        raise SystemExit(f"No .wav files found at {args.input}")

    for wav_path in files:
        transcript = transcribe(model, processor, wav_path, device)
        print(f"\nFile: {wav_path}")
        print(f"  Dialect : {transcript}")
        if args.translate:
            key, central = translate(transcript, lookup)
            if central is None:
                print(f"  Central : <no match in lookup>")
            elif key == transcript:
                print(f"  Central : {central}")
            else:
                print(f"  Central : {central}   (fuzzy match: {key!r})")


if __name__ == "__main__":
    main()
