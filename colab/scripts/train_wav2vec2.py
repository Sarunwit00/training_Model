"""
Fine-tune Wav2Vec2 (XLS-R Thai) on the Southern Thai dialect dataset.

Pipeline:
  1. Load manifests (train/val) as a Hugging Face DatasetDict
  2. Cast audio_filepath -> 16 kHz Audio column
  3. Use a pretrained Thai Wav2Vec2 processor or build one from vocab.json
  4. Map waveform → input_values, transcript → labels (CTC ids)
  5. Train with HF Trainer, evaluating WER on the val split

Usage:
    # build vocab once
    python scripts/prepare_vocab.py --target text_dialect

    # then train
    python scripts/train_wav2vec2.py \\
        --base_model airesearch/wav2vec2-large-xlsr-53-th \\
        --target text_dialect \\
        --output_dir models/wav2vec2-south-th \\
        --epochs 30 \\
        --batch_size 8

Tip:
    Use --skip_augmented to train only on the 50 originals first
    (sanity check), then re-run with augmented data once the loss curves
    look healthy.
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
from datasets import Audio, load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

import evaluate

# Script lives at <project_root>/colab/scripts/train_wav2vec2.py
# so we need 3 ``.parent`` hops to reach the project root.
ROOT = Path(__file__).resolve().parent.parent.parent
MANIFESTS = ROOT / "manifests"
VOCAB_PATH = ROOT / "models" / "vocab" / "vocab.json"


# ---------------------------------------------------------------------------
# Data collator (dynamic padding for both audio and labels)
# ---------------------------------------------------------------------------
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(
            input_features, padding=self.padding, return_tensors="pt"
        )
        labels_batch = self.processor.pad(
            labels=label_features, padding=self.padding, return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


def build_processor(base_model: str, vocab_path: Path) -> Wav2Vec2Processor:
    """Use feature extractor from the base model + custom Thai vocab tokenizer."""
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(base_model)

    if vocab_path.exists():
        tokenizer = Wav2Vec2CTCTokenizer(
            str(vocab_path),
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token="|",
        )
    else:
        # fall back to the base model's tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model)

    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="airesearch/wav2vec2-large-xlsr-53-th")
    ap.add_argument(
        "--target",
        default="text_dialect",
        choices=["text_dialect", "text_central"],
        help="Which transcript field to train on",
    )
    ap.add_argument("--output_dir", default="models/wav2vec2-south-th")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--save_steps", type=int, default=400)
    ap.add_argument("--eval_steps", type=int, default=400)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=2)
    ap.add_argument(
        "--skip_augmented",
        action="store_true",
        help="Train only on original (non-augmented) clips",
    )
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Path to a checkpoint folder (e.g. models/.../checkpoint-1500) to resume from",
    )
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    data_files = {
        "train": str(MANIFESTS / "train.jsonl"),
        "val": str(MANIFESTS / "val.jsonl"),
    }
    ds = load_dataset("json", data_files=data_files)

    if args.skip_augmented:
        ds = ds.filter(lambda ex: not ex["is_augmented"])

    # Resolve audio paths to absolute and cast to Audio
    def _resolve(ex):
        ex["audio_filepath"] = str(ROOT / ex["audio_filepath"])
        return ex

    ds = ds.map(_resolve)
    ds = ds.cast_column("audio_filepath", Audio(sampling_rate=16000))

    print("Dataset:", ds)

    # ------------------------------------------------------------------
    # 2. Processor (feature extractor + tokenizer)
    # ------------------------------------------------------------------
    processor = build_processor(args.base_model, VOCAB_PATH)

    # Save processor up-front so checkpoints are self-contained
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(args.output_dir)

    # ------------------------------------------------------------------
    # 3. Prepare features
    # ------------------------------------------------------------------
    def prepare(example):
        audio = example["audio_filepath"]
        example["input_values"] = processor(
            audio["array"], sampling_rate=16000
        ).input_values[0]
        # input_length needed for group_by_length (safe speed optimization)
        example["input_length"] = len(example["input_values"])
        example["labels"] = processor(text=example[args.target]).input_ids
        return example

    keep_cols = ["input_values", "input_length", "labels"]
    ds = ds.map(
        prepare,
        remove_columns=[c for c in ds["train"].column_names if c not in keep_cols],
        num_proc=args.num_workers,
    )

    # ------------------------------------------------------------------
    # 4. Model
    # ------------------------------------------------------------------
    model = Wav2Vec2ForCTC.from_pretrained(
        args.base_model,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,
    )
    # Freeze the feature encoder for stability with small data
    model.freeze_feature_encoder()

    # ------------------------------------------------------------------
    # 5. Metrics
    # ------------------------------------------------------------------
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        labels = pred.label_ids
        labels[labels == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(labels, group_tokens=False)

        return {
            "wer": wer_metric.compute(predictions=pred_str, references=label_str),
            "cer": cer_metric.compute(predictions=pred_str, references=label_str),
        }

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="steps",
        num_train_epochs=args.epochs,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=50,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to="none",
        # ── Safe speed optimizations (no accuracy impact) ──────────────────
        group_by_length=True,                  # ลด padding waste ในแต่ละ batch
        length_column_name="input_length",     # ใช้คู่กับ group_by_length
        dataloader_pin_memory=True,            # data → GPU เร็วขึ้น
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor)

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        tokenizer=processor.feature_extractor,
    )

    if args.resume_from_checkpoint:
        print(f"▶ Resuming from checkpoint: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"\nDone. Best model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
