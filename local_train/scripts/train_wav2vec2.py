"""
Fine-tune Wav2Vec2 (XLS-R Thai) on the Southern Thai dialect dataset.

Usage:
    python scripts/prepare_vocab.py --target text_dialect
    python scripts/train_wav2vec2.py --epochs 30

    # resume after interrupt
    python scripts/train_wav2vec2.py \
        --output_dir models/wav2vec2-south-th \
        --resume_from_checkpoint models/wav2vec2-south-th/checkpoint-1000
"""

import argparse
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
import librosa
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

import evaluate

ROOT = Path(__file__).resolve().parent.parent
MANIFESTS = ROOT / "manifests"
VOCAB_PATH = ROOT / "models" / "vocab" / "vocab.json"


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features):
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def build_processor(base_model, vocab_path):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(base_model)
    if vocab_path.exists():
        tokenizer = Wav2Vec2CTCTokenizer(
            str(vocab_path),
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token="|",
        )
    else:
        print(f"WARN: vocab not found at {vocab_path}, falling back to base model tokenizer")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def safe_num_workers(requested):
    if requested == 0:
        return 0
    if platform.system() == "Windows" and requested > 2:
        print(f"WARN: Windows: reducing num_workers from {requested} to 2")
        return 2
    return requested


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="airesearch/wav2vec2-large-xlsr-53-th")
    ap.add_argument("--target", default="text_dialect", choices=["text_dialect", "text_central"])
    ap.add_argument("--output_dir", default="models/wav2vec2-south-th")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--save_steps", type=int, default=400)
    ap.add_argument("--eval_steps", type=int, default=400)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=2)
    ap.add_argument("--weight_decay", type=float, default=0.005)
    ap.add_argument("--skip_augmented", action="store_true",
                    help="Train only on originals (incompatible with build_manifest_v2.py)")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--resume_from_checkpoint", type=str, default=None)
    ap.add_argument("--bf16", action="store_true",
                    help="Use bfloat16 (faster on Ampere+ GPUs like A100/RTX30/40)")
    ap.add_argument("--no_freeze_encoder", action="store_true")
    args = ap.parse_args()

    args.num_workers = safe_num_workers(args.num_workers)

    # ------------------------------------------------------------------
    # 1. Load dataset (manifest only, audio loaded later via librosa)
    # ------------------------------------------------------------------
    train_path = MANIFESTS / "train.jsonl"
    val_path = MANIFESTS / "val.jsonl"
    if not train_path.exists() or not val_path.exists():
        sys.exit(f"ERROR: manifests not found in {MANIFESTS}\n"
                 f"  Run: python scripts/build_manifest_v2.py first")

    data_files = {"train": str(train_path), "val": str(val_path)}
    ds = load_dataset("json", data_files=data_files)

    if args.skip_augmented:
        print("Filtering augmented clips ...")
        ds = ds.filter(lambda ex: not ex["is_augmented"])
        if len(ds["train"]) == 0:
            sys.exit("ERROR: train set is empty after filtering!\n"
                     "  build_manifest_v2.py puts originals in test set only.\n"
                     "  Either run without --skip_augmented OR use build_manifest.py instead.\n")
        print(f"  Train: {len(ds['train'])} | Val: {len(ds['val'])}")

    print("Dataset:", ds)

    # ------------------------------------------------------------------
    # 2. Processor
    # ------------------------------------------------------------------
    processor = build_processor(args.base_model, VOCAB_PATH)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(args.output_dir)

    # ------------------------------------------------------------------
    # 3. Prepare features (โหลด audio ผ่าน librosa เอง — ไม่พึ่ง torchcodec)
    # ------------------------------------------------------------------
    def prepare(example):
        # Resolve relative path -> absolute
        wav_path = example["audio_filepath"]
        if not os.path.isabs(wav_path):
            wav_path = str(ROOT / wav_path)

        # Load audio via librosa (no torchcodec dependency)
        audio_array, _ = librosa.load(wav_path, sr=16000, mono=True)

        # Feature extract
        example["input_values"] = processor(audio_array, sampling_rate=16000).input_values[0]
        # Tokenize transcript
        example["labels"] = processor(text=example[args.target]).input_ids
        return example

    keep_cols = ["input_values", "labels"]
    ds = ds.map(
        prepare,
        remove_columns=[c for c in ds["train"].column_names if c not in keep_cols],
        num_proc=args.num_workers if args.num_workers > 0 else None,
    )

    # ------------------------------------------------------------------
    # 4. Model
    # ------------------------------------------------------------------
    print(f"Loading {args.base_model} ...")
    model = Wav2Vec2ForCTC.from_pretrained(
        args.base_model,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,
    )

    # CRITICAL FIX: save model config early so output_dir always has config.json
    model.config.save_pretrained(args.output_dir)
    print(f"Saved model config.json at {args.output_dir}")

    if args.no_freeze_encoder:
        print("Not freezing feature encoder")
    else:
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
    use_cuda = torch.cuda.is_available()
    use_bf16 = args.bf16 and use_cuda
    use_fp16 = use_cuda and not use_bf16

    if use_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
        print(f"Precision: {'bfloat16' if use_bf16 else 'float16'}")
    else:
        print("WARN: No GPU detected — training on CPU will be very slow")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        num_train_epochs=args.epochs,
        gradient_checkpointing=True,
        fp16=use_fp16,
        bf16=use_bf16,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=50,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=args.num_workers,
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
        ckpt = args.resume_from_checkpoint
        if not Path(ckpt).exists():
            sys.exit(f"ERROR: checkpoint not found: {ckpt}")
        print(f"Resuming training from {ckpt}")
        trainer.train(resume_from_checkpoint=ckpt)
    else:
        trainer.train()

    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"\nDone. Best model saved to: {args.output_dir}")
    print(f"Next: python scripts/export_model.py --model_dir {args.output_dir}")


if __name__ == "__main__":
    main()
