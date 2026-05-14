"""
Unit tests สำหรับโมเดล Wav2Vec2 ที่เทรนมา (ภาษาใต้)

วิธีรัน:
    cd local_train
    pip install pytest jiwer
    pytest tests/test_model.py -v

    # ทดสอบเฉพาะกลุ่ม
    pytest tests/test_model.py -v -m "not slow"      # ข้าม inference จริง (เร็ว)
    pytest tests/test_model.py -v -m slow            # เฉพาะ inference จริง
    pytest tests/test_model.py::test_inference_returns_thai_text -v

    # เลือกโมเดลอื่น
    MODEL_DIR=models/wav2vec2-south-th-v2 pytest tests/test_model.py -v
"""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

import pytest

# -----------------------------------------------------------------------------
# Paths / config
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_REL = "models/wav2vec2-south-th-fast"
MODEL_DIR = Path(os.getenv("MODEL_DIR", DEFAULT_MODEL_REL))
if not MODEL_DIR.is_absolute():
    MODEL_DIR = ROOT / MODEL_DIR

TEST_MANIFEST = ROOT / "manifests" / "test.jsonl"
LOOKUP_PATH = ROOT / "manifests" / "dialect_to_central.json"

# กี่ตัวอย่างที่จะใช้ประเมิน CER (อย่ามากเกินไปเพราะรัน CPU จะช้า)
N_EVAL_SAMPLES = int(os.getenv("N_EVAL_SAMPLES", "5"))
# CER threshold คร่าว ๆ — ปรับให้สอดคล้องกับคุณภาพโมเดล
CER_THRESHOLD = float(os.getenv("CER_THRESHOLD", "0.6"))


# -----------------------------------------------------------------------------
# Helpers / fixtures
# -----------------------------------------------------------------------------
def _model_exists() -> bool:
    return MODEL_DIR.exists() and (MODEL_DIR / "config.json").exists()


# Skip ทุก test ในไฟล์นี้ถ้ายังไม่มีโมเดล (ป้องกัน CI พัง)
pytestmark = pytest.mark.skipif(
    not _model_exists(),
    reason=f"Model dir not found at {MODEL_DIR}. ลองเทรนก่อน หรือเซ็ต MODEL_DIR",
)


@pytest.fixture(scope="session")
def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def processor():
    from transformers import Wav2Vec2Processor
    return Wav2Vec2Processor.from_pretrained(str(MODEL_DIR))


@pytest.fixture(scope="session")
def model(device):
    from transformers import Wav2Vec2ForCTC
    m = Wav2Vec2ForCTC.from_pretrained(str(MODEL_DIR)).to(device)
    m.eval()
    return m


@pytest.fixture(scope="session")
def test_samples():
    """อ่าน sample จาก test.jsonl แบบสุ่ม (มี seed ให้ reproducible)"""
    if not TEST_MANIFEST.exists():
        pytest.skip(f"test manifest not found: {TEST_MANIFEST}")
    with TEST_MANIFEST.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    # กรองเฉพาะที่ไฟล์ wav ยังอยู่จริง
    rows = [r for r in rows if (ROOT / r["audio_filepath"]).exists()]
    if not rows:
        pytest.skip("ไม่มีไฟล์ wav ของ test set อยู่บนดิสก์")
    rng = random.Random(42)
    rng.shuffle(rows)
    return rows[:N_EVAL_SAMPLES]


def _transcribe(model, processor, wav_path: Path, device: str) -> str:
    import librosa
    import torch
    audio, _ = librosa.load(str(wav_path), sr=16000, mono=True)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0]


# -----------------------------------------------------------------------------
# 1) Sanity: ไฟล์ artifact ครบ
# -----------------------------------------------------------------------------
class TestModelArtifacts:
    """โมเดลที่ save มาควรมีไฟล์เหล่านี้ ครบ"""

    REQUIRED_FILES = [
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "vocab.json",
    ]

    @pytest.mark.parametrize("fname", REQUIRED_FILES)
    def test_required_file_exists(self, fname):
        assert (MODEL_DIR / fname).exists(), f"missing {fname} in {MODEL_DIR}"

    def test_weights_file_exists(self):
        candidates = ["model.safetensors", "pytorch_model.bin"]
        assert any((MODEL_DIR / c).exists() for c in candidates), (
            f"ไม่พบ weights file ({candidates}) ใน {MODEL_DIR}"
        )

    def test_config_is_wav2vec2(self):
        config = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
        assert config.get("model_type") == "wav2vec2", config.get("model_type")
        # vocab size ใน config ต้องตรงกับ vocab.json
        vocab = json.loads((MODEL_DIR / "vocab.json").read_text(encoding="utf-8"))
        assert config["vocab_size"] == len(vocab), (
            f"config.vocab_size={config['vocab_size']} แต่ vocab.json มี {len(vocab)} tokens"
        )

    def test_vocab_has_thai_characters(self):
        vocab = json.loads((MODEL_DIR / "vocab.json").read_text(encoding="utf-8"))
        thai_chars = [k for k in vocab.keys() if re.match(r"[฀-๿]", k)]
        assert len(thai_chars) >= 30, f"vocab มีอักขระไทยแค่ {len(thai_chars)} ตัว"
        # อักขระที่ใช้บ่อย ๆ ต้องมี
        for ch in ["ก", "า", "น", "ร"]:
            assert ch in vocab, f"vocab ขาด '{ch}'"

    def test_vocab_has_special_tokens(self):
        vocab = json.loads((MODEL_DIR / "vocab.json").read_text(encoding="utf-8"))
        for token in ["[PAD]", "[UNK]", "|"]:
            assert token in vocab, f"vocab ขาด special token '{token}'"


# -----------------------------------------------------------------------------
# 2) โหลด processor / model สำเร็จ
# -----------------------------------------------------------------------------
class TestModelLoading:
    def test_processor_loads(self, processor):
        # processor ต้องมีทั้ง feature extractor และ tokenizer
        assert processor.feature_extractor is not None
        assert processor.tokenizer is not None
        assert processor.feature_extractor.sampling_rate == 16000

    def test_model_loads_and_in_eval_mode(self, model):
        import torch
        assert isinstance(model, torch.nn.Module)
        assert not model.training, "model ควรอยู่ใน eval mode"

    def test_model_vocab_size_matches_processor(self, model, processor):
        # final layer ต้องคืนค่าเท่ากับ vocab
        assert model.config.vocab_size == processor.tokenizer.vocab_size


# -----------------------------------------------------------------------------
# 3) Inference smoke test (ใช้เวลา — ตีว่า slow)
# -----------------------------------------------------------------------------
@pytest.mark.slow
class TestInference:
    def test_inference_on_dummy_audio_runs(self, model, processor, device):
        """ป้อน silence 1 วินาที — ต้องไม่ crash และคืน string"""
        import numpy as np
        import torch

        audio = np.zeros(16000, dtype=np.float32)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inputs.input_values.to(device)).logits

        assert logits.dim() == 3, f"logits ต้องเป็น 3D (B, T, V) แต่ได้ {logits.shape}"
        assert logits.shape[0] == 1
        assert logits.shape[2] == model.config.vocab_size

        pred_ids = torch.argmax(logits, dim=-1)
        out = processor.batch_decode(pred_ids)[0]
        assert isinstance(out, str)

    def test_inference_returns_thai_text(self, model, processor, device, test_samples):
        """ป้อนไฟล์จริง — output ควรเป็น string ที่ไม่ว่าง และมีอักขระไทย"""
        sample = test_samples[0]
        wav = ROOT / sample["audio_filepath"]
        out = _transcribe(model, processor, wav, device)

        assert isinstance(out, str)
        assert len(out.strip()) > 0, f"ได้ string ว่างจากไฟล์ {wav}"
        thai_chars = [c for c in out if re.match(r"[฀-๿]", c)]
        assert len(thai_chars) > 0, f"output ไม่มีอักขระไทยเลย: {out!r}"

    def test_inference_is_deterministic(self, model, processor, device, test_samples):
        """รันสองครั้งบนไฟล์เดียวกัน ต้องได้ผลเหมือนเดิม (eval mode + argmax)"""
        wav = ROOT / test_samples[0]["audio_filepath"]
        out1 = _transcribe(model, processor, wav, device)
        out2 = _transcribe(model, processor, wav, device)
        assert out1 == out2, f"ผลไม่ตรงกัน:\n  {out1!r}\n  {out2!r}"


# -----------------------------------------------------------------------------
# 4) คุณภาพคร่าว ๆ — CER บน sample เล็ก ๆ
# -----------------------------------------------------------------------------
@pytest.mark.slow
class TestQuality:
    def test_cer_below_threshold(self, model, processor, device, test_samples):
        """ประเมิน CER บน sample เล็ก ๆ จาก test set"""
        try:
            from jiwer import cer
        except ImportError:
            pytest.skip("ติดตั้ง jiwer ก่อน: pip install jiwer")

        preds, refs = [], []
        for s in test_samples:
            wav = ROOT / s["audio_filepath"]
            preds.append(_transcribe(model, processor, wav, device))
            refs.append(s["text_dialect"])

        score = cer(refs, preds)
        # log ให้เห็นค่า CER จริงในรายงาน pytest -v
        print(f"\n[CER on {len(refs)} samples] = {score:.4f}")
        for r, p in zip(refs, preds):
            print(f"  REF: {r}\n  HYP: {p}\n")

        assert score <= CER_THRESHOLD, (
            f"CER={score:.3f} เกิน threshold {CER_THRESHOLD}. "
            f"โมเดลอาจยังเทรนไม่พอ หรือ data mismatch"
        )


# -----------------------------------------------------------------------------
# 5) Translation lookup (dialect → central Thai)
# -----------------------------------------------------------------------------
class TestTranslationLookup:
    @pytest.fixture(scope="class")
    def lookup(self):
        if not LOOKUP_PATH.exists():
            pytest.skip(f"lookup file ไม่พบ: {LOOKUP_PATH}")
        return json.loads(LOOKUP_PATH.read_text(encoding="utf-8"))

    def test_lookup_is_non_empty(self, lookup):
        assert len(lookup) > 0

    def test_lookup_entries_have_central(self, lookup):
        for k, v in list(lookup.items())[:50]:
            assert isinstance(v, dict), f"value ของ {k!r} ไม่ใช่ dict"
            assert "text_central" in v, f"key {k!r} ขาด text_central"
            assert isinstance(v["text_central"], str) and v["text_central"].strip()

    def test_exact_match_translation(self, lookup):
        from difflib import get_close_matches
        # หยิบ key แรก แล้วลอง lookup ตรง ๆ
        key = next(iter(lookup))
        assert lookup[key]["text_central"]
        # fuzzy ก็ยังเจอ
        matches = get_close_matches(key, list(lookup.keys()), n=1, cutoff=0.6)
        assert matches and matches[0] == key
