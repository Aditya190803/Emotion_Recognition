#!/usr/bin/env python3
"""
Best readily-available facial emotion model wrapper.

Model: mo-thecreator/vit-Facial-Expression-Recognition
  - Architecture: ViT-Base (google/vit-base-patch16-224-in21k fine-tune)
  - Trained on: FER2013 + AffectNet + MMI (multi-dataset, more robust)
  - Eval accuracy: ~84.3-84.9% (vs ~71% for trpakov ViT, ~72-77% for
    MobileNetV2/EfficientNetB0 transfer-learning baselines)
  - HF config id2label: 0=anger,1=disgust,2=fear,3=happy,4=neutral,5=sad,6=surprise

Why this one (researched Sept 2026):
  - Paper SOTA (POSTER-Var 92.7% RAF-DB, MSAG 90.4% FER2013, FERMam, IAE-Net)
    has no pip-installable weights — custom research code only.
  - Among plug-and-play weights, this HF ViT is the best: multi-dataset,
    2k+ pulls/month, 11 Spaces, openly loadable via transformers.
  - Runner-ups (trpakov/vit-face-expression, abhilash88, clip-face-expression)
    all sit at ~71-72% on FER2013-only.

Usage:
    from emotion_vit import predict_emotion_vit, get_vit_labels
    result = predict_emotion_vit(face_bgr)  # BGR numpy array (face ROI or full frame)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

HF_MODEL_ID = os.getenv(
    "HF_MODEL_ID", "mo-thecreator/vit-Facial-Expression-Recognition"
)

# On-disk weight cache: snapshot the HF repo once into ./models so Streamlit
# (and offline runs) load from local disk instead of re-downloading.
# Overridable via VIT_CACHE_DIR. This dir is git-ignored (350MB+ of weights).
VIT_CACHE_BASE = Path(os.getenv("VIT_CACHE_DIR", "models"))


def _local_snapshot_dir(model_id: str = HF_MODEL_ID) -> Path:
    return VIT_CACHE_BASE / model_id.replace("/", "--")


def is_vit_cached_locally(model_id: str = HF_MODEL_ID) -> bool:
    return (_local_snapshot_dir(model_id) / "config.json").exists()


def resolve_model_source(model_id: str = HF_MODEL_ID) -> str:
    """Local snapshot path if cached, else the HF repo id (downloads on load)."""
    d = _local_snapshot_dir(model_id)
    return str(d) if (d / "config.json").exists() else model_id


def ensure_vit_snapshot(model_id: str = HF_MODEL_ID) -> Path:
    """Download HF weights once into ./models. Returns the snapshot dir."""
    from huggingface_hub import snapshot_download
    d = _local_snapshot_dir(model_id)
    snapshot_download(repo_id=model_id, local_dir=str(d))
    _load_vit.cache_clear()
    return d

# Canonical labels used across app.py / realtime_prediction.py
CANONICAL_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# HF -> canonical (HF uses "anger", we use "angry")
_HF_TO_CANONICAL = {"anger": "angry"}

EMOTION_EMOJIS = {
    "angry": "😠", "disgust": "🤢", "fear": "😨",
    "happy": "😊", "neutral": "😐", "sad": "😢", "surprise": "😲",
}


@lru_cache(maxsize=2)
def _load_vit(model_id: str = HF_MODEL_ID):
    """Lazy-load processor + model (cached). Import transformers/torch only here."""
    from transformers import AutoImageProcessor, ViTForImageClassification
    import torch

    src = resolve_model_source(model_id)
    try:
        processor = AutoImageProcessor.from_pretrained(src, backend="pil")
    except TypeError:  # transformers < 4.48 has no `backend` arg
        processor = AutoImageProcessor.from_pretrained(src, use_fast=False)
    model = ViTForImageClassification.from_pretrained(src)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # Resolve label order from config (handles anger vs angry)
    raw_labels: List[str] = [
        model.config.id2label[i] for i in range(len(model.config.id2label))
    ]
    canonical = [_HF_TO_CANONICAL.get(l.lower(), l.lower()) for l in raw_labels]
    return processor, model, device, canonical


def get_vit_labels(model_id: str = HF_MODEL_ID) -> List[str]:
    try:
        _, _, _, canonical = _load_vit(model_id)
        return canonical
    except Exception:
        return CANONICAL_LABELS


def predict_emotion_vit(face_bgr: np.ndarray, model_id: str = HF_MODEL_ID) -> Dict:
    """Predict emotion from a BGR face ROI. Returns same dict shape as app.py."""
    from transformers import logging as hf_logging
    import torch

    hf_logging.set_verbosity_error()
    processor, model, device, canonical = _load_vit(model_id)

    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

    order = np.argsort(-probs)
    predictions = [
        {
            "label": canonical[i],
            "score": float(probs[i]),
            "emoji": EMOTION_EMOJIS.get(canonical[i], ""),
        }
        for i in order
    ]
    return {
        "predictions": predictions,
        "top_label": predictions[0]["label"],
        "top_score": predictions[0]["score"],
        "top_emoji": predictions[0]["emoji"],
    }


def clear_cache():
    _load_vit.cache_clear()
