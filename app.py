#!/usr/bin/env python3
"""
Facial Emotion Recognition — Production Streamlit App

Features:
- Sidebar model selection from all available .keras checkpoints
- In-app model training (runs in background subprocess)
- Dataset download
- Live webcam + image upload testing
- Optimized for Streamlit Community Cloud deployment
"""

from __future__ import annotations

import os
import io
import json
import logging
import time
import subprocess
import sys
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:  # Streamlit Cloud: TF is optional, ViT backend needs only torch
    tf = None  # type: ignore
    _TF_AVAILABLE = False

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_DIR = Path(os.getenv("DATASET_DIR", "dataset"))
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
TRAIN_LOG_PATH = Path("training_log.txt")
TRAIN_LOCK_PATH = Path("training.lock")
HF_MODEL_ID = os.getenv(
    "HF_MODEL_ID", "mo-thecreator/vit-Facial-Expression-Recognition"
)  # SOTA plug-and-play ViT (~84.9% eval, FER2013+AffectNet+MMI)

EMOTION_LABELS = [
    "angry", "disgust", "fear", "happy",
    "neutral", "sad", "surprise",
]

EMOTION_EMOJIS = {
    "angry": "😠", "disgust": "🤢", "fear": "😨",
    "happy": "😊", "neutral": "😐", "sad": "😢", "surprise": "😲",
}

EMOTION_COLORS = {
    "angry": "#ff5252", "disgust": "#69f0ae", "fear": "#e040fb",
    "happy": "#ffd740", "neutral": "#b0bec5", "sad": "#448aff", "surprise": "#ff6e40",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_running_on_streamlit_cloud() -> bool:
    """Heuristic to detect Streamlit Cloud / containerized environment."""
    return (
        os.environ.get("STREAMLIT_SHARING", "") == "true"
        or os.environ.get("STREAMLIT_CLOUD", "") == "true"
        or Path("/mount/src").exists()
    )


def get_available_models() -> list[Path]:
    """Scan working directory for all .keras model files."""
    return sorted(Path(".").glob("*.keras"), key=lambda p: p.stat().st_mtime, reverse=True)


def get_selected_model_path() -> Optional[Path]:
    """Return the currently selected model path from session state."""
    path_str = st.session_state.get("selected_model", "")
    if path_str and Path(path_str).exists():
        return Path(path_str)
    models = get_available_models()
    if models:
        st.session_state.selected_model = str(models[0])
        return models[0]
    return None


# ---------------------------------------------------------------------------
# Model loading — ViT (SOTA, default) + legacy Keras fallback
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading ViT emotion model...")
def load_vit_cached(model_id: str):
    from emotion_vit import _load_vit
    return _load_vit(model_id)


def get_backend() -> str:
    return st.session_state.get("backend", "vit")


def is_model_ready() -> bool:
    if get_backend() == "vit":
        return True  # downloaded lazily from HF on first prediction
    return get_selected_model_path() is not None


@st.cache_resource(show_spinner="Loading model...")
def load_model_cached(path: str):
    if not _TF_AVAILABLE:
        st.error("TensorFlow is not installed in this environment. Use the ViT backend.")
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return tf.keras.models.load_model(str(p))
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        return None


def get_model():
    path = get_selected_model_path()
    if path is None:
        return None
    return load_model_cached(str(path))


# ---------------------------------------------------------------------------
# Face detection
# ---------------------------------------------------------------------------
@st.cache_resource
def get_face_cascade():
    # opencv-python 5.x removed the Haar CascadeClassifier API; this app needs 4.x.
    # If you see this error, restart Streamlit after (re)installing deps:
    #   pip install "opencv-python>=4.8,<5"
    if not hasattr(cv2, "CascadeClassifier"):
        raise RuntimeError(
            f"Broken OpenCV ({getattr(cv2, '__version__', '?')}): "
            "cv2.CascadeClassifier is missing. Your running Streamlit process "
            "likely loaded opencv 5.x before it was downgraded — stop the server "
            "(Ctrl+C) and restart it. If it persists, run: "
            'pip install "opencv-python>=4.8,<5"'
        )
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )


def detect_faces(gray: np.ndarray):
    cascade = get_face_cascade()
    return cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(48, 48))


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def get_model_input_spec(model) -> tuple[int, int, int]:
    """Return (height, width, channels) expected by the model."""
    shape = model.input_shape
    return int(shape[1]), int(shape[2]), int(shape[3])


def _has_builtin_normalization(model) -> bool:
    if not _TF_AVAILABLE:
        return False
    """
    True if the model normalizes inputs itself (transfer-learning models),
    i.e. it expects raw RGB values in [0, 255]. Legacy models expect /255.
    """
    def check(layers) -> bool:
        for layer in layers:
            name = getattr(layer, "name", "").lower()
            if isinstance(layer, tf.keras.layers.Rescaling) or "rescaling" in name \
                    or "efficientnet" in name:
                return True
        return False

    if check(model.layers):
        return True
    # Also look one level deep (nested backbone sub-model)
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and check(layer.layers):
            return True
    return False


def preprocess_face(face_roi_bgr: np.ndarray, model) -> np.ndarray:
    """Resize/convert a BGR face ROI to match whatever model is selected."""
    h, w, c = get_model_input_spec(model)
    face = cv2.resize(face_roi_bgr, (w, h))
    if c == 1:
        # Legacy grayscale CNN trained on /255-scaled images
        arr = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
        arr = arr[..., np.newaxis]
    else:
        if _has_builtin_normalization(model):
            arr = face.astype("float32")  # raw [0, 255]; model normalizes internally
        else:
            arr = face.astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def predict_emotion(face_input: np.ndarray) -> dict:
    """Legacy Keras path (TTA flip-averaged). For ViT use predict_emotion_roi()."""
    model = get_model()
    if model is None:
        raise RuntimeError("No model loaded.")
    # Test-time augmentation: average prediction with the horizontal flip (+~1% acc)
    preds = model.predict(face_input, verbose=0)[0]
    flipped = model.predict(face_input[:, :, ::-1, :], verbose=0)[0]
    preds = (preds + flipped) / 2.0
    predictions = [
        {"label": label, "score": float(score), "emoji": EMOTION_EMOJIS[label]}
        for label, score in zip(EMOTION_LABELS, preds)
    ]
    predictions.sort(key=lambda x: x["score"], reverse=True)
    return {
        "predictions": predictions,
        "top_label": predictions[0]["label"],
        "top_score": predictions[0]["score"],
        "top_emoji": predictions[0]["emoji"],
    }


def predict_emotion_roi(roi_bgr: np.ndarray) -> dict:
    """Route a BGR face ROI to the active backend (ViT default, Keras fallback)."""
    if get_backend() == "vit":
        from emotion_vit import predict_emotion_vit
        model_id = st.session_state.get("hf_model_id", HF_MODEL_ID)
        return predict_emotion_vit(roi_bgr, model_id=model_id)
    model = get_model()
    if model is None:
        raise RuntimeError("No model loaded.")
    return predict_emotion(preprocess_face(roi_bgr, model))


def annotate_frame(frame: np.ndarray) -> tuple:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)
    results = []
    keras_model = get_model() if get_backend() == "keras" else None
    for (x, y, w, h) in faces:
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        roi = frame[y:y + h, x:x + w]  # keep color; preprocessing adapts per model
        if get_backend() == "vit":
            result = predict_emotion_roi(roi)
        else:
            face_input = preprocess_face(roi, keras_model)
            result = predict_emotion(face_input)
        results.append(result)
        text = f"{result['top_label']} {result['top_emoji']} {result['top_score'] * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x, y - th - 10), (x + tw, y), color, -1)
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return frame, results


def predict_from_image(np_img_bgr: np.ndarray) -> tuple:
    frame = np_img_bgr.copy()
    annotated, results = annotate_frame(frame)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return annotated_rgb, results


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def dataset_exists() -> bool:
    train_dir = DATASET_DIR / "train"
    if not train_dir.exists():
        return False
    # Need at least one subdir with images
    for sub in train_dir.iterdir():
        if sub.is_dir() and any(sub.glob("*.png")):
            return True
    return False


def count_dataset_images() -> dict:
    stats = {}
    for split in ("train", "validation", "test"):
        split_path = DATASET_DIR / split
        if not split_path.exists():
            continue
        count = 0
        for emotion_dir in split_path.iterdir():
            if emotion_dir.is_dir():
                count += len(list(emotion_dir.glob("*.png"))) + len(list(emotion_dir.glob("*.jpg")))
        stats[split] = count
    return stats


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    st.sidebar.title("🧠 Emotion Recognition")
    st.sidebar.markdown("SOTA ViT facial emotion detection (HF) + legacy Keras fallback.")
    st.sidebar.divider()

    # --- Backend selection (SOTA first) ---
    st.sidebar.subheader("🧠 Backend")
    backend = st.sidebar.radio(
        "Prediction engine",
        ["vit", "keras"],
        format_func=lambda x: "✨ ViT SOTA (~84.9%) — Recommended"
        if x == "vit" else "🗂️ Legacy Keras (.keras)",
        index=0 if st.session_state.get("backend", "vit") == "vit" else 1,
        key="backend_radio",
    )
    st.session_state.backend = backend
    if backend == "vit":
        hf_id = st.sidebar.text_input("HF model", value=st.session_state.get("hf_model_id", HF_MODEL_ID))
        st.session_state.hf_model_id = hf_id
        from emotion_vit import ensure_vit_snapshot, is_vit_cached_locally
        if is_vit_cached_locally(hf_id):
            st.sidebar.caption("💾 Weights cached on disk (`./models/`) — fast/offline load")
        elif st.sidebar.button("⬇️ Cache model on disk (~350MB, one-time)", key="cache_vit"):
            with st.spinner("Downloading ViT weights (one-time)..."):
                ensure_vit_snapshot(hf_id)
            load_vit_cached.clear()
            st.rerun()
        try:
            load_vit_cached(hf_id)
            st.sidebar.success("✅ ViT loaded")
            st.sidebar.caption(f"`{hf_id}` — ViT-Base, 7 emotions, 224×224")
        except Exception as exc:
            st.sidebar.warning(f"⏳ ViT downloads on first use ({exc!s:.120})")
    st.sidebar.divider()

    # --- Model Management (legacy Keras) ---
    st.sidebar.subheader("📁 Legacy Keras Models")
    models = get_available_models()

    model_names = [str(m.name) for m in models]
    options = model_names if model_names else ["No models found"]

    current = st.session_state.get("selected_model", options[0] if options else "No models found")
    if current not in options:
        current = options[0]

    selected = st.sidebar.radio(
        "Select model",
        options,
        index=options.index(current) if current in options else 0,
        key="model_radio",
    )

    if selected in model_names:
        if st.session_state.get("selected_model") != selected:
            st.session_state.selected_model = selected
            # Clear cached model so next run loads the new one
            load_model_cached.clear()
            st.rerun()

        model = get_model()
        if model is not None:
            st.sidebar.success("✅ Model loaded")
            st.sidebar.caption(
                f"`{selected}` — {model.output_shape[-1]} classes, "
                f"{len(model.layers):,} layers, {model.count_params():,} params"
            )
        else:
            st.sidebar.error("❌ Load failed")
    else:
        st.sidebar.warning("⚠️ No model available")
        st.sidebar.caption("Train or upload a `.keras` model.")

    st.sidebar.divider()

    # --- Dataset ---
    st.sidebar.subheader("📥 Dataset")
    if dataset_exists():
        counts = count_dataset_images()
        total = sum(counts.values())
        st.sidebar.success(f"✅ Dataset ready ({total:,} images)")
        for split, c in counts.items():
            st.sidebar.caption(f"  {split}: {c:,}")
    else:
        st.sidebar.error("❌ No dataset found")
        if st.sidebar.button("⬇️ Download FER2013", width="stretch", key="dl_dataset"):
            with st.spinner("Downloading dataset..."):
                proc = subprocess.run(
                    [sys.executable, "download_dataset.py"],
                    capture_output=True,
                    text=True,
                )
                if proc.returncode == 0:
                    st.sidebar.success("Dataset downloaded")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.sidebar.error(f"Download failed: {proc.stderr[:500]}")

    st.sidebar.divider()

    # --- Training ---
    st.sidebar.subheader("🏋️ Train Model")
    if TRAIN_LOCK_PATH.exists():
        st.sidebar.info("⏳ Training in progress...")
        if st.sidebar.button("🔄 Refresh Status", width="stretch", key="refresh_status"):
            st.rerun()
    else:
        backbone = st.sidebar.select_slider(
            "Backbone", options=["mobilenetv2", "efficientnetb0"], value="mobilenetv2"
        )
        epochs = st.sidebar.slider("Epochs", 5, 200, 60, 5)
        batch_size = st.sidebar.select_slider("Batch size", options=[16, 32, 64], value=32)

        train_disabled = not dataset_exists()
        if train_disabled:
            st.sidebar.caption("📥 Download dataset first to enable training.")

        if st.sidebar.button(
            "▶️ Start Training",
            type="primary",
            width="stretch",
            disabled=train_disabled,
        ):
            # Clear old log and start training subprocess
            TRAIN_LOG_PATH.write_text("")
            TRAIN_LOCK_PATH.write_text(str(datetime.now()))
            model_name = f"emotion_model_{datetime.now():%Y%m%d_%H%M%S}"
            cmd = [
                sys.executable,
                "train_model.py",
                "--epochs", str(epochs),
                "--batch-size", str(batch_size),
                "--backbone", backbone,
                "--model", f"{model_name}.keras",
                "--history", f"training_history_{model_name}.json",
            ]
            subprocess.Popen(
                cmd,
                stdout=open(TRAIN_LOG_PATH, "w"),
                stderr=subprocess.STDOUT,
            )
            st.rerun()

    st.sidebar.divider()

    # --- Emotions legend ---
    st.sidebar.subheader("📋 Emotions")
    for label in EMOTION_LABELS:
        st.sidebar.markdown(
            f"<span style='color:{EMOTION_COLORS[label]}'>●</span> "
            f"{EMOTION_EMOJIS[label]} **{label.capitalize()}**",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main: Status banner
# ---------------------------------------------------------------------------
def render_status_banner():
    ready = is_model_ready()
    dataset_ok = dataset_exists()
    on_cloud = is_running_on_streamlit_cloud()

    cols = st.columns(3)
    with cols[0]:
        if ready:
            label = "ViT SOTA" if get_backend() == "vit" else "Keras"
            st.success(f"✅ Model Ready ({label})")
        else:
            st.error("❌ No Model")
    with cols[1]:
        if dataset_ok:
            st.success("✅ Dataset Ready")
        else:
            st.error("❌ No Dataset")
    with cols[2]:
        if on_cloud:
            st.info("☁️ Running on Cloud (no webcam)")
        else:
            st.info("💻 Local Mode (webcam available)")

    st.divider()


# ---------------------------------------------------------------------------
# Main: Training Monitor
# ---------------------------------------------------------------------------
def render_training_monitor():
    if not TRAIN_LOCK_PATH.exists():
        return

    st.header("🏋️ Training Monitor")

    # Show training log
    if TRAIN_LOG_PATH.exists():
        log_text = TRAIN_LOG_PATH.read_text()
        st.code(log_text if log_text else "Waiting for output...", language="bash")

    # Poll if still running
    # A simple heuristic: if log hasn't changed in 60s and lock exists, maybe done
    # Better: check if training process is alive by trying to update lock file timestamp?
    # Simpler: just provide a "Check if done" button
    if st.button("🔄 Refresh Log", key="refresh_training"):
        st.rerun()

    # Detect completion by checking for "Model saved" in log
    if TRAIN_LOG_PATH.exists() and "Model saved to" in TRAIN_LOG_PATH.read_text():
        TRAIN_LOCK_PATH.unlink(missing_ok=True)
        st.success("🎉 Training complete!")
        st.balloons()
        if st.button("Reload app to use new model"):
            st.rerun()


# ---------------------------------------------------------------------------
# Tab: Live Camera
# ---------------------------------------------------------------------------
def tab_live_camera():
    st.header("📷 Live Camera")
    st.caption("Real-time webcam emotion detection with face bounding boxes.")

    on_cloud = is_running_on_streamlit_cloud()
    if on_cloud:
        st.warning("📵 Webcam is unavailable on Streamlit Community Cloud. Run locally to use the camera.")
        st.info("You can still test with the **Upload Image** tab.")
        return

    if not is_model_ready():
        st.warning("Model not loaded. Train or select a model first.")
        return

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("▶️ Start Camera", disabled=st.session_state.camera_running, width="stretch", key="start_cam"):
            st.session_state.camera_running = True
            st.session_state.cam_frames = 0
            st.rerun()
    with col2:
        if st.button("⏹️ Stop Camera", disabled=not st.session_state.camera_running, width="stretch", key="stop_cam"):
            st.session_state.camera_running = False
            st.rerun()
    with col3:
        max_frames = st.slider(
            "Max frames", 30, 600, 300, 30,
            disabled=st.session_state.camera_running
        )

    feed_placeholder = st.empty()
    stats_placeholder = st.empty()

    if st.session_state.camera_running:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        progress = st.progress(0.0)
        frame_counter = 0

        try:
            while st.session_state.camera_running and frame_counter < max_frames:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame from camera.")
                    break

                annotated, results = annotate_frame(frame)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                feed_placeholder.image(annotated_rgb, width="stretch")

                if results:
                    top = results[0]
                    st.session_state.emotion_history.append(
                        {"label": top["top_label"], "score": top["top_score"]}
                    )

                frame_counter += 1
                st.session_state.cam_frames = frame_counter
                progress.progress(min(frame_counter / max_frames, 1.0))
                time.sleep(0.03)
        finally:
            cap.release()
            st.session_state.camera_running = False
            progress.empty()
            st.rerun()
    else:
        st.info("Press **Start Camera** to begin real-time detection.")

    with stats_placeholder.container():
        if st.session_state.emotion_history:
            st.subheader("Recent Detections")
            df = pd.DataFrame(st.session_state.emotion_history)
            counts = df["label"].value_counts().reindex(EMOTION_LABELS, fill_value=0)
            st.bar_chart(counts, width="stretch")


# ---------------------------------------------------------------------------
# Tab: Upload Image
# ---------------------------------------------------------------------------
def tab_upload():
    st.header("🖼️ Upload Image")
    st.caption("Upload a photo to detect emotions.")

    if not is_model_ready():
        st.warning("Model not loaded. Train or select a model first.")
        return

    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp", "bmp"])

    if uploaded is not None:
        pil_img = Image.open(uploaded).convert("RGB")
        np_img = np.array(pil_img)
        frame = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

        annotated_rgb, results = predict_from_image(frame)

        col1, col2 = st.columns(2)
        with col1:
            st.image(pil_img, caption="Original", width="stretch")
        with col2:
            st.image(annotated_rgb, caption="Detected", width="stretch")

        st.session_state.last_upload_result = results

        if not results:
            st.warning("No faces detected in the image.")
            return

        st.subheader(f"Detected {len(results)} face(s)")

        for idx, result in enumerate(results):
            with st.expander(
                f"Face #{idx + 1} — {result['top_emoji']} {result['top_label'].capitalize()}",
                expanded=(idx == 0),
            ):
                top = result["predictions"][0]
                cols = st.columns([1, 3])
                with cols[0]:
                    st.metric("Top Emotion", top["emoji"], f"{top['score'] * 100:.1f}%")
                with cols[1]:
                    for pred in result["predictions"]:
                        label = pred["label"]
                        score = pred["score"]
                        st.progress(
                            score,
                            text=f"{pred['emoji']} {label.capitalize()} — {score * 100:.1f}%",
                        )


# ---------------------------------------------------------------------------
# Tab: Model Info
# ---------------------------------------------------------------------------
def tab_model_info():
    st.header("📊 Model & Dataset")

    # Dataset stats
    st.subheader("Dataset Statistics")
    stats = []
    for split in ("train", "validation", "test"):
        split_path = DATASET_DIR / split
        if not split_path.exists():
            continue
        for emotion_dir in sorted(split_path.iterdir()):
            if emotion_dir.is_dir():
                count = len(list(emotion_dir.glob("*.png"))) + len(list(emotion_dir.glob("*.jpg")))
                stats.append({"split": split, "emotion": emotion_dir.name, "count": count})

    if stats:
        df_stats = pd.DataFrame(stats)
        df_pivot = df_stats.pivot(index="emotion", columns="split", values="count").fillna(0).astype(int)
        st.bar_chart(df_pivot, width="stretch")
        st.dataframe(df_pivot, width="stretch")
    else:
        st.info("No dataset found. Download via the sidebar or run `python download_dataset.py`.")

    st.divider()

    # Model summary
    st.subheader("Model Summary")
    if get_backend() == "vit":
        hf_id = st.session_state.get("hf_model_id", HF_MODEL_ID)
        st.json({
            "backend": "ViT SOTA (HuggingFace)",
            "model_id": hf_id,
            "architecture": "ViT-Base patch16-224",
            "training_data": "FER2013 + AffectNet + MMI",
            "eval_accuracy": "~84.3-84.9%",
            "input": "224x224 RGB",
            "classes": EMOTION_LABELS,
        })
        st.info("ViT is the active backend. Switch to Legacy Keras below to inspect .keras checkpoints.")
    model = get_model()
    if model is not None:
        path = get_selected_model_path()
        st.json({
            "file": str(path.name) if path else None,
            "input_shape": list(model.input_shape),
            "output_shape": list(model.output_shape),
            "layers": len(model.layers),
            "trainable_params": f"{model.count_params():,}",
        })

        arch = []
        for layer in model.layers:
            try:
                if hasattr(layer, "output_shape") and layer.output_shape is not None:
                    output_shape = str(layer.output_shape)
                elif hasattr(layer, "output"):
                    output_shape = str(layer.output.shape)
                else:
                    output_shape = "?"
            except Exception:
                output_shape = "?"
            arch.append({
                "Layer": layer.name,
                "Type": layer.__class__.__name__,
                "Output Shape": output_shape,
            })
        st.dataframe(pd.DataFrame(arch), width="stretch")
    else:
        st.info("No model loaded. Train a model from the sidebar.")

    st.divider()

    st.subheader("Available Checkpoints")
    models = get_available_models()
    if models:
        model_data = []
        for m in models:
            stat = m.stat()
            model_data.append({
                "Model": m.name,
                "Size (MB)": f"{stat.st_size / 1_048_576:.2f}",
                "Modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            })
        st.dataframe(pd.DataFrame(model_data), width="stretch")
    else:
        st.info("No checkpoints found.")

    st.divider()

    st.subheader("Training History")
    history_files = sorted(Path(".").glob("training_history*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if history_files:
        selected_hist = st.selectbox("Select history file", history_files, format_func=lambda p: p.name)
        with open(selected_hist) as f:
            hist = json.load(f)
        df_hist = pd.DataFrame(hist)
        df_hist["epoch"] = df_hist.index + 1
        st.line_chart(df_hist.set_index("epoch")[["accuracy", "val_accuracy"]], width="stretch")
        st.line_chart(df_hist.set_index("epoch")[["loss", "val_loss"]], width="stretch")
    else:
        st.info("No training history found.")


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
def init_state():
    defaults = {
        "camera_running": False,
        "cam_frames": 0,
        "emotion_history": deque(maxlen=50),
        "last_upload_result": None,
        "selected_model": "",
        "backend": "vit",
        "hf_model_id": HF_MODEL_ID,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Emotion Recognition",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_state()
    render_sidebar()
    render_status_banner()

    if TRAIN_LOCK_PATH.exists():
        render_training_monitor()
        st.divider()

    st.title("Facial Emotion Recognition")
    st.caption("End-to-end deep learning system with automatic dataset download, CNN training, and real-time detection.")

    tab_live, tab_up, tab_info = st.tabs(
        ["📷 Live Camera", "🖼️ Upload Image", "📊 Model & Dataset"]
    )

    with tab_live:
        tab_live_camera()
    with tab_up:
        tab_upload()
    with tab_info:
        tab_model_info()

    st.divider()
    st.caption("Built with TensorFlow + OpenCV + Streamlit · Deploy-ready for Streamlit Community Cloud")


if __name__ == "__main__":
    main()
