#!/usr/bin/env python3
"""
Facial Emotion Recognition — ViT SOTA Streamlit App.

Single-backend app: HuggingFace ViT
(mo-thecreator/vit-Facial-Expression-Recognition, ~84.9% eval).
No training, no local Keras checkpoints, no sidebar.
"""

from __future__ import annotations

import os
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

HF_MODEL_ID = os.getenv(
    "HF_MODEL_ID", "mo-thecreator/vit-Facial-Expression-Recognition"
)

EMOTION_LABELS = [
    "angry", "disgust", "fear", "happy",
    "neutral", "sad", "surprise",
]

EMOTION_EMOJIS = {
    "angry": "😠", "disgust": "🤢", "fear": "😨",
    "happy": "😊", "neutral": "😐", "sad": "😢", "surprise": "😲",
}


def is_running_on_cloud() -> bool:
    return (
        os.environ.get("STREAMLIT_SHARING", "") == "true"
        or os.environ.get("STREAMLIT_CLOUD", "") == "true"
        or Path("/mount/src").exists()
    )


@st.cache_resource(show_spinner="Loading ViT emotion model...")
def load_vit_cached(model_id: str):
    from emotion_vit import _load_vit
    return _load_vit(model_id)


@st.cache_resource
def get_face_cascade():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )


def predict_roi(roi_bgr: np.ndarray) -> dict:
    from emotion_vit import predict_emotion_vit
    return predict_emotion_vit(roi_bgr, model_id=HF_MODEL_ID)


def annotate_frame(frame: np.ndarray) -> tuple:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = get_face_cascade().detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(48, 48)
    )
    results = []
    for (x, y, w, h) in faces:
        roi = frame[y:y + h, x:x + w]
        result = predict_roi(roi)
        results.append(result)
        text = f"{result['top_label']} {result['top_emoji']} {result['top_score'] * 100:.0f}%"
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x, y - th - 10), (x + tw, y), color, -1)
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), results


def render_result(result: dict, expanded: bool):
    with st.expander(
        f"{result['top_emoji']} {result['top_label'].capitalize()} "
        f"— {result['top_score'] * 100:.1f}%",
        expanded=expanded,
    ):
        for pred in result["predictions"]:
            st.progress(
                pred["score"],
                text=f"{pred['emoji']} {pred['label'].capitalize()} — {pred['score'] * 100:.1f}%",
            )


def tab_upload():
    st.header("🖼️ Upload Image")
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp", "bmp"])
    if uploaded is None:
        return
    pil_img = Image.open(uploaded).convert("RGB")
    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    annotated, results = annotate_frame(frame)

    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_img, caption="Original", width="stretch")
    with col2:
        st.image(annotated, caption="Detected", width="stretch")

    if not results:
        st.warning("No faces detected in the image.")
        return
    st.subheader(f"Detected {len(results)} face(s)")
    for idx, result in enumerate(results):
        render_result(result, expanded=(idx == 0))


def tab_live_camera():
    st.header("📷 Live Camera")
    if is_running_on_cloud():
        st.warning("📵 Webcam is unavailable on Streamlit Community Cloud. Run locally to use the camera.")
        return

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start Camera", disabled=st.session_state.camera_running, width="stretch"):
            st.session_state.camera_running = True
            st.rerun()
    with col2:
        if st.button("⏹️ Stop Camera", disabled=not st.session_state.camera_running, width="stretch"):
            st.session_state.camera_running = False
            st.rerun()

    if not st.session_state.camera_running:
        st.info("Press **Start Camera** to begin real-time detection.")
    else:
        feed = st.empty()
        cap = cv2.VideoCapture(int(os.getenv("CAMERA_INDEX", "0")))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        try:
            for _ in range(300):
                if not st.session_state.camera_running:
                    break
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame from camera.")
                    break
                annotated, results = annotate_frame(frame)
                feed.image(annotated, width="stretch")
                for r in results:
                    st.session_state.emotion_history.append(r["top_label"])
                time.sleep(0.03)
        finally:
            cap.release()
            st.session_state.camera_running = False

    if st.session_state.emotion_history:
        st.subheader("Recent Detections")
        counts = Counter(st.session_state.emotion_history)
        total = sum(counts.values())
        for label in EMOTION_LABELS:
            st.progress(
                counts[label] / total,
                text=f"{EMOTION_EMOJIS[label]} {label.capitalize()} — {counts[label]}",
            )


def tab_about():
    st.header("📊 Model")
    st.json({
        "backend": "ViT SOTA (HuggingFace)",
        "model_id": HF_MODEL_ID,
        "architecture": "ViT-Base patch16-224",
        "training_data": "FER2013 + AffectNet + MMI",
        "eval_accuracy": "~84.3-84.9%",
        "input": "224x224 RGB",
        "classes": EMOTION_LABELS,
    })
    st.markdown(
        " ".join(f"{EMOTION_EMOJIS[l]} **{l.capitalize()}**" for l in EMOTION_LABELS)
    )


def main():
    st.set_page_config(page_title="Emotion Recognition", page_icon="🧠", layout="wide")

    if "camera_running" not in st.session_state:
        st.session_state.camera_running = False
    if "emotion_history" not in st.session_state:
        st.session_state.emotion_history = deque(maxlen=50)

    try:
        load_vit_cached(HF_MODEL_ID)
        st.success("✅ ViT model ready")
    except Exception as exc:
        st.warning(f"⏳ Model downloads on first use ({str(exc)[:120]})")

    st.title("Facial Emotion Recognition")
    st.caption("State-of-the-art ViT facial emotion detection.")

    tab_live, tab_up, tab_info = st.tabs(
        ["📷 Live Camera", "🖼️ Upload Image", "📊 Model"]
    )
    with tab_live:
        tab_live_camera()
    with tab_up:
        tab_upload()
    with tab_info:
        tab_about()


if __name__ == "__main__":
    main()
