#!/usr/bin/env python3
"""
Facial Emotion Recognition — ViT SOTA Streamlit App.

Single-backend app: HuggingFace ViT
(mo-thecreator/vit-Facial-Expression-Recognition, ~84.9% eval).
No training, no local Keras checkpoints, no sidebar.
"""

from __future__ import annotations

import os

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


def draw_box(frame_bgr: np.ndarray, x: int, y: int, w: int, h: int, label: str, emoji: str, score: float):
    text = f"{label} {emoji} {score * 100:.0f}%"
    color = (0, 255, 0)
    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame_bgr, (x, y - th - 10), (x + tw, y), color, -1)
    cv2.putText(frame_bgr, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


def annotate_frame_bgr(frame: np.ndarray) -> tuple:
    """Detect + ViT-predict every face. Returns (annotated BGR, results)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = get_face_cascade().detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(48, 48)
    )
    results = []
    for (x, y, w, h) in faces:
        result = predict_roi(frame[y:y + h, x:x + w])
        results.append(result)
        draw_box(frame, x, y, w, h,
                 result["top_label"], result["top_emoji"], result["top_score"])
    return frame, results


def annotate_frame(frame: np.ndarray) -> tuple:
    """Same as above but returns RGB (for st.image)."""
    annotated_bgr, results = annotate_frame_bgr(frame)
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), results


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
    st.caption("Streams from your browser camera — works locally and on Streamlit Cloud. Allow camera access when prompted.")
    try:
        from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer
    except ImportError:
        st.error("Live camera needs the `streamlit-webrtc` package (not installed).")
        return
    try:
        load_vit_cached(HF_MODEL_ID)  # warm up so first frames aren't blank
    except Exception:
        pass

    class EmotionVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self._frame_idx = 0
            self._cached_boxes: list = []  # (x, y, w, h, label, emoji, score)

        def recv(self, frame):
            import av

            img = frame.to_ndarray(format="bgr24")
            self._frame_idx += 1
            # ViT on CPU is slow: full inference every 5th frame, reuse boxes between.
            if self._frame_idx % 5 == 1:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = get_face_cascade().detectMultiScale(
                    gray, scaleFactor=1.3, minNeighbors=5, minSize=(48, 48)
                )
                boxes = []
                for (x, y, w, h) in faces:
                    try:
                        r = predict_roi(img[y:y + h, x:x + w])
                        boxes.append((x, y, w, h, r["top_label"], r["top_emoji"], r["top_score"]))
                    except Exception:
                        continue
                self._cached_boxes = boxes
            for (x, y, w, h, label, emoji, score) in self._cached_boxes:
                draw_box(img, int(x), int(y), int(w), int(h), label, emoji, float(score))
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="emotion-live",
        video_processor_factory=EmotionVideoProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
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
