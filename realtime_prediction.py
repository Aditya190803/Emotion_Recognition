#!/usr/bin/env python3
"""
Real-Time Emotion Detection (Standalone Script)

Opens webcam, detects faces, and overlays predicted emotions.
Compatible with both .h5 and .keras model formats.

Usage:
    python realtime_prediction.py [--model PATH] [--camera INDEX] [--tta]

Works with both legacy 48x48 grayscale CNNs and new 224x224 RGB
transfer-learning models (input size/channels are read from the model).
"""

import argparse
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# All possible labels — the model will use as many as its output dimension
ALL_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


def load_model_and_labels(model_path: Path):
    """Load Keras model and infer emotion labels from output shape."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    logger.info("Loading model from %s", model_path)
    model = tf.keras.models.load_model(str(model_path))
    num_classes = model.output_shape[-1]
    labels = ALL_LABELS[:num_classes]
    h, w, c = (int(v) for v in model.input_shape[1:4])

    def has_builtin_normalization(layers) -> bool:
        for layer in layers:
            name = getattr(layer, "name", "").lower()
            if isinstance(layer, tf.keras.layers.Rescaling) or "rescaling" in name \
                    or "efficientnet" in name:
                return True
            if isinstance(layer, tf.keras.Model) and has_builtin_normalization(layer.layers):
                return True
        return False

    normalize_input = not has_builtin_normalization(model.layers)
    logger.info(
        "Model input: %dx%dx%d (%s)", h, w, c,
        "raw RGB, normalization built in" if not normalize_input else "/255-normalized",
    )
    return model, labels, (h, w, c), normalize_input


def preprocess_face(roi_bgr: np.ndarray, input_spec: tuple,
                    normalize_input: bool) -> np.ndarray:
    """Resize/convert a BGR face ROI to match the loaded model."""
    h, w, c = input_spec
    face = cv2.resize(roi_bgr, (w, h))
    if c == 1:
        arr = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
        arr = arr[..., np.newaxis]
    elif normalize_input:
        arr = face.astype("float32") / 255.0
    else:
        arr = face.astype("float32")  # raw [0, 255]; model normalizes internally
    return np.expand_dims(arr, axis=0)


def draw_prediction(frame, x, y, w, h, label, score, emoji):
    """Draw bounding box and prediction text on frame."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    text = f"{label} {emoji}  {score * 100:.1f}%"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x, y - th - 10), (x + tw, y), (0, 255, 0), -1)
    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


def main():
    parser = argparse.ArgumentParser(description="Real-time emotion detection (ViT SOTA default)")
    parser.add_argument("--model", type=str, default="emotion_model.keras", help="Path to Keras model (keras backend only)")
    parser.add_argument("--backend", type=str, default="vit", choices=("vit", "keras"),
                        help="vit = mo-thecreator ViT SOTA (~84.9%%), keras = legacy .keras")
    parser.add_argument("--hf-model", type=str,
                        default=os.getenv("HF_MODEL_ID", "mo-thecreator/vit-Facial-Expression-Recognition"))
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--tta", action="store_true",
                        help="Test-time augmentation (flip averaging); more accurate but ~2x slower")
    parser.add_argument("--cache-model", action="store_true",
                        help="Download ViT weights to ./models and exit (no webcam needed)")
    args = parser.parse_args()

    if args.cache_model:
        from emotion_vit import ensure_vit_snapshot
        path = ensure_vit_snapshot(args.hf_model)
        logger.info("ViT weights cached at %s", path)
        return

    use_vit = args.backend == "vit"
    vit_predict = None
    if use_vit:
        try:
            from emotion_vit import predict_emotion_vit
            vit_predict = predict_emotion_vit
            logger.info("Using ViT SOTA backend: %s", args.hf_model)
        except Exception as exc:
            logger.warning("ViT load failed (%s), falling back to Keras.", exc)
            use_vit = False

    if not use_vit:
        model_path = Path(args.model)
        model, labels, input_spec, normalize_input = load_model_and_labels(model_path)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cam = cv2.VideoCapture(args.camera)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv2.CAP_PROP_FPS, 30)

    logger.info("Press ESC to exit.")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(48, 48))

        for (x, y, w, h) in faces:
            roi_bgr = frame[y:y + h, x:x + w]
            if use_vit:
                result = vit_predict(roi_bgr, model_id=args.hf_model)
                label, score = result["top_label"], result["top_score"]
            else:
                face_input = preprocess_face(roi_bgr, input_spec, normalize_input)
                preds = model.predict(face_input, verbose=0)[0]
                if args.tta:
                    flipped = model.predict(face_input[:, :, ::-1, :], verbose=0)[0]
                    preds = (preds + flipped) / 2.0
                max_idx = int(np.argmax(preds))
                label = labels[max_idx]
                score = float(np.max(preds))
            emoji_map = {
                "angry": "😠", "disgust": "🤢", "fear": "😨",
                "happy": "😊", "neutral": "😐", "sad": "😢", "surprise": "😲",
            }
            emoji = emoji_map.get(label, "")
            draw_prediction(frame, x, y, w, h, label, score, emoji)

        cv2.imshow("Emotion Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
    logger.info("Exited.")


if __name__ == "__main__":
    main()
