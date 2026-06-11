#!/usr/bin/env python3
"""
Real-Time Emotion Detection (Standalone Script)

Opens webcam, detects faces, and overlays predicted emotions.
Compatible with both .h5 and .keras model formats.

Usage:
    python realtime_prediction.py [--model PATH] [--camera INDEX]
"""

import argparse
import logging
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
    logger.info("Model loaded. Output classes: %d -> %s", num_classes, labels)
    return model, labels


def preprocess_face(roi_gray: np.ndarray) -> np.ndarray:
    """Resize and normalize face ROI for model input."""
    face = cv2.resize(roi_gray, (48, 48))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=-1)
    return face


def draw_prediction(frame, x, y, w, h, label, score, emoji):
    """Draw bounding box and prediction text on frame."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    text = f"{label} {emoji}  {score * 100:.1f}%"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x, y - th - 10), (x + tw, y), (0, 255, 0), -1)
    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


def main():
    parser = argparse.ArgumentParser(description="Real-time emotion detection")
    parser.add_argument("--model", type=str, default="emotion_model.keras", help="Path to Keras model")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    args = parser.parse_args()

    model_path = Path(args.model)
    model, labels = load_model_and_labels(model_path)

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
            roi_gray = gray[y:y + h, x:x + w]
            face_input = preprocess_face(roi_gray)
            preds = model.predict(face_input, verbose=0)[0]

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
