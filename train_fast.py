#!/usr/bin/env python3
"""
Fast transfer-learning trainer for FER2013.

Loads the whole dataset into RAM (FER2013 is small at native 48x48), resizes
to 224x224 RGB on the fly in a tf.data pipeline, and fine-tunes an
ImageNet-pretrained backbone using the two-phase strategy from train_model.py.
Avoids the disk-I/O bottleneck of ImageDataGenerator (~2x faster per epoch).

Usage:
    python train_fast.py [--epochs 48] [--batch-size 32] [--backbone efficientnetb0]
"""

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from train_model import build_model, unfreeze_top_blocks, make_callbacks, FINE_TUNE_FRACTION

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET_DIR = Path("dataset")
MODEL_PATH = Path("emotion_model_transfer.keras")
HISTORY_PATH = Path("training_history_transfer.json")


def load_split(split: str):
    """Load all images of a split as (N, 48, 48) uint8 grayscale + labels."""
    root = DATASET_DIR / split
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    xs, ys = [], []
    for ci, cname in enumerate(classes):
        for p in sorted((root / cname).glob("*.png")):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            xs.append(img)
            ys.append(ci)
    X = np.stack(xs)[:, :, :, None]  # (N, 48, 48, 1)
    y = np.array(ys, dtype=np.int32)
    logger.info("%s: %d images, classes=%s", split, len(X), classes)
    return X, y, classes


def make_dataset(X, y, batch_size: int, training: bool) -> tf.data.Dataset:
    def _prep(img, label):
        img = tf.image.resize(img, (224, 224), method="bilinear")
        img = tf.image.grayscale_to_rgb(img)  # replicate gray -> 3 channels
        img = tf.cast(img, tf.float32)  # raw [0,255]; model normalizes internally
        label = tf.one_hot(label, 7)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(len(X), reshuffle_each_iteration=True)
    ds = ds.map(_prep, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=48, help="Total epochs (25%% phase 1 / 75%% phase 2)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--backbone", type=str, default="efficientnetb0",
                    choices=("mobilenetv2", "efficientnetb0"))
    args = ap.parse_args()

    t0 = time.time()
    Xtr, ytr, classes = load_split("train")
    Xva, yva, _ = load_split("validation")
    logger.info("Dataset loaded in %.1fs", time.time() - t0)

    cw_arr = compute_class_weight("balanced", classes=np.arange(7), y=ytr)
    class_weights = {i: float(w) for i, w in enumerate(cw_arr)}
    logger.info("Class weights: %s", class_weights)

    train_ds = make_dataset(Xtr, ytr, args.batch_size, training=True)
    val_ds = make_dataset(Xva, yva, args.batch_size, training=False)

    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    history_all = {}

    phase1_epochs = max(5, round(args.epochs * 0.25))
    phase2_epochs = max(5, args.epochs - phase1_epochs)

    # ---- Phase 1: frozen backbone ----
    model = build_model(7, backbone=args.backbone)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
                  loss=loss, metrics=["accuracy"])
    logger.info("--- Phase 1: %d epochs (backbone frozen) ---", phase1_epochs)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=phase1_epochs,
                   callbacks=make_callbacks(MODEL_PATH, patience=6),
                   class_weight=class_weights)
    for k, v in h1.history.items():
        history_all.setdefault(k, []).extend(float(x) for x in v)

    # ---- Phase 2: fine-tune top blocks ----
    unfreeze_top_blocks(model)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9, nesterov=True),
                  loss=loss, metrics=["accuracy"])
    logger.info("--- Phase 2: %d epochs (top %.0f%% unfrozen) ---",
                phase2_epochs, FINE_TUNE_FRACTION * 100)
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=phase2_epochs,
                   callbacks=make_callbacks(MODEL_PATH, patience=10),
                   class_weight=class_weights)
    for k, v in h2.history.items():
        history_all.setdefault(k, []).extend(float(x) for x in v)

    # ---- Final test evaluation + per-class report ----
    Xte, yte, classes = load_split("test")
    test_ds = make_dataset(Xte, yte, args.batch_size, training=False)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    logger.info("TEST ACCURACY: %.4f", test_acc)

    probs = model.predict(test_ds, verbose=0)
    preds = probs.argmax(axis=1)
    for ci, cname in enumerate(classes):
        mask = yte == ci
        logger.info("  %-10s: %5.1f%%  (n=%d)", cname, (preds[mask] == ci).mean() * 100, int(mask.sum()))

    with open(HISTORY_PATH, "w") as f:
        json.dump(history_all, f, indent=2)
    model.save(MODEL_PATH)
    logger.info("Model saved to %s — DONE", MODEL_PATH)


if __name__ == "__main__":
    main()
