#!/usr/bin/env python3
"""
Emotion Recognition Model Trainer (Transfer Learning)

Fine-tunes an ImageNet-pretrained backbone (MobileNetV2 or EfficientNetB0) on the
FER2013 dataset (7 emotions). Key accuracy improvements over the previous
from-scratch CNN:

- Transfer learning from ImageNet weights (+8-15% accuracy)
- Data augmentation as in-graph Keras layers (active only during training)
- Label smoothing 0.1 (FER2013 labels are noisy)
- SGD with Nesterov momentum + ReduceLROnPlateau (best combo for FER2013)
- Two-phase training: frozen backbone, then fine-tune top blocks at low LR
- Class weighting for imbalanced classes (disgust has only ~436 images)

Normalization is baked INTO the model graph, so saved models accept raw
RGB pixel values in [0, 255]. The Streamlit app detects this automatically.

Saves training history as JSON for visualization.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    BatchNormalization,
    RandomFlip,
    RandomRotation,
    RandomTranslation,
    RandomZoom,
    RandomContrast,
    Rescaling,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

EMOTION_LABELS = [
    "angry", "disgust", "fear", "happy",
    "neutral", "sad", "surprise",
]

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 60
BACKBONES = ("mobilenetv2", "efficientnetb0")
DEFAULT_BACKBONE = "mobilenetv2"

DATASET_DIR = Path("dataset")
MODEL_PATH = Path("emotion_model.keras")
HISTORY_PATH = Path("training_history.json")

# Fraction of backbone layers (from the top) to unfreeze in phase 2.
FINE_TUNE_FRACTION = 0.3


def get_backbone(name: str, image_size: tuple):
    """Return a fresh ImageNet-pretrained base model."""
    input_shape = (*image_size, 3)
    if name == "mobilenetv2":
        from tensorflow.keras.applications import MobileNetV2
        return MobileNetV2(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
    if name == "efficientnetb0":
        from tensorflow.keras.applications import EfficientNetB0
        return EfficientNetB0(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
    raise ValueError(f"Unknown backbone: {name}")


def build_model(num_classes: int, backbone: str = DEFAULT_BACKBONE,
                image_size: tuple = IMAGE_SIZE) -> tf.keras.Model:
    """
    Build the transfer-learning model.

    Augmentation and normalization live inside the model graph:
    - Augmentation layers are active only when the model is training.
    - Normalization is included, so the saved model takes raw RGB [0, 255].
    """
    inputs = tf.keras.Input(shape=(*image_size, 3), name="input_image")

    # In-graph data augmentation (no-ops at inference time)
    x = RandomFlip("horizontal", name="aug_flip")(inputs)
    x = RandomRotation(0.1, name="aug_rotation")(x)
    x = RandomTranslation(0.15, 0.15, name="aug_translation")(x)
    x = RandomZoom(0.2, name="aug_zoom")(x)
    x = RandomContrast(0.15, name="aug_contrast")(x)

    # Backbone-specific input normalization (baked into the saved model)
    if backbone == "efficientnetb0":
        pass  # EfficientNet includes preprocessing internally (raw [0,255] input)
    else:  # mobilenetv2 expects [-1, 1]
        x = Rescaling(1.0 / 127.5, offset=-1.0, name="rescale_mobilenet")(x)

    base = get_backbone(backbone, image_size)
    base.trainable = False  # Phase 1: frozen feature extractor
    x = base(x, training=False)  # keep BatchNorm in inference mode

    x = GlobalAveragePooling2D(name="gap")(x)
    x = Dropout(0.3, name="head_dropout_1")(x)
    x = Dense(256, activation="relu", kernel_regularizer=l2(1e-4),
              name="head_dense_256")(x)
    x = BatchNormalization(name="head_bn")(x)
    x = Dropout(0.4, name="head_dropout_2")(x)
    outputs = Dense(num_classes, activation="softmax", name="predictions")(x)

    return Model(inputs, outputs, name=f"fer_{backbone}_transfer")


def unfreeze_top_blocks(model: tf.keras.Model, fraction: float = FINE_TUNE_FRACTION):
    """Unfreeze the top `fraction` of backbone layers (BatchNorm stays frozen)."""
    base_layer = next(
        (l for l in model.layers if isinstance(l, tf.keras.Model)), None
    )
    if base_layer is None:
        logger.warning("No nested backbone found; skipping fine-tune unfreeze.")
        return 0
    n = max(1, int(len(base_layer.layers) * fraction))
    unfrozen = 0
    for layer in base_layer.layers[-n:]:
        if isinstance(layer, BatchNormalization):
            continue  # keep BN statistics frozen
        layer.trainable = True
        unfrozen += 1
    logger.info("Unfroze %d/%d backbone layers for fine-tuning.", unfrozen,
                len(base_layer.layers))
    return unfrozen


def make_callbacks(model_path: Path, patience: int) -> list:
    return [
        EarlyStopping(monitor="val_loss", patience=patience,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(model_path), monitor="val_accuracy",
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                          min_lr=1e-6, verbose=1),
    ]


def compute_class_weights(train_dir: Path, class_indices: dict) -> dict:
    """Compute balanced class weights from training data."""
    labels = []
    for class_name, class_idx in class_indices.items():
        class_dir = train_dir / class_name
        if class_dir.exists():
            count = len(list(class_dir.glob("*.png"))) + len(list(class_dir.glob("*.jpg")))
            labels.extend([class_idx] * count)

    if not labels:
        logger.warning("No training images found. Returning equal weights.")
        return {i: 1.0 for i in range(len(class_indices))}

    classes = np.array(sorted(class_indices.values()))
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=np.array(labels)
    )
    return {int(cls): float(w) for cls, w in zip(classes, weights)}


def train(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    dataset_dir: Path = DATASET_DIR,
    model_path: Path = MODEL_PATH,
    history_path: Path = HISTORY_PATH,
    image_size: tuple = IMAGE_SIZE,
    backbone: str = DEFAULT_BACKBONE,
):
    """Two-phase transfer-learning training loop."""
    logger.info("Starting transfer-learning training (backbone=%s)...", backbone)

    # No rescale here — normalization happens inside the model.
    datagen = ImageDataGenerator()
    train_generator = datagen.flow_from_directory(
        str(dataset_dir / "train"),
        target_size=image_size,
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )
    validation_generator = datagen.flow_from_directory(
        str(dataset_dir / "validation"),
        target_size=image_size,
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    test_generator = datagen.flow_from_directory(
        str(dataset_dir / "test"),
        target_size=image_size,
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    logger.info("Class indices: %s", train_generator.class_indices)
    num_classes = len(train_generator.class_indices)

    class_weights = compute_class_weights(dataset_dir / "train", train_generator.class_indices)
    logger.info("Class weights: %s", class_weights)

    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    history_all = {}

    # ------------------------------------------------------------------
    # Phase 1: train classification head on a frozen backbone
    # ------------------------------------------------------------------
    model = build_model(num_classes=num_classes, backbone=backbone,
                        image_size=image_size)
    model.summary()

    phase1_epochs = max(5, round(epochs * 0.25))
    phase2_epochs = max(5, epochs - phase1_epochs)

    model.compile(
        optimizer=SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
        loss=loss,
        metrics=["accuracy"],
    )

    logger.info("--- Phase 1: training head (%d epochs, backbone frozen) ---", phase1_epochs)
    hist1 = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=phase1_epochs,
        callbacks=make_callbacks(model_path, patience=6),
        class_weight=class_weights,
    )
    for k, v in hist1.history.items():
        history_all.setdefault(k, []).extend(float(x) for x in v)

    # ------------------------------------------------------------------
    # Phase 2: fine-tune top backbone blocks at a low learning rate
    # ------------------------------------------------------------------
    unfreeze_top_blocks(model)
    model.compile(
        optimizer=SGD(learning_rate=1e-4, momentum=0.9, nesterov=True),
        loss=loss,
        metrics=["accuracy"],
    )

    logger.info("--- Phase 2: fine-tuning (%d epochs, top %.0f%% of backbone) ---",
                phase2_epochs, FINE_TUNE_FRACTION * 100)
    hist2 = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=phase2_epochs,
        callbacks=make_callbacks(model_path, patience=10),
        class_weight=class_weights,
    )
    for k, v in hist2.history.items():
        history_all.setdefault(k, []).extend(float(x) for x in v)

    test_loss, test_accuracy = model.evaluate(test_generator)
    logger.info("Test accuracy: %.4f", test_accuracy)
    logger.info("Test loss: %.4f", test_loss)

    with open(history_path, "w") as f:
        json.dump(history_all, f, indent=2)
    logger.info("Training history saved to %s", history_path)

    model.save(str(model_path))
    logger.info("Model saved to %s", model_path)

    return history_all


def main():
    parser = argparse.ArgumentParser(description="Train emotion recognition model (transfer learning)")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Total epochs across both phases (25%% phase 1 / 75%% phase 2)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--dataset", type=str, default=str(DATASET_DIR))
    parser.add_argument("--model", type=str, default=str(MODEL_PATH))
    parser.add_argument("--history", type=str, default=str(HISTORY_PATH))
    parser.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE,
                        choices=BACKBONES,
                        help="ImageNet-pretrained backbone to fine-tune")
    parser.add_argument("--img-size", type=int, default=IMAGE_SIZE[0],
                        help="Square input resolution (backbone pretrained at 224)")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        dataset_dir=Path(args.dataset),
        model_path=Path(args.model),
        history_path=Path(args.history),
        image_size=(args.img_size, args.img_size),
        backbone=args.backbone,
    )


if __name__ == "__main__":
    main()
