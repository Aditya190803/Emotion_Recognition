#!/usr/bin/env python3
"""
Emotion Recognition Model Trainer

Trains a CNN on the FER2013 dataset (7 emotions).
Supports data augmentation, class weights, model checkpointing, and learning rate scheduling.
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
    Flatten,
    BatchNormalization,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam
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

IMAGE_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 50
DATASET_DIR = Path("dataset")
MODEL_PATH = Path("emotion_model.keras")
HISTORY_PATH = Path("training_history.json")


def build_model(num_classes: int = 7) -> tf.keras.Model:
    """Build and return the CNN model."""
    model = Sequential()

    # Block 1
    model.add(Conv2D(32, (3, 3), activation="relu", padding="same",
                     kernel_regularizer=l2(1e-4), input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # Block 2
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # Block 3
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # Block 4
    model.add(Conv2D(256, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # Fully Connected
    model.add(Flatten())
    model.add(Dense(256, activation="relu", kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(128, activation="relu", kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Output
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


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
):
    """Main training loop."""
    logger.info("Starting training...")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        shear_range=0.1,
        fill_mode="nearest",
    )
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        str(dataset_dir / "train"),
        target_size=image_size,
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )
    validation_generator = val_test_datagen.flow_from_directory(
        str(dataset_dir / "validation"),
        target_size=image_size,
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    test_generator = val_test_datagen.flow_from_directory(
        str(dataset_dir / "test"),
        target_size=image_size,
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    logger.info("Class indices: %s", train_generator.class_indices)
    num_classes = len(train_generator.class_indices)

    class_weights = compute_class_weights(dataset_dir / "train", train_generator.class_indices)
    logger.info("Class weights: %s", class_weights)

    model = build_model(num_classes=num_classes)
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(model_path), monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    test_loss, test_accuracy = model.evaluate(test_generator)
    logger.info("Test accuracy: %.4f", test_accuracy)
    logger.info("Test loss: %.4f", test_loss)

    # Save history
    hist_data = {
        k: [float(v) for v in vals]
        for k, vals in history.history.items()
    }
    with open(history_path, "w") as f:
        json.dump(hist_data, f, indent=2)
    logger.info("Training history saved to %s", history_path)

    model.save(str(model_path))
    logger.info("Model saved to %s", model_path)

    return history


def main():
    parser = argparse.ArgumentParser(description="Train emotion recognition CNN")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--dataset", type=str, default=str(DATASET_DIR))
    parser.add_argument("--model", type=str, default=str(MODEL_PATH))
    parser.add_argument("--history", type=str, default=str(HISTORY_PATH))
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        dataset_dir=Path(args.dataset),
        model_path=Path(args.model),
        history_path=Path(args.history),
    )


if __name__ == "__main__":
    main()
