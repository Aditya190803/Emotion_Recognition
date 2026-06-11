#!/usr/bin/env python3
"""
FER2013 Dataset Downloader

Downloads the FER2013 facial emotion recognition dataset from public mirrors,
extracts the CSV, and organizes images into train/validation/test folders.

Emotion mapping (FER2013):
    0: angry
    1: disgust
    2: fear
    3: happy
    4: sad
    5: surprise
    6: neutral
"""

import os
import sys
import io
import zipfile
import csv
import shutil
import argparse
import logging
from pathlib import Path

import numpy as np
import requests
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

EMOTION_MAP = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}

DATASET_URLS = [
    "https://github.com/nicolo-felicioni/fer2013/raw/master/fer2013.csv",
    "https://github.com/nageshsinghc4/Exploratory-data-analysis/raw/master/fer2013.csv",
    "https://raw.githubusercontent.com/karansjc1/emotion-detection/master/fer2013.csv",
]

DATASET_DIR = Path("dataset")
IMAGE_SIZE = (48, 48)


def ensure_dirs():
    """Create train/validation/test folders for each emotion."""
    for split in ("train", "validation", "test"):
        for emotion in EMOTION_MAP.values():
            (DATASET_DIR / split / emotion).mkdir(parents=True, exist_ok=True)
    logger.info("Directory structure ready.")


def download_csv(url: str, timeout: int = 60) -> bytes:
    """Download CSV content from URL."""
    logger.info("Downloading from %s", url)
    response = requests.get(url, timeout=timeout, stream=True)
    response.raise_for_status()
    return response.content


def write_image_from_pixels(pixel_str: str, dest_path: Path):
    """Convert a space-separated pixel string to a grayscale PNG image."""
    pixels = np.array(pixel_str.split(), dtype=np.uint8)
    if pixels.size != 48 * 48:
        logger.warning("Invalid pixel count %d for %s", pixels.size, dest_path)
        return False
    img = Image.fromarray(pixels.reshape(48, 48), mode="L")
    img.save(dest_path)
    return True


def process_csv(csv_bytes: bytes) -> dict:
    """Parse CSV and return organized file paths."""
    text = csv_bytes.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))

    counts = {"train": 0, "validation": 0, "test": 0}

    for row in reader:
        emotion_idx = int(row["emotion"])
        pixels = row["pixels"]
        usage = row["Usage"].strip().lower()

        # Normalize usage names
        if "train" in usage and "public" not in usage:
            split = "train"
        elif "public" in usage or "val" in usage:
            split = "validation"
        elif "test" in usage:
            split = "test"
        else:
            split = "train"

        emotion_name = EMOTION_MAP.get(emotion_idx)
        if emotion_name is None:
            continue

        dest_dir = DATASET_DIR / split / emotion_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{split}_{counts[split]:05d}.png"
        dest_path = dest_dir / filename

        if write_image_from_pixels(pixels, dest_path):
            counts[split] += 1

    return counts


def generate_synthetic_data(samples_per_class: int = 100):
    """
    Fallback: generate synthetic 48x48 grayscale images so the pipeline still works.
    Images are random noise — sufficient to verify the training / UI flow.
    """
    logger.warning("Generating synthetic demo data (random noise images).")
    for split, count in (("train", samples_per_class), ("validation", samples_per_class // 4), ("test", samples_per_class // 4)):
        for emotion_idx, emotion_name in EMOTION_MAP.items():
            dest_dir = DATASET_DIR / split / emotion_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                noise = np.random.randint(0, 256, (48, 48), dtype=np.uint8)
                img = Image.fromarray(noise, mode="L")
                img.save(dest_dir / f"{split}_{i:05d}.png")
    return {"train": samples_per_class * 7, "validation": (samples_per_class // 4) * 7, "test": (samples_per_class // 4) * 7}


def download_dataset(prefer_synthetic: bool = False):
    """Main entry point: download or generate the FER2013 dataset."""
    if prefer_synthetic:
        counts = generate_synthetic_data()
        logger.info("Synthetic dataset created: %s", counts)
        return counts

    # Try downloading from mirrors
    csv_bytes = None
    for url in DATASET_URLS:
        try:
            csv_bytes = download_csv(url)
            logger.info("Downloaded CSV (%d bytes)", len(csv_bytes))
            break
        except Exception as exc:
            logger.warning("Failed to download from %s: %s", url, exc)

    if csv_bytes is None:
        logger.error("All download sources failed. Falling back to synthetic data.")
        counts = generate_synthetic_data()
        logger.info("Synthetic dataset created: %s", counts)
        return counts

    ensure_dirs()
    counts = process_csv(csv_bytes)
    logger.info("Dataset prepared: %s", counts)
    return counts


def print_summary():
    """Print dataset folder contents summary."""
    for split in ("train", "validation", "test"):
        print(f"\n[{split}]")
        split_path = DATASET_DIR / split
        if not split_path.exists():
            continue
        for emotion_dir in sorted(split_path.iterdir()):
            if emotion_dir.is_dir():
                count = len(list(emotion_dir.glob("*.png")))
                print(f"  {emotion_dir.name:12s}: {count} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FER2013 dataset")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic demo data instead of downloading",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        default=True,
        help="Print dataset summary after download",
    )
    args = parser.parse_args()

    download_dataset(prefer_synthetic=args.synthetic)
    if args.summary:
        print_summary()
