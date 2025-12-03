#!/usr/bin/env python3
"""
Create YOLO training dataset from card_database images.

Since data/card_database contains individual card images (not scenes),
we can auto-generate bounding box annotations - each image is one card
covering the entire image.
"""

import os
import shutil
from pathlib import Path
import random
import yaml
from PIL import Image

def create_yolo_dataset(
    source_dir="data/card_database",
    output_dir="data/yolo_training",
    train_split=0.8,
    val_split=0.15,
    # test_split=0.05 (calculated)
):
    """
    Create YOLO dataset from card images.

    Args:
        source_dir: Directory containing card images
        output_dir: Output directory for YOLO dataset
        train_split: Fraction for training set
        val_split: Fraction for validation set
    """

    print("=" * 60)
    print("Creating YOLO Training Dataset")
    print("=" * 60)
    print()

    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # Find all card images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(source_path.rglob(ext))

    print(f"Found {len(image_files)} card images")

    if len(image_files) == 0:
        print("ERROR: No images found!")
        return

    # Shuffle images
    random.seed(42)  # For reproducibility
    random.shuffle(image_files)

    # Calculate split indices
    n_total = len(image_files)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)

    train_images = image_files[:n_train]
    val_images = image_files[n_train:n_train + n_val]
    test_images = image_files[n_train + n_val:]

    print(f"\nDataset splits:")
    print(f"  Train: {len(train_images)} images ({train_split*100:.0f}%)")
    print(f"  Val:   {len(val_images)} images ({val_split*100:.0f}%)")
    print(f"  Test:  {len(test_images)} images ({(1-train_split-val_split)*100:.0f}%)")
    print()

    # Create directory structure
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Process each split
    def process_split(images, split_name):
        print(f"Processing {split_name} split...")

        for img_path in images:
            try:
                # Load image to get dimensions
                img = Image.open(img_path)
                width, height = img.size

                # Copy image
                img_dest = output_path / split_name / 'images' / img_path.name
                shutil.copy(img_path, img_dest)

                # Create YOLO annotation (entire image is the card)
                # YOLO format: <class_id> <x_center> <y_center> <width> <height>
                # All normalized to 0-1
                # For entire image: class=0, center=(0.5, 0.5), size=(1.0, 1.0)

                label_path = output_path / split_name / 'labels' / f"{img_path.stem}.txt"
                with open(label_path, 'w') as f:
                    # Class 0 (pokemon_card), center at (0.5, 0.5), full size (1.0, 1.0)
                    f.write("0 0.5 0.5 1.0 1.0\n")

            except Exception as e:
                print(f"  Warning: Failed to process {img_path.name}: {e}")
                continue

        print(f"  ✓ {split_name}: {len(images)} images processed")

    process_split(train_images, 'train')
    process_split(val_images, 'val')
    process_split(test_images, 'test')

    # Create data.yaml
    data_yaml = {
        'path': str(output_path.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 1,  # Number of classes
        'names': ['pokemon_card']  # Class names
    }

    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print()
    print("=" * 60)
    print("✓ Dataset created successfully!")
    print("=" * 60)
    print(f"Location: {output_path.absolute()}")
    print(f"Config: {yaml_path.absolute()}")
    print()
    print("Next steps:")
    print(f"1. Train: python scripts/train_card_model.py")
    print(f"2. Or manually: yolo detect train data={yaml_path} model=yolo11n.pt epochs=50")
    print()


if __name__ == "__main__":
    create_yolo_dataset()
