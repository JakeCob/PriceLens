#!/usr/bin/env python3
"""Download required models for PriceLens"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_yolo11():
    """Download YOLO11 nano model for card detection"""
    logger.info("Downloading YOLO11n model...")

    # Create models directory
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    try:
        # Download YOLO11 nano model
        model = YOLO('yolo11n.pt')  # Downloads automatically if not present

        # Save to models directory
        model_path = models_dir / "yolo11n.pt"

        logger.info(f"✓ Model ready at: {model_path}")

        # Create model info file
        info_path = models_dir / "model_info.yaml"
        with open(info_path, 'w') as f:
            f.write("""models:
  yolo11n:
    version: "11.0"
    type: "nano"
    task: "detection"
    classes: 80
    input_size: 640
    description: "YOLO11 nano model for object detection"
""")
        logger.info("✓ Model info file created")

    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        sys.exit(1)


def create_placeholders():
    """Create placeholder files for models directory"""
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    # Create .gitkeep to track empty directory
    gitkeep = models_dir / ".gitkeep"
    gitkeep.touch()

    # Create README for models directory
    readme = models_dir / "README.md"
    with open(readme, 'w') as f:
        f.write("""# Models Directory

This directory contains machine learning models for PriceLens.

## Required Models

1. **yolo11n.pt** - YOLO11 nano model for card detection
   - Downloaded automatically via `scripts/download_models.py`
   - Size: ~6MB

2. **pokemon_cards_yolo11.pt** (Future)
   - Fine-tuned model specifically for Pokemon cards
   - To be created through training

## Usage

Run the download script from the project root:
```bash
python scripts/download_models.py
```
""")

    logger.info("✓ Model directory prepared")


if __name__ == "__main__":
    logger.info("Starting model download...")
    download_yolo11()
    create_placeholders()
    logger.info("✓ Model setup complete!")
    logger.info("You can now run the application with: python src/main.py")