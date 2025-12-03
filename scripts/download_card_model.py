#!/usr/bin/env python3
"""
Download pre-trained Pokemon card detection model from Roboflow.

This script downloads a Pokemon card-specific YOLO model from Roboflow,
which should provide better accuracy than the generic COCO-trained model.

Usage:
    python scripts/download_card_model.py

Requirements:
    - Roboflow API key (get free key from https://roboflow.com)
    - Set ROBOFLOW_API_KEY environment variable or provide in .env file
"""

import os
import sys
from pathlib import Path
from roboflow import Roboflow
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def download_pokemon_card_model(api_key: str = None):
    """
    Download Pokemon card detection model from Roboflow.

    Args:
        api_key: Roboflow API key. If None, will try to get from environment.

    Returns:
        Path to downloaded model weights
    """
    # Load environment variables
    load_dotenv()

    # Get API key
    if api_key is None:
        api_key = os.getenv('ROBOFLOW_API_KEY')

    if not api_key:
        print("ERROR: Roboflow API key not found!")
        print("\nPlease either:")
        print("1. Set ROBOFLOW_API_KEY environment variable")
        print("2. Add ROBOFLOW_API_KEY to your .env file")
        print("3. Get a free API key from https://roboflow.com")
        sys.exit(1)

    print("=" * 60)
    print("Pokemon Card Model Downloader")
    print("=" * 60)
    print()

    # Initialize Roboflow
    print(f"Initializing Roboflow...")
    rf = Roboflow(api_key=api_key)

    # Download Pokemon Card Detector
    # Using the publicly available Pokemon card detection model
    print(f"Accessing Pokemon card detector project...")
    try:
        project = rf.workspace("pokemon-scanner").project("pokemon-card-detector-cuyon")
        version = project.version(1)

        print(f"Project: {project.name}")
        print(f"Version: {version.version}")
        print()

        # Get model information
        print("Model Information:")
        print(f"  - Model type: YOLOv11")
        print(f"  - Classes: {version.model.classes if hasattr(version.model, 'classes') else 'Pokemon cards'}")
        print()

        # Download model
        print("Downloading model...")
        dataset = version.download("yolov11")

        # Create models directory if it doesn't exist
        models_dir = project_root / "models"
        models_dir.mkdir(exist_ok=True)

        # Find the weights file
        dataset_path = Path(dataset.location)
        weights_file = None

        # Look for weights in common locations
        possible_weights = [
            dataset_path / "weights" / "best.pt",
            dataset_path / "train" / "weights" / "best.pt",
            dataset_path / "best.pt",
        ]

        for weight_path in possible_weights:
            if weight_path.exists():
                weights_file = weight_path
                break

        if weights_file is None:
            print(f"\nWARNING: Could not find weights file in standard locations.")
            print(f"Dataset downloaded to: {dataset_path}")
            print(f"Please manually move the weights file to: {models_dir / 'pokemon_card_yolo11.pt'}")
            return None

        # Copy weights to models directory
        import shutil
        target_path = models_dir / "pokemon_card_yolo11.pt"
        shutil.copy(weights_file, target_path)

        print()
        print("=" * 60)
        print("✓ Model downloaded successfully!")
        print("=" * 60)
        print(f"Model saved to: {target_path}")
        print(f"Dataset saved to: {dataset_path}")
        print()
        print("Next steps:")
        print("1. Update your config to use: models/pokemon_card_yolo11.pt")
        print("2. Test the model with: python scripts/test_detector_images.py")
        print()

        return target_path

    except Exception as e:
        print(f"\nERROR: Failed to download model: {e}")
        print("\nTroubleshooting:")
        print("1. Verify your API key is correct")
        print("2. Check your internet connection")
        print("3. Try accessing https://universe.roboflow.com/pokemon-scanner/pokemon-card-detector-cuyon")
        print("4. You may need to fork the project to your own workspace first")
        print()
        print("Alternative: Download manually from Roboflow Universe and place in models/")
        sys.exit(1)


def verify_model(model_path: Path):
    """Verify the downloaded model works with YOLO."""
    try:
        from ultralytics import YOLO

        print("Verifying model...")
        model = YOLO(str(model_path))

        print("\nModel Details:")
        print(f"  - Classes: {model.names}")
        print(f"  - Number of classes: {len(model.names)}")
        print()

        print("✓ Model verification successful!")
        return True

    except Exception as e:
        print(f"\nWARNING: Model verification failed: {e}")
        print("The model file may still be valid, but couldn't be loaded.")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download Pokemon card detection model from Roboflow"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="Roboflow API key (optional, can use env variable)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the model after downloading"
    )

    args = parser.parse_args()

    # Download model
    model_path = download_pokemon_card_model(api_key=args.api_key)

    # Verify if requested
    if model_path and args.verify:
        print()
        verify_model(model_path)
