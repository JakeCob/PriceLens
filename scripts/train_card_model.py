#!/usr/bin/env python3
"""
Train Pokemon card detection model.

Handles CUDA compatibility issues by offering CPU/GPU options.
"""

import argparse
from ultralytics import YOLO
from pathlib import Path


def train_model(
    data_yaml="data/datasets/pokemon-card-detection/data.yaml",
    model="yolo11n.pt",
    epochs=100,
    batch=16,
    imgsz=640,
    device="0",
    name="yolo11n_cleveland_5k",
    patience=20
):
    """
    Train YOLO model on Pokemon cards.

    Args:
        data_yaml: Path to data.yaml config
        model: Base model to start from
        epochs: Number of training epochs
        batch: Batch size
        imgsz: Image size
        device: 'cpu', 'cuda', or device number (0, 1, etc.)
        name: Training run name
        patience: Early stopping patience
    """

    print("=" * 70)
    print("Pokemon Card Detection Model Training")
    print("=" * 70)
    print()

    # Validate data file exists
    data_path = Path(data_yaml)
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_yaml}")
        print("Run: python scripts/create_training_dataset.py")
        return

    print(f"Configuration:")
    print(f"  Data: {data_yaml}")
    print(f"  Base model: {model}")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch}")
    print(f"  Image size: {imgsz}")
    print(f"  Patience: {patience}")
    print()

    if device == "cpu":
        print("⚠️  Training on CPU - This will be slow but stable")
        print("   Estimated time: 2-4 hours for 50 epochs")
        print()
    else:
        print("⚠️  Training on GPU with RTX 4070")
        print("   NOTE: CUDA errors may occur due to PyTorch 2.9.1 compatibility")
        print("   If training crashes, use --device cpu")
        print()

    # Load model
    print("Loading base model...")
    yolo_model = YOLO(model)

    # Train
    print("Starting training...")
    print("-" * 70)

    try:
        results = yolo_model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch if device != "cpu" else min(batch, 4),  # Smaller batch for CPU
            device=device,
            name=name,
            patience=patience,
            # Augmentation
            augment=True,
            degrees=15,
            translate=0.1,
            scale=0.3,
            fliplr=0.5,
            mosaic=1.0,
            # Optimizer
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            # Performance (cuDNN stability fixes)
            workers=0,  # Prevent cuDNN stream errors
            cache=True,  # RAM cache
            amp=False,  # Disable for stability
            # Output
            project='models/training_runs',
            save=True,
            plots=True,
            verbose=True
        )

        print()
        print("=" * 70)
        print("✓ Training Complete!")
        print("=" * 70)
        print(f"Best weights: runs/detect/{name}/weights/best.pt")
        print(f"Last weights: runs/detect/{name}/weights/last.pt")
        print()
        print("Copy to models directory:")
        print(f"  cp runs/detect/{name}/weights/best.pt models/pokemon_card_yolo11.pt")
        print()

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print(f"Partial weights saved in: runs/detect/{name}/weights/")

    except Exception as e:
        print(f"\n\nERROR during training: {e}")
        print(f"Partial weights may be saved in: runs/detect/{name}/weights/")

        if "CUDA" in str(e) or "illegal instruction" in str(e):
            print()
            print("CUDA error detected!")
            print("Solutions:")
            print("  1. Retry with: python scripts/train_card_model.py --device cpu")
            print("  2. Train on Google Colab (see docs/CUDA_COMPATIBILITY.md)")
            print("  3. Downgrade PyTorch to 2.1.2")


def main():
    parser = argparse.ArgumentParser(description="Train Pokemon card detection model")

    parser.add_argument(
        "--data",
        default="data/datasets/pokemon-card-detection/data.yaml",
        help="Path to data.yaml"
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Base model (yolo11n.pt, yolo11s.pt, yolo11m.pt)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size"
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Device: 'cpu', 'cuda', or device number (0, 1, etc.)"
    )
    parser.add_argument(
        "--name",
        default="yolo11n_cleveland_5k",
        help="Training run name"
    )

    args = parser.parse_args()

    train_model(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        name=args.name
    )


if __name__ == "__main__":
    main()
