#!/usr/bin/env python3
"""
Monitor YOLO training progress.
Reads the results.csv file and displays current status.
"""

import os
import time
from pathlib import Path

def monitor_training(training_dir):
    """Monitor training progress from results.csv"""
    results_file = Path(training_dir) / "results.csv"

    if not results_file.exists():
        print(f"❌ Training results not found: {results_file}")
        return

    # Read results
    with open(results_file, 'r') as f:
        lines = f.readlines()

    if len(lines) < 2:
        print("⏳ Training just started, no results yet...")
        return

    # Parse header and latest results
    header = lines[0].strip().split(',')
    latest = lines[-1].strip().split(',')

    # Create dict
    results = dict(zip(header, latest))

    # Display progress
    epoch = int(results.get('epoch', 0).strip())
    total_epochs = 100  # From config

    print("=" * 70)
    print(f"YOLO11n Pokemon Card Detector Training - RTX 4070 GPU")
    print("=" * 70)
    print(f"📊 Epoch: {epoch}/{total_epochs} ({epoch/total_epochs*100:.1f}%)")
    print()

    # Metrics
    print("📈 Training Metrics:")
    metrics = {
        'train/box_loss': 'Box Loss',
        'train/cls_loss': 'Class Loss',
        'train/dfl_loss': 'DFL Loss',
        'metrics/precision(B)': 'Precision',
        'metrics/recall(B)': 'Recall',
        'metrics/mAP50(B)': 'mAP@50',
        'metrics/mAP50-95(B)': 'mAP@50-95'
    }

    for key, label in metrics.items():
        value = results.get(key, 'N/A')
        if value != 'N/A':
            try:
                value = float(value.strip())
                print(f"  {label:20s}: {value:.4f}")
            except:
                pass

    print()
    print(f"📁 Weights: models/training_runs/yolo11n_cleveland_full/weights/")
    print(f"   - best.pt   (best validation performance)")
    print(f"   - last.pt   (latest checkpoint)")
    print()

    # Check if training is complete
    if epoch >= total_epochs:
        print("✅ TRAINING COMPLETE!")
        print()

    # Estimate remaining time
    if epoch > 0:
        # Rough estimate: ~4-5 minutes per epoch
        remaining_epochs = total_epochs - epoch
        eta_minutes = remaining_epochs * 4.5
        eta_hours = eta_minutes / 60
        print(f"⏱️  Estimated time remaining: {eta_hours:.1f} hours")


if __name__ == "__main__":
    import sys

    training_dir = sys.argv[1] if len(sys.argv) > 1 else "models/training_runs/yolo11n_cleveland_full"

    try:
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            monitor_training(training_dir)
            print()
            print("Press Ctrl+C to exit monitoring...")
            time.sleep(30)  # Update every 30 seconds
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
