#!/usr/bin/env python3
"""
Test Roboflow Pokemon card detection model.

Quick validation before training our own model.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from roboflow import Roboflow
import cv2

# Load environment
load_dotenv()
api_key = os.getenv('ROBOFLOW_API_KEY')

if not api_key:
    print("ERROR: ROBOFLOW_API_KEY not found in .env")
    exit(1)

print("=" * 70)
print("Testing Roboflow Pokemon Card Detection Model")
print("=" * 70)
print()

# Initialize Roboflow
print("Connecting to Roboflow...")
rf = Roboflow(api_key=api_key)
project = rf.workspace("pokemon-scanner").project("pokemon-card-detector-cuyon")
model = project.version(1).model

print("Model loaded: pokemon-card-detector-cuyon v1")
print()

# Test cases
test_cases = []

# Case 1: Real Pokemon card
card_images = list(Path("data/card_database").rglob("*.jpg"))
if card_images:
    test_cases.append(("Pokemon Card", str(card_images[0])))

# Case 2: Test images from Roboflow dataset
roboflow_test = list(Path("Pokemon-Card-Detector-1/test/images").glob("*.jpg"))
if roboflow_test:
    test_cases.append(("Roboflow Test", str(roboflow_test[0])))

print(f"Test Cases: {len(test_cases)}")
print()

# Run tests
for name, img_path in test_cases:
    print(f"Testing: {name}")
    print(f"  Image: {Path(img_path).name}")

    try:
        # Run inference via Roboflow API
        prediction = model.predict(img_path, confidence=40)

        # Get predictions
        predictions = prediction.json()['predictions']

        print(f"  Detections: {len(predictions)}")

        if predictions:
            for i, pred in enumerate(predictions[:3], 1):  # Show first 3
                print(f"    {i}. {pred['class']}: {pred['confidence']:.2f}")

        # Save visualization
        output_path = f"test_output_{name.replace(' ', '_').lower()}.jpg"
        prediction.save(output_path)
        print(f"  Saved: {output_path}")

    except Exception as e:
        print(f"  ERROR: {e}")

    print()

print("=" * 70)
print("Test Complete!")
print("=" * 70)
print()
print("Analysis:")
print("  - If detections work on cards: ✓ Use Roboflow model")
print("  - If no detections or errors: ✗ Train custom model")
