#!/usr/bin/env python3
"""
Create synthetic test images with card-like rectangles for testing
"""

import cv2
import numpy as np
from pathlib import Path


def create_synthetic_cards():
    """Create synthetic card images for testing"""
    print("Creating synthetic test images...")

    output_dir = Path("test_images")
    output_dir.mkdir(exist_ok=True)

    # Test 1: Single card (vertical)
    print("Creating test 1: Single vertical card...")
    img1 = np.ones((720, 1280, 3), dtype=np.uint8) * 200  # Light gray background
    # Draw a card-like rectangle (aspect ~0.71)
    cv2.rectangle(img1, (400, 150), (650, 500), (255, 255, 255), -1)  # White card
    cv2.rectangle(img1, (400, 150), (650, 500), (0, 0, 0), 3)  # Black border
    # Add some text to simulate card
    cv2.putText(
        img1, "POKEMON", (450, 220), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2
    )
    cv2.putText(
        img1, "Pikachu", (450, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
    )
    cv2.imwrite(str(output_dir / "test_single_card.jpg"), img1)
    print("  ✓ Saved: test_single_card.jpg")

    # Test 2: Multiple cards
    print("Creating test 2: Multiple cards...")
    img2 = np.ones((720, 1280, 3), dtype=np.uint8) * 200
    # Card 1
    cv2.rectangle(img2, (150, 150), (350, 500), (255, 200, 200), -1)
    cv2.rectangle(img2, (150, 150), (350, 500), (0, 0, 0), 3)
    # Card 2
    cv2.rectangle(img2, (450, 150), (650, 500), (200, 255, 200), -1)
    cv2.rectangle(img2, (450, 150), (650, 500), (0, 0, 0), 3)
    # Card 3
    cv2.rectangle(img2, (750, 150), (950, 500), (200, 200, 255), -1)
    cv2.rectangle(img2, (750, 150), (950, 500), (0, 0, 0), 3)
    cv2.imwrite(str(output_dir / "test_multiple_cards.jpg"), img2)
    print("  ✓ Saved: test_multiple_cards.jpg")

    # Test 3: Tilted card
    print("Creating test 3: Tilted card...")
    img3 = np.ones((720, 1280, 3), dtype=np.uint8) * 200
    # Create rotated rectangle
    center = (640, 360)
    size = (250, 350)
    angle = 15  # degrees
    rect = ((center[0], center[1]), (size[0], size[1]), angle)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    cv2.fillPoly(img3, [box], (255, 255, 200))
    cv2.polylines(img3, [box], True, (0, 0, 0), 3)
    cv2.imwrite(str(output_dir / "test_tilted_card.jpg"), img3)
    print("  ✓ Saved: test_tilted_card.jpg")

    # Test 4: Small cards
    print("Creating test 4: Small cards (distance simulation)...")
    img4 = np.ones((720, 1280, 3), dtype=np.uint8) * 200
    # Small card 1
    cv2.rectangle(img4, (300, 250), (400, 400), (255, 255, 255), -1)
    cv2.rectangle(img4, (300, 250), (400, 400), (0, 0, 0), 2)
    # Small card 2
    cv2.rectangle(img4, (600, 250), (700, 400), (255, 255, 255), -1)
    cv2.rectangle(img4, (600, 250), (700, 400), (0, 0, 0), 2)
    cv2.imwrite(str(output_dir / "test_small_cards.jpg"), img4)
    print("  ✓ Saved: test_small_cards.jpg")

    # Test 5: No cards (negative test)
    print("Creating test 5: No cards (empty background)...")
    img5 = np.ones((720, 1280, 3), dtype=np.uint8) * 200
    # Just some random shapes (not card-like)
    cv2.circle(img5, (400, 300), 100, (100, 100, 255), -1)
    cv2.rectangle(img5, (700, 200), (900, 250), (255, 100, 100), -1)  # Too wide
    cv2.imwrite(str(output_dir / "test_no_cards.jpg"), img5)
    print("  ✓ Saved: test_no_cards.jpg")

    print(f"\n✅ Created 5 synthetic test images in {output_dir}/")
    print("\nRun: python scripts/test_detector_images.py")


if __name__ == "__main__":
    create_synthetic_cards()
