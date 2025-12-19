#!/usr/bin/env python3
"""
Test OCR preprocessing for shadow/poor lighting conditions.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.identification.feature_matcher import FeatureMatcher


def simulate_shadow(image: np.ndarray, intensity: float = 0.4) -> np.ndarray:
    """
    Simulate a shadow across part of the image.

    Args:
        image: Input BGR image
        intensity: Shadow darkness (0 = no shadow, 1 = black)

    Returns:
        Image with simulated shadow
    """
    result = image.copy()
    h, w = result.shape[:2]

    # Create gradient mask for shadow (darker on left side)
    mask = np.zeros((h, w), dtype=np.float32)
    for x in range(w // 2):
        mask[:, x] = 1.0 - (x / (w // 2)) * (1 - intensity)

    # Apply shadow
    for c in range(3):
        result[:, :, c] = (result[:, :, c] * (1 - mask * (1 - intensity))).astype(np.uint8)

    return result


def simulate_low_light(image: np.ndarray, factor: float = 0.3) -> np.ndarray:
    """
    Simulate low light conditions by darkening the image.

    Args:
        image: Input BGR image
        factor: Brightness factor (0 = black, 1 = original)

    Returns:
        Darkened image
    """
    return (image * factor).astype(np.uint8)


def test_ocr_with_preprocessing():
    """Test OCR with and without preprocessing on degraded images."""

    # Initialize matcher with OCR
    print("Initializing FeatureMatcher with OCR...")
    matcher = FeatureMatcher(use_ocr=True, use_vector_db=False)

    # Load database
    db_path = Path("data/me1_me2_features.pkl")
    if db_path.exists():
        matcher.load_database(str(db_path))
    else:
        print(f"Database not found: {db_path}")
        return

    # Load a test card image
    test_images = [
        "data/card_database/me1/Alakazam_me1-1.jpg",
        "data/card_database/me1/Charizard_me1-3.jpg",
        "data/card_database/base1/Charizard_base1-4.jpg",
    ]

    test_image_path = None
    for path in test_images:
        if Path(path).exists():
            test_image_path = path
            break

    if not test_image_path:
        print("No test image found")
        return

    print(f"\nTest image: {test_image_path}")
    original = cv2.imread(test_image_path)

    if original is None:
        print("Failed to load image")
        return

    print(f"Image size: {original.shape}")

    # Test scenarios
    scenarios = [
        ("Original (good lighting)", original),
        ("Simulated shadow", simulate_shadow(original, 0.4)),
        ("Low light (30%)", simulate_low_light(original, 0.3)),
        ("Very low light (15%)", simulate_low_light(original, 0.15)),
    ]

    print("\n" + "="*60)
    print("OCR PREPROCESSING TEST RESULTS")
    print("="*60)

    for name, image in scenarios:
        print(f"\n--- {name} ---")

        # Test OCR WITHOUT preprocessing (raw image)
        print("  Without preprocessing:")
        try:
            raw_result = matcher.ocr_reader.readtext(image)
            raw_text = " ".join([text[1] for text in raw_result]).lower()
            print(f"    OCR text: '{raw_text[:60]}...'")
        except Exception as e:
            print(f"    OCR failed: {e}")
            raw_text = ""

        # Test OCR WITH preprocessing
        print("  With preprocessing:")
        try:
            preprocessed = matcher._preprocess_for_ocr(image)
            proc_result = matcher.ocr_reader.readtext(preprocessed)
            proc_text = " ".join([text[1] for text in proc_result]).lower()
            print(f"    OCR text: '{proc_text[:60]}...'")
        except Exception as e:
            print(f"    OCR failed: {e}")
            proc_text = ""

        # Check for card name in detected text
        card_name = Path(test_image_path).stem.split("_")[0].lower()
        raw_found = card_name in raw_text
        proc_found = card_name in proc_text

        print(f"  Card name '{card_name}' found:")
        print(f"    Raw: {'YES' if raw_found else 'NO'}")
        print(f"    Preprocessed: {'YES' if proc_found else 'NO'}")

        # Save debug images
        debug_dir = Path("tests/fixtures/ocr_debug")
        debug_dir.mkdir(parents=True, exist_ok=True)

        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")
        cv2.imwrite(str(debug_dir / f"{safe_name}_original.jpg"), image)
        cv2.imwrite(str(debug_dir / f"{safe_name}_preprocessed.jpg"), preprocessed)

    print("\n" + "="*60)
    print("Debug images saved to tests/fixtures/ocr_debug/")
    print("="*60)


if __name__ == "__main__":
    test_ocr_with_preprocessing()
