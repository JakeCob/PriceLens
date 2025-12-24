#!/usr/bin/env python3
"""
Test YOLO detector on sample Pokemon card images
Shows detection results visually
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from src.detection.yolo_detector import YOLOCardDetector


def test_on_images():
    """Test detector on sample images"""
    print("=" * 60)
    print("Testing YOLO Detector on Sample Images")
    print("=" * 60)
    print()

    # Initialize detector
    print("Initializing detector...")
    detector = YOLOCardDetector(
        model_path="models/yolo11n.pt", conf_threshold=0.3, quantize=False
    )
    print(f"✓ Detector ready (device: {detector.device})")
    print()

    # Get test images
    test_dir = Path("test_images")
    if not test_dir.exists():
        print(f"❌ Test images directory not found: {test_dir}")
        print("Run: python scripts/download_test_images.py")
        return False

    image_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))

    if not image_files:
        print(f"❌ No images found in {test_dir}")
        print("Run: python scripts/download_test_images.py")
        return False

    print(f"Found {len(image_files)} test images\n")

    # Create output directory
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)

    # Test on each image
    results = []
    for img_path in image_files:
        print(f"Testing: {img_path.name}")

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  ⚠️  Could not load image")
            continue

        # Detect
        detections = detector.detect(image)
        print(f"  Found: {len(detections)} detections")

        # Draw results
        result_image = image.copy()
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox

            # Color based on confidence
            if det.confidence > 0.7:
                color = (0, 255, 0)  # Green - high confidence
            elif det.confidence > 0.5:
                color = (0, 165, 255)  # Orange - medium
            else:
                color = (0, 0, 255)  # Red - low

            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 3)

            # Label
            label = f"Card {i + 1}: {det.confidence:.2f}"
            label_size, _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )

            # Background for label
            cv2.rectangle(
                result_image,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1,
            )

            # Text
            cv2.putText(
                result_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            print(
                f"    Detection {i + 1}: conf={det.confidence:.3f}, "
                f"aspect={det.aspect_ratio:.3f}"
            )

        # Save result
        output_path = output_dir / f"result_{img_path.stem}.jpg"
        cv2.imwrite(str(output_path), result_image)
        print(f"  ✓ Saved: {output_path}")

        results.append(
            {"image": img_path.name, "detections": len(detections), "saved": output_path}
        )
        print()

    # Summary
    print("=" * 60)
    print("Detection Summary")
    print("=" * 60)

    total_detections = sum(r["detections"] for r in results)
    avg_detections = total_detections / len(results) if results else 0

    print(f"Images tested: {len(results)}")
    print(f"Total detections: {total_detections}")
    print(f"Average per image: {avg_detections:.1f}")
    print()
    print(f"Results saved to: {output_dir}/")
    print()
    print("Next steps:")
    print("1. Check the result images in test_results/")
    print("2. If detection looks good, proceed to Phase 2")
    print("3. If issues, adjust conf_threshold or min/max aspect ratios")

    return True


if __name__ == "__main__":
    try:
        success = test_on_images()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
