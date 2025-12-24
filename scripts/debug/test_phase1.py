#!/usr/bin/env python3
"""
Quick test script for YOLO detector.
Tests basic functionality without requiring webcam.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def test_imports():
    """Test that all required packages are installed"""
    print("Testing imports...")

    try:
        import cv2
        print(f"✓ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False

    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"✓ CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA Device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False

    try:
        from ultralytics import YOLO
        print("✓ Ultralytics YOLO: OK")
    except ImportError as e:
        print(f"✗ Ultralytics import failed: {e}")
        return False

    try:
        import yaml
        print("✓ PyYAML: OK")
    except ImportError as e:
        print(f"✗ PyYAML import failed: {e}")
        return False

    print("\n✅ All imports successful!\n")
    return True


def test_detector_init():
    """Test detector initialization"""
    print("Testing detector initialization...")

    try:
        from src.detection.yolo_detector import YOLOCardDetector, Detection

        # Check if model exists
        model_path = Path("models/yolo11n.pt")
        if not model_path.exists():
            print(
                f"⚠ Model not found at {model_path}\n"
                f"  Run: python scripts/download_models.py"
            )
            return False

        # Initialize detector
        detector = YOLOCardDetector(
            model_path=str(model_path), conf_threshold=0.5, quantize=False
        )

        print("✓ Detector initialized successfully")
        print(f"  Device: {detector.device}")
        print(f"  Confidence threshold: {detector.conf_threshold}")
        print(f"  IoU threshold: {detector.iou_threshold}")

        return True

    except Exception as e:
        print(f"✗ Detector initialization failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_detection_on_synthetic():
    """Test detection on a synthetic image"""
    print("\nTesting detection on synthetic image...")

    try:
        import cv2
        from src.detection.yolo_detector import YOLOCardDetector

        # Create synthetic image (black background)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Draw a white rectangle (simulated card)
        cv2.rectangle(frame, (300, 200), (500, 500), (255, 255, 255), -1)

        # Initialize detector
        detector = YOLOCardDetector(
            model_path="models/yolo11n.pt", conf_threshold=0.3
        )

        # Run detection
        detections = detector.detect(frame)

        print(f"✓ Detection completed")
        print(f"  Found {len(detections)} objects")

        for i, det in enumerate(detections):
            print(f"  Detection {i + 1}:")
            print(f"    BBox: {det.bbox}")
            print(f"    Confidence: {det.confidence:.3f}")
            print(f"    Aspect Ratio: {det.aspect_ratio:.3f}")

        return True

    except FileNotFoundError:
        print("⚠ Model not found - skipping detection test")
        return True  # Not a failure, just not ready
    except Exception as e:
        print(f"✗ Detection test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_camera_access():
    """Test if camera is accessible"""
    print("\nTesting camera access...")

    try:
        import cv2

        cap = cv2.VideoCapture(0)
        is_opened = cap.isOpened()

        if is_opened:
            print("✓ Camera is accessible")
            ret, frame = cap.read()
            if ret:
                print(f"✓ Frame capture successful: {frame.shape}")
            else:
                print("⚠ Could not read frame from camera")
        else:
            print("⚠ Camera not accessible (WSL or no camera)")
            print("  This is OK - you can use video files for testing")

        cap.release()
        return True

    except Exception as e:
        print(f"⚠ Camera test failed: {e}")
        return True  # Not critical


def main():
    """Run all tests"""
    print("=" * 60)
    print("PriceLens - Quick Test Suite")
    print("=" * 60)
    print()

    tests = [
        ("Package Imports", test_imports),
        ("Detector Initialization", test_detector_init),
        ("Synthetic Detection", test_detection_on_synthetic),
        ("Camera Access", test_camera_access),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results[test_name] = False
        print()

    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    print()
    print(f"Result: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Ready to proceed.")
        print("\nNext steps:")
        print("1. python scripts/download_models.py  # Download YOLO model")
        print("2. python src/detection/yolo_detector.py  # Test with webcam")
    else:
        print("\n⚠ Some tests failed. Please fix issues before proceeding.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
