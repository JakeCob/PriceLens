#!/usr/bin/env python3
"""
Enhanced YOLO Card Detector with tracking and optimization support.
Implements Phase 1 detection with modern features.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # Will be handled gracefully

from src.detection.detector_base import DetectorBase

logger = logging.getLogger(__name__)


class Detection:
    """Data class for card detections"""

    def __init__(
        self,
        bbox: List[int],
        confidence: float,
        class_id: int,
        class_name: str,
        aspect_ratio: float,
        card_id: Optional[str] = None,
    ):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.aspect_ratio = aspect_ratio
        self.card_id = card_id or f"card_{id(self)}"

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "bbox": self.bbox,
            "confidence": float(self.confidence),
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "aspect_ratio": float(self.aspect_ratio),
            "card_id": self.card_id,
        }


class YOLOCardDetector(DetectorBase):
    """
    Enhanced YOLO-based card detector with tracking and optimization.

    Features:
    - Card detection with aspect ratio filtering
    - Model quantization support
    - Card tracking across frames (Phase 1.3)
    - GPU/CPU fallback
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "auto",
        quantize: bool = False,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 0.9,
    ):
        """
        Initialize YOLO card detector

        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold for detections (0-1)
            iou_threshold: IoU threshold for NMS
            device: Device to run inference ('cpu', 'cuda', 'auto')
            quantize: Use INT8 quantization for faster inference
            min_aspect_ratio: Minimum aspect ratio for card filtering
            max_aspect_ratio: Maximum aspect ratio for card filtering
        """
        if YOLO is None:
            raise ImportError(
                "Ultralytics YOLO not installed. "
                "Install with: pip install ultralytics"
            )

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Run 'python scripts/download_models.py' to download"
            )

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio

        # Determine device
        self.device = self._get_device(device)

        # Load model
        logger.info(f"Loading YOLO model from {model_path}")
        self.model = YOLO(str(model_path))

        # Apply quantization if requested
        if quantize and self.device != "cpu":
            logger.info("Applying INT8 quantization for faster inference")
            self._apply_quantization()

        # Card tracking (Phase 1.3)
        self.tracked_cards: Dict[str, Detection] = {}
        self.next_card_id = 0
        
        # Initialize Frame Interpolator
        try:
            from src.detection.frame_interpolator import FrameInterpolator
            self.interpolator = FrameInterpolator()
            logger.info("Frame Interpolator initialized")
        except ImportError:
            self.interpolator = None
            logger.warning("FrameInterpolator not available (filterpy missing?)")

        logger.info(
            f"YOLO detector initialized on {self.device} "
            f"(conf={conf_threshold}, iou={iou_threshold})"
        )

    def _get_device(self, device: str) -> str:
        """Determine device for inference"""
        if device == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    logger.info("CUDA available - using GPU")
                    return "cuda"
                else:
                    logger.info("CUDA not available - using CPU")
                    return "cpu"
            except ImportError:
                logger.warning("PyTorch not available - using CPU")
                return "cpu"
        return device

    def _apply_quantization(self) -> None:
        """Apply INT8 quantization for 4x faster inference"""
        try:
            # Export to quantized format
            quantized_path = self.model_path.parent / f"{self.model_path.stem}_int8.pt"
            if not quantized_path.exists():
                logger.info("Exporting quantized model...")
                self.model.export(format="onnx", int8=True)
                logger.info(f"Quantized model saved to {quantized_path}")
        except Exception as e:
            logger.warning(f"Quantization failed: {e}. Using standard model.")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect cards in a frame

        Args:
            frame: Input image (BGR format from OpenCV)

        Returns:
            List of Detection objects
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame received")
            return []

        # Run YOLO inference
        try:
            results = self.model.predict(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
                stream=False,
            )
        except Exception as e:
            logger.error(f"YOLO inference failed: {e}")
            return []

        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes

            for box in boxes:
                # Extract box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]

                # Validate detection is card-like (rectangular)
                width = x2 - x1
                height = y2 - y1
                aspect_ratio = width / height if height > 0 else 0

                # Pokemon cards are approximately 2.5" x 3.5" (aspect ~0.71)
                if self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio:
                    detection = Detection(
                        bbox=[int(x1), int(y1), int(x2), int(y2)],
                        confidence=confidence,
                        class_id=class_id,
                        class_name=class_name,
                        aspect_ratio=aspect_ratio,
                    )
                    detections.append(detection)

        logger.debug(f"Detected {len(detections)} cards")
        return detections

    def extract_card_regions(
        self, frame: np.ndarray, detections: List[Detection]
    ) -> List[np.ndarray]:
        """
        Extract card regions from frame

        Args:
            frame: Input image
            detections: List of detected cards

        Returns:
            List of cropped card images
        """
        card_regions = []

        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # Add small padding (5 pixels)
            padding = 5
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)

            # Extract region
            card_region = frame[y1:y2, x1:x2].copy()

            if card_region.size > 0:
                card_regions.append(card_region)
            else:
                logger.warning(f"Empty card region for detection {det.card_id}")

        return card_regions

    def track_cards(self, detections: List[Detection], frame_idx: int = 0) -> List[Detection]:
        """
        Track cards across frames (simple IoU-based tracking)
        
        Args:
            detections: Current frame detections
            frame_idx: Current frame number (for interpolation)

        Returns:
            Detections with consistent card_ids
        """
        if not self.tracked_cards:
            # First frame - assign new IDs
            for det in detections:
                det.card_id = f"card_{self.next_card_id}"
                self.tracked_cards[det.card_id] = det
                
                # Initialize interpolator
                if self.interpolator:
                    self.interpolator.update(det.card_id, det.bbox, frame_idx)
                
                self.next_card_id += 1
            return detections

        # Match current detections to tracked cards using IoU
        matched_detections = []
        unmatched_detections = []
        
        # Keep track of which tracked cards were matched
        matched_card_ids = set()

        for det in detections:
            best_match_id = None
            best_iou = 0.3  # Minimum IoU threshold

            # Find best matching tracked card
            for card_id, tracked_det in self.tracked_cards.items():
                iou = self._calculate_iou(det.bbox, tracked_det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match_id = card_id

            if best_match_id:
                # Update existing card
                det.card_id = best_match_id
                self.tracked_cards[best_match_id] = det
                matched_detections.append(det)
                matched_card_ids.add(best_match_id)
                
                # Update interpolator
                if self.interpolator:
                    self.interpolator.update(best_match_id, det.bbox, frame_idx)
                    # Use smoothed bbox
                    smoothed_bbox = self.interpolator.get_smoothed_bbox(best_match_id)
                    if smoothed_bbox:
                        det.bbox = smoothed_bbox
            else:
                # New card detected
                unmatched_detections.append(det)

        # Handle lost cards (coasting)
        if self.interpolator:
            for card_id, tracked_det in self.tracked_cards.items():
                if card_id not in matched_card_ids:
                    # Predict next position
                    pred_bbox = self.interpolator.predict(card_id)
                    
                    # Check if we should keep it (e.g. not too old)
                    last_update = self.interpolator.last_update.get(card_id, 0)
                    if pred_bbox and (frame_idx - last_update < 10): # Coast for 10 frames
                        # Create a predicted detection
                        new_det = Detection(
                            bbox=pred_bbox,
                            confidence=tracked_det.confidence * 0.95, # Decay confidence
                            class_id=tracked_det.class_id,
                            class_name=tracked_det.class_name,
                            aspect_ratio=tracked_det.aspect_ratio,
                            card_id=card_id
                        )
                        matched_detections.append(new_det)
                        # Note: We don't update self.tracked_cards with prediction to avoid drift
                        # But we return it so the overlay sees it

        # Assign new IDs to unmatched detections
        for det in unmatched_detections:
            det.card_id = f"card_{self.next_card_id}"
            self.tracked_cards[det.card_id] = det
            
            # Initialize interpolator for new card
            if self.interpolator:
                self.interpolator.update(det.card_id, det.bbox, frame_idx)
                
            self.next_card_id += 1
            matched_detections.append(det)

        # Clean up old filters
        if self.interpolator:
            self.interpolator.cleanup(frame_idx, max_age=30)
            
            # Also cleanup tracked_cards
            to_remove = []
            for card_id in self.tracked_cards:
                if card_id not in self.interpolator.filters:
                    to_remove.append(card_id)
            for card_id in to_remove:
                del self.tracked_cards[card_id]

        return matched_detections

    @staticmethod
    def _calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def export_optimized_model(
        self, format: str = "onnx", output_path: Optional[str] = None
    ) -> str:
        """
        Export model to optimized format

        Args:
            format: Export format ('onnx', 'openvino', 'engine')
            output_path: Optional output path

        Returns:
            Path to exported model
        """
        if output_path is None:
            output_path = self.model_path.parent / f"{self.model_path.stem}.{format}"

        logger.info(f"Exporting model to {format} format...")
        export_path = self.model.export(format=format)
        logger.info(f"Model exported to {export_path}")

        return str(export_path)


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test detector initialization
    try:
        detector = YOLOCardDetector(
            model_path="models/yolo11n.pt", conf_threshold=0.5, quantize=False
        )

        # Test with webcam (if available)
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            logger.warning("No camera available. Exiting.")
            exit(0)

        logger.info("Press 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect cards
            detections = detector.detect(frame)
            tracked_detections = detector.track_cards(detections)

            # Draw bounding boxes
            for det in tracked_detections:
                x1, y1, x2, y2 = det.bbox

                # Color based on confidence
                color = (0, 255, 0) if det.confidence > 0.7 else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Label
                label = f"{det.card_id} ({det.confidence:.2f})"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            # Show FPS
            cv2.putText(
                frame,
                f"Cards: {len(tracked_detections)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Card Detection Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        logger.info("Run: python scripts/download_models.py")
    except Exception as e:
        logger.error(f"Error: {e}")
