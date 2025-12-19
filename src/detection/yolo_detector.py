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

# Import enhancer for type hints (lazy import at runtime)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.preprocessing.enhancer import ImageEnhancer

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
        use_card_specific_model: bool = False,
        enhancer: Optional["ImageEnhancer"] = None,
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
            use_card_specific_model: Set to True if using Pokemon card-specific model
                                      (e.g., downloaded from Roboflow). This disables
                                      COCO class filtering and trusts the model output.
            enhancer: Optional ImageEnhancer instance for pre-detection image enhancement.
                      If provided, frames will be preprocessed before YOLO inference.
        """
        # Store enhancer for pre-detection enhancement
        self.enhancer = enhancer

        if YOLO is None:
            raise ImportError(
                "Ultralytics YOLO not installed. "
                "Install with: pip install ultralytics"
            )

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"For COCO model: Run 'python scripts/download_models.py'\n"
                f"For Pokemon card model: Run 'python scripts/download_card_model.py'"
            )

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.min_aspect_ratio = 0.4  # Widened from 0.6
        self.max_aspect_ratio = 1.8  # Widened from 0.85 to allow landscape/rotatedtio

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
        self.hit_streaks = {} # Track consecutive detections (hits)
        self.miss_counts = {} # Track consecutive misses
        self.confirmed_cards = set() # Cards that passed 3-hit confirmation
        self.interpolator = None

        # Kalman smoothing temporarily disabled until tracking is stable
        self.use_kalman_smoothing = False

        # Configure class filtering based on model type
        self.use_card_specific_model = use_card_specific_model

        if use_card_specific_model:
            # Card-specific model (e.g., from Roboflow) - trust all detections
            self.blocked_class_ids = set()
            logger.info("Using card-specific model - no class filtering applied")
        else:
            # COCO-pretrained model
            # Expanded class list to improve card detection
            # COCO Classes that might detect cards:
            # 24: backpack (card sleeves), 25: umbrella (rectangular)
            # 26: handbag, 27: tie, 28: suitcase (rectangular)
            # 39: bottle, 62: tv, 63: laptop
            # 65: remote, 67: cell phone, 73: book
            # 74: clock, 75: vase, 76: scissors
            
            # Broader allowlist - trust aspect ratio filtering to reject non-cards
            self.allowed_class_ids = {24, 25, 26, 27, 28, 39, 62, 63, 65, 67, 73, 74, 75, 76}
            self.blocked_class_ids = {0}  # Block person only
            logger.info(f"COCO MAPPING: Accepting classes {self.allowed_class_ids}")
            
            # Tighten aspect ratio for standard model (Pokemon cards are ~0.71)
            self.min_aspect_ratio = 0.5
            self.max_aspect_ratio = 1.5

        try:
            from .frame_interpolator import FrameInterpolator
            self.interpolator = FrameInterpolator()
            if self.use_kalman_smoothing:
                logger.info("Frame Interpolator initialized")
            else:
                logger.info("Frame Interpolator initialized but disabled (use_kalman_smoothing=False)")
        except ImportError:
            logger.warning("FrameInterpolator not available")
            
        # Initialize Face Detector (Haar Cascade) to filter false positives
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            logger.info("Face detector initialized for filtering")
        except Exception as e:
            logger.warning(f"Failed to initialize face detector: {e}")
            self.face_cascade = None

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

        # Apply pre-detection enhancement if enhancer is available
        if self.enhancer is not None:
            try:
                frame = self.enhancer.enhance_for_detection(frame)
            except Exception as e:
                logger.warning(f"Pre-detection enhancement failed: {e}")

        logger.info(f"Detector received frame: {frame.shape}")  # Force INFO log
        scale = 1.0  # No resizing, so scale is 1.0

        # Run face detection on the whole frame once to filter false positives
        # Only needed for COCO-pretrained models
        faces = []
        if self.face_cascade and not self.use_card_specific_model:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(60, 60)
                )
            except Exception as e:
                logger.warning(f"Face detection failed: {e}")

        # Run YOLO inference
        try:
            # Run inference
            # verbose=True enables YOLO's own internal logging to stdout
            results = self.model.predict(
                frame,
                imgsz=320,  # Force 320px inference for speed on CPU
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
                stream=False
            )
        except Exception as e:
            logger.error(f"YOLO inference failed: {e}")
            return []

        # Log raw YOLO output BEFORE any filtering
        logger.warning(f"=== YOLO RAW RESULTS ===")
        logger.warning(f"Number of result objects: {len(results)}")

        # Parse results
        detections = []
        filtered_count = {"class": 0, "aspect": 0, "blocked": 0, "size": 0, "face": 0}

        frame_h, frame_w = frame.shape[:2]
        frame_area = max(1, frame_h * frame_w)

        for result in results:
            logger.warning(f"Result boxes: {len(result.boxes)}")  # Log raw box count
            boxes = result.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]

                logger.warning(f"  Box: class={class_name} (id={class_id}), conf={confidence:.3f}")

                # Skip blocked classes (person, face, etc.)
                if self.blocked_class_ids and class_id in self.blocked_class_ids:
                    logger.warning(f"  FILTERED (blocked): {class_name} (id={class_id})")
                    filtered_count["blocked"] += 1
                    continue
                    
                # Filter by allowed classes (if set)
                if hasattr(self, 'allowed_class_ids') and self.allowed_class_ids is not None:
                    if class_id not in self.allowed_class_ids:
                        logger.warning(f"  FILTERED (class): {class_name} (id={class_id}) not in allowed")
                        filtered_count["class"] += 1
                        continue

                # Validate detection is card-like (rectangular)
                width = x2 - x1
                height = y2 - y1

                # Filter by absolute/relative size (mainly for COCO-pretrained models).
                # COCO often returns large "rectangles" over people/background objects that pass aspect ratio.
                # Real cards are usually much smaller than the full frame in this scanner view.
                bbox_area = max(1.0, width * height)
                area_frac = bbox_area / float(frame_area)

                if not self.use_card_specific_model:
                    # Reject absurdly large boxes (commonly "stuck" around a person/background).
                    if area_frac > 0.40 or width > frame_w * 0.95 or height > frame_h * 0.95:
                        logger.warning(f"  FILTERED (size): area={area_frac:.2f} w={width:.0f} h={height:.0f}")
                        filtered_count["size"] += 1
                        continue
                    # Reject tiny noise
                    if area_frac < 0.002:
                        filtered_count["size"] += 1
                        continue
                
                # Calculate aspect ratio
                aspect_ratio = width / height if height > 0 else 0
                
                # Check aspect ratio
                aspect_valid = self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio
                
                # For card-specific model, be more lenient or trust the model
                if self.use_card_specific_model:
                    # Trust the model more, but still sanity check
                    aspect_valid = self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio

                if not aspect_valid:
                    valid_range = "0.4-1.8" if self.use_card_specific_model else f"{self.min_aspect_ratio}-{self.max_aspect_ratio}"
                    logger.warning(f"  FILTERED (aspect): {aspect_ratio:.2f} (valid: {valid_range})")
                    filtered_count["aspect"] += 1
                    continue

                if aspect_valid:
                    # Face-based rejection for COCO model:
                    # If a detection contains a face center and is relatively large, it's likely the person/background.
                    if faces and not self.use_card_specific_model:
                        is_face_related = False
                        for (fx, fy, fw, fh) in faces:
                            fx, fy, fw, fh = fx / scale, fy / scale, fw / scale, fh / scale
                            fcx = fx + (fw / 2.0)
                            fcy = fy + (fh / 2.0)
                            if (x1 <= fcx <= x2) and (y1 <= fcy <= y2):
                                # Only reject if the box is big enough that it’s probably "the person",
                                # not a small card near a face.
                                if area_frac > 0.08:
                                    is_face_related = True
                                    break

                        if is_face_related:
                            logger.warning(f"  FILTERED (face): contains face center, area={area_frac:.2f}")
                            filtered_count["face"] += 1
                            continue

                    logger.warning(f"  PASSED: {class_name}, aspect={aspect_ratio:.3f}")
                    
                    # Create detection object
                    detection = Detection(
                        bbox=[int(x1), int(y1), int(x2), int(y2)],
                        confidence=confidence,
                        class_id=class_id,
                        class_name=class_name,
                        aspect_ratio=aspect_ratio
                    )
                    detections.append(detection)

        logger.warning(f"=== DETECT COMPLETE: {len(detections)} detections, filtered: {filtered_count} ===")
        return detections

    def _is_face_overlap(self, card_bbox, faces) -> bool:
        """Check if card bbox overlaps with any detected face"""
        cx1, cy1, cx2, cy2 = card_bbox
        card_area = max(1, (cx2 - cx1) * (cy2 - cy1))
        
        for (fx, fy, fw, fh) in faces:
            fx2, fy2 = fx + fw, fy + fh
            
            # Calculate intersection
            ix1 = max(cx1, fx)
            iy1 = max(cy1, fy)
            ix2 = min(cx2, fx2)
            iy2 = min(cy2, fy2)
            
            if ix2 > ix1 and iy2 > iy1:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                # If intersection covers significant portion of the card (30%)
                if intersection / card_area > 0.3:
                    return True
        return False

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
                # Initialize streak bookkeeping so confirmation thresholds work
                self.hit_streaks[det.card_id] = 1
                self.miss_counts[det.card_id] = 0

                # Initialize interpolator (if smoothing enabled)
                if self.use_kalman_smoothing and self.interpolator:
                    self.interpolator.update(det.card_id, det.bbox, frame_idx)

                self.next_card_id += 1
            # Return only confirmed cards (usually none on first frame)
            return [d for d in detections if d.card_id in self.confirmed_cards]

        # Match current detections to tracked cards using IoU
        matched_detections = []
        unmatched_detections = []
        
        # Keep track of which tracked cards were matched
        matched_card_ids = set()

        for det in detections:
            best_match_id = None
            best_iou = 0.5  # Minimum IoU threshold (raised from 0.3 to reduce false positives)

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

                # Increment hit streak and reset miss count
                self.hit_streaks[best_match_id] = self.hit_streaks.get(best_match_id, 0) + 1
                self.miss_counts[best_match_id] = 0  # Reset miss count on successful match

                # Mark as confirmed - instant for card-specific models, 3 hits for generic models
                use_card_specific = bool(getattr(self, "use_card_specific_model", False))
                required_hits = 1 if use_card_specific else 3
                if self.hit_streaks[best_match_id] >= required_hits:
                    self.confirmed_cards.add(best_match_id)

                # Update interpolator (if smoothing enabled)
                if self.use_kalman_smoothing and self.interpolator:
                    self.interpolator.update(best_match_id, det.bbox, frame_idx)
                    # Use smoothed bbox
                    smoothed_bbox = self.interpolator.get_smoothed_bbox(best_match_id)
                    if smoothed_bbox:
                        det.bbox = smoothed_bbox
            else:
                # New card detected
                unmatched_detections.append(det)

        # Handle unmatched tracked cards (increment miss counts)
        cards_to_remove = []
        for card_id in self.tracked_cards.keys():
            if card_id not in matched_card_ids:
                # Increment miss count and reset hit streak
                self.miss_counts[card_id] = self.miss_counts.get(card_id, 0) + 1
                self.hit_streaks[card_id] = 0  # Reset hit streak on miss

                # Remove if missed too many times (>= 2 frames)
                if self.miss_counts[card_id] >= 2:
                    cards_to_remove.append(card_id)
                    if card_id in self.confirmed_cards:
                        self.confirmed_cards.remove(card_id)
                    logger.debug(f"Removing card {card_id} after {self.miss_counts[card_id]} misses")

        # Remove cards that have been missed too many times
        for card_id in cards_to_remove:
            del self.tracked_cards[card_id]
            if card_id in self.hit_streaks:
                del self.hit_streaks[card_id]
            if card_id in self.miss_counts:
                del self.miss_counts[card_id]
            if self.use_kalman_smoothing and self.interpolator and card_id in self.interpolator.filters:
                # Remove from interpolator as well
                del self.interpolator.filters[card_id]
                if card_id in self.interpolator.last_update:
                    del self.interpolator.last_update[card_id]

        # Assign new IDs to unmatched detections
        for det in unmatched_detections:
            det.card_id = f"card_{self.next_card_id}"
            self.tracked_cards[det.card_id] = det
            self.hit_streaks[det.card_id] = 1  # Initialize streak with 1 hit
            self.miss_counts[det.card_id] = 0  # Initialize miss count

            # Initialize interpolator for new card (if smoothing enabled)
            if self.use_kalman_smoothing and self.interpolator:
                self.interpolator.update(det.card_id, det.bbox, frame_idx)

            self.next_card_id += 1
            matched_detections.append(det)

        # Clean up old filters and streaks (if smoothing enabled)
        if self.use_kalman_smoothing and self.interpolator:
            self.interpolator.cleanup(frame_idx, max_age=30)

            # Also cleanup tracked_cards and streaks
            to_remove = []
            for card_id in self.tracked_cards:
                if card_id not in self.interpolator.filters:
                    to_remove.append(card_id)
            for card_id in to_remove:
                del self.tracked_cards[card_id]
                if card_id in self.hit_streaks:
                    del self.hit_streaks[card_id]
                    
        # Filter output based on hit streak (Temporal Consistency)
        # Only show confirmed cards (3+ consecutive hits)
        final_detections = []
        for det in matched_detections:
            if det.card_id in self.confirmed_cards:
                final_detections.append(det)

        return final_detections

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
