"""
Image Enhancer - Orchestrates preprocessing pipeline for card detection and identification.

Provides configurable preprocessing pipelines with three modes:
- speed: Minimal preprocessing for low-end hardware (~1.5ms overhead)
- balanced: Good quality/speed trade-off (~3.5ms overhead)
- quality: Maximum enhancement for difficult conditions (~7ms overhead)
"""

import cv2
import numpy as np
import logging
import time
from typing import Dict, List, Optional, Callable

from src.preprocessing.techniques import (
    normalize_lighting,
    apply_fast_clahe,
    apply_enhanced_clahe,
    white_balance_gray_world,
    reduce_glare,
    remove_shadows,
    histogram_stretch,
)

logger = logging.getLogger(__name__)


# Technique registry - maps technique names to functions
TECHNIQUE_REGISTRY: Dict[str, Callable] = {
    "lighting_normalization": normalize_lighting,
    "clahe_fast": apply_fast_clahe,
    "clahe_enhanced": apply_enhanced_clahe,
    "white_balance": white_balance_gray_world,
    "glare_reduction": reduce_glare,
    "shadow_removal": remove_shadows,
    "histogram_stretch": histogram_stretch,
}

# Pre-defined profiles
PROFILES = {
    "speed": {
        "detection": ["lighting_normalization"],
        "identification": ["clahe_fast"],
    },
    "balanced": {
        "detection": ["lighting_normalization", "clahe_fast"],
        "identification": ["white_balance", "clahe_enhanced"],
    },
    "quality": {
        "detection": ["lighting_normalization", "clahe_fast", "white_balance"],
        "identification": ["white_balance", "clahe_enhanced", "glare_reduction", "shadow_removal"],
    },
}


class ImageEnhancer:
    """
    Configurable image preprocessing pipeline for card detection and identification.

    Supports two separate pipelines:
    - Detection pipeline: Applied to full frames before YOLO inference
    - Identification pipeline: Applied to card crops before feature extraction/OCR

    Example usage:
        config = {
            "enabled": True,
            "mode": "balanced",
            "detection": {"enabled": True},
            "identification": {"enabled": True},
        }
        enhancer = ImageEnhancer(config)

        # Pre-detection enhancement
        enhanced_frame = enhancer.enhance_for_detection(frame)

        # Identification enhancement (on card crop)
        enhanced_crop = enhancer.enhance_for_identification(card_crop)
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize ImageEnhancer with configuration.

        Args:
            config: Configuration dict with structure:
                {
                    "enabled": bool,
                    "mode": "speed" | "balanced" | "quality",
                    "detection": {"enabled": bool, "techniques": [...]},
                    "identification": {"enabled": bool, "techniques": [...]},
                    "clahe": {"clip_limit": float, "tile_grid_size": [int, int]},
                    "glare": {"threshold": int, "blur_kernel": int},
                }
        """
        config = config or {}

        self.enabled = config.get("enabled", True)
        self.mode = config.get("mode", "balanced")

        # Get profile-based defaults
        profile = PROFILES.get(self.mode, PROFILES["balanced"])

        # Detection pipeline config
        detection_config = config.get("detection", {})
        self.detection_enabled = detection_config.get("enabled", True) and self.enabled
        self.detection_techniques = detection_config.get("techniques", profile["detection"])

        # Identification pipeline config
        identification_config = config.get("identification", {})
        self.identification_enabled = identification_config.get("enabled", True) and self.enabled
        self.identification_techniques = identification_config.get("techniques", profile["identification"])

        # Technique-specific parameters
        clahe_config = config.get("clahe", {})
        self.clahe_clip_limit = clahe_config.get("clip_limit", 2.0)
        self.clahe_tile_size = tuple(clahe_config.get("tile_grid_size", [8, 8]))

        glare_config = config.get("glare", {})
        self.glare_threshold = glare_config.get("threshold", 240)
        self.glare_blur_kernel = glare_config.get("blur_kernel", 5)

        # Performance tracking
        self._timing_enabled = config.get("timing_enabled", False)
        self._detection_times: List[float] = []
        self._identification_times: List[float] = []

        logger.info(
            f"ImageEnhancer initialized: mode={self.mode}, "
            f"detection={self.detection_enabled}, "
            f"identification={self.identification_enabled}"
        )
        if self.detection_enabled:
            logger.info(f"  Detection techniques: {self.detection_techniques}")
        if self.identification_enabled:
            logger.info(f"  Identification techniques: {self.identification_techniques}")

    def enhance_for_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Apply detection preprocessing pipeline to a full frame.

        This pipeline is optimized for speed since it runs on every frame.
        Techniques are applied in order to prepare the frame for YOLO detection.

        Args:
            image: Input BGR frame (typically 720p or 1080p)

        Returns:
            Enhanced BGR frame ready for YOLO detection
        """
        if not self.detection_enabled or image is None:
            return image

        start_time = time.perf_counter() if self._timing_enabled else None

        result = image
        for technique_name in self.detection_techniques:
            result = self._apply_technique(result, technique_name)

        if self._timing_enabled and start_time:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._detection_times.append(elapsed_ms)
            if len(self._detection_times) % 100 == 0:
                avg = sum(self._detection_times[-100:]) / 100
                logger.debug(f"Detection preprocessing avg: {avg:.2f}ms")

        return result

    def enhance_for_identification(self, image: np.ndarray) -> np.ndarray:
        """
        Apply identification preprocessing pipeline to a card crop.

        This pipeline can be more aggressive since it only runs on detected card regions.
        Includes techniques for handling glare, shadows, and color correction.

        Args:
            image: Input BGR card crop (typically 300-500px)

        Returns:
            Enhanced BGR card image ready for feature extraction/OCR
        """
        if not self.identification_enabled or image is None:
            return image

        start_time = time.perf_counter() if self._timing_enabled else None

        result = image
        for technique_name in self.identification_techniques:
            result = self._apply_technique(result, technique_name)

        if self._timing_enabled and start_time:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._identification_times.append(elapsed_ms)
            if len(self._identification_times) % 100 == 0:
                avg = sum(self._identification_times[-100:]) / 100
                logger.debug(f"Identification preprocessing avg: {avg:.2f}ms")

        return result

    def enhance_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Apply OCR-optimized preprocessing pipeline.

        Uses all identification techniques plus histogram stretch for maximum
        text clarity. Slightly slower but produces best OCR results.

        Args:
            image: Input BGR card crop

        Returns:
            Enhanced BGR image optimized for OCR
        """
        if not self.identification_enabled or image is None:
            return image

        # Apply standard identification pipeline
        result = self.enhance_for_identification(image)

        # Add histogram stretch for better OCR contrast
        result = histogram_stretch(result)

        return result

    def _apply_technique(self, image: np.ndarray, technique_name: str) -> np.ndarray:
        """
        Apply a single preprocessing technique with appropriate parameters.

        Args:
            image: Input image
            technique_name: Name of technique from TECHNIQUE_REGISTRY

        Returns:
            Processed image
        """
        if technique_name not in TECHNIQUE_REGISTRY:
            logger.warning(f"Unknown preprocessing technique: {technique_name}")
            return image

        technique_func = TECHNIQUE_REGISTRY[technique_name]

        try:
            # Apply technique with custom parameters where applicable
            if technique_name == "clahe_fast":
                return apply_fast_clahe(image, clip_limit=self.clahe_clip_limit, tile_size=(4, 4))
            elif technique_name == "clahe_enhanced":
                return apply_enhanced_clahe(image, clip_limit=self.clahe_clip_limit, tile_size=self.clahe_tile_size)
            elif technique_name == "glare_reduction":
                return reduce_glare(image, threshold=self.glare_threshold, blur_kernel=self.glare_blur_kernel)
            else:
                return technique_func(image)
        except Exception as e:
            logger.error(f"Error applying {technique_name}: {e}")
            return image

    def get_timing_stats(self) -> Dict:
        """
        Get performance statistics for preprocessing.

        Returns:
            Dict with timing statistics
        """
        stats = {
            "detection": {
                "enabled": self.detection_enabled,
                "techniques": self.detection_techniques,
            },
            "identification": {
                "enabled": self.identification_enabled,
                "techniques": self.identification_techniques,
            },
        }

        if self._detection_times:
            stats["detection"]["avg_ms"] = sum(self._detection_times) / len(self._detection_times)
            stats["detection"]["samples"] = len(self._detection_times)

        if self._identification_times:
            stats["identification"]["avg_ms"] = sum(self._identification_times) / len(self._identification_times)
            stats["identification"]["samples"] = len(self._identification_times)

        return stats

    def set_mode(self, mode: str) -> None:
        """
        Change preprocessing mode at runtime.

        Args:
            mode: One of "speed", "balanced", "quality"
        """
        if mode not in PROFILES:
            logger.warning(f"Unknown mode '{mode}', keeping current settings")
            return

        profile = PROFILES[mode]
        self.mode = mode
        self.detection_techniques = profile["detection"]
        self.identification_techniques = profile["identification"]
        logger.info(f"Preprocessing mode changed to: {mode}")

    def disable(self) -> None:
        """Disable all preprocessing."""
        self.enabled = False
        self.detection_enabled = False
        self.identification_enabled = False
        logger.info("Preprocessing disabled")

    def enable(self) -> None:
        """Re-enable preprocessing with current settings."""
        self.enabled = True
        self.detection_enabled = True
        self.identification_enabled = True
        logger.info("Preprocessing enabled")
