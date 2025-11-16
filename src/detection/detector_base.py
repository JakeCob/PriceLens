"""Base class for card detectors"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict


class DetectorBase(ABC):
    """Abstract base class for card detection implementations"""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect cards in a frame

        Args:
            frame: Input image (BGR format)

        Returns:
            List of detections with bbox, confidence, etc.
        """
        pass

    @abstractmethod
    def extract_card_regions(self, frame: np.ndarray,
                           detections: List[Dict]) -> List[np.ndarray]:
        """
        Extract card regions from frame based on detections

        Args:
            frame: Input image
            detections: List of detected cards

        Returns:
            List of cropped card images
        """
        pass