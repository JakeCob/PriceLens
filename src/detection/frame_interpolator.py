"""
Frame Interpolation System
Uses Kalman Filters to predict card positions between detection frames.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from filterpy.kalman import KalmanFilter

logger = logging.getLogger(__name__)

class FrameInterpolator:
    """
    Predicts card positions using Kalman Filters to enable smoother
    tracking and reduce the need for detection on every frame.
    """

    def __init__(self, dt: float = 1/30.0):
        """
        Initialize interpolator
        
        Args:
            dt: Time step between frames (default 30 FPS)
        """
        self.dt = dt
        self.filters: Dict[str, KalmanFilter] = {}
        self.last_update: Dict[str, int] = {}  # Frame number of last update
        
        logger.info(f"FrameInterpolator initialized (dt={dt:.3f})")

    def _create_filter(self) -> KalmanFilter:
        """Create a new Kalman Filter for a card"""
        kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State: [x, y, w, h, dx, dy, dw, dh]
        # Measurement: [x, y, w, h]
        
        # State Transition Matrix (F)
        kf.F = np.array([
            [1, 0, 0, 0, self.dt, 0, 0, 0],
            [0, 1, 0, 0, 0, self.dt, 0, 0],
            [0, 0, 1, 0, 0, 0, self.dt, 0],
            [0, 0, 0, 1, 0, 0, 0, self.dt],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement Function (H)
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        # Measurement Noise Covariance (R)
        kf.R *= 10
        
        # Process Noise Covariance (Q)
        kf.Q *= 0.1
        
        # Initial State Covariance (P)
        kf.P *= 1000
        
        return kf

    def update(self, card_id: str, bbox: List[int], frame_idx: int) -> None:
        """
        Update filter with new detection
        
        Args:
            card_id: Unique card identifier
            bbox: [x1, y1, x2, y2]
            frame_idx: Current frame number
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        
        # Center coordinates
        cx = x1 + w/2
        cy = y1 + h/2
        
        z = np.array([cx, cy, w, h])
        
        if card_id not in self.filters:
            kf = self._create_filter()
            kf.x[:4] = z.reshape((4, 1))
            self.filters[card_id] = kf
        else:
            self.filters[card_id].update(z)
            
        self.last_update[card_id] = frame_idx

    def predict(self, card_id: str) -> Optional[List[int]]:
        """
        Predict next position for a card
        
        Args:
            card_id: Card ID to predict
            
        Returns:
            Predicted bbox [x1, y1, x2, y2] or None if not found
        """
        if card_id not in self.filters:
            return None
            
        kf = self.filters[card_id]
        kf.predict()
        
        # Extract prediction
        cx, cy, w, h = kf.x[:4].flatten()
        
        # Convert back to [x1, y1, x2, y2]
        x1 = int(cx - w/2)
        y1 = int(cy - h/2)
        x2 = int(cx + w/2)
        y2 = int(cy + h/2)
        
        return [x1, y1, x2, y2]

    def cleanup(self, current_frame: int, max_age: int = 30) -> None:
        """
        Remove old filters
        
        Args:
            current_frame: Current frame number
            max_age: Maximum frames since last update
        """
        to_remove = []
        for card_id, last_frame in self.last_update.items():
            if current_frame - last_frame > max_age:
                to_remove.append(card_id)
                
        for card_id in to_remove:
            del self.filters[card_id]
            del self.last_update[card_id]
            logger.debug(f"Removed stale filter for {card_id}")

    def get_smoothed_bbox(self, card_id: str) -> Optional[List[int]]:
        """Get current smoothed bounding box from filter state"""
        if card_id not in self.filters:
            return None
        
        kf = self.filters[card_id]
        cx, cy, w, h = kf.x[:4].flatten()
        
        x1 = int(cx - w/2)
        y1 = int(cy - h/2)
        x2 = int(cx + w/2)
        y2 = int(cy + h/2)
        
        return [x1, y1, x2, y2]
