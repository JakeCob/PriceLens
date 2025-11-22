"""
Overlay Renderer
Handles drawing HUD and card information on images.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple

from src.api.base import PriceData


class OverlayRenderer:
    """Renders card information and prices on images"""

    def __init__(self, font_scale: float = 1.0, thickness: int = 2):
        """
        Initialize renderer
        
        Args:
            font_scale: Base font scale
            thickness: Text thickness
        """
        self.font_scale = font_scale
        self.thickness = thickness
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Colors (BGR)
        self.colors = {
            "text": (255, 255, 255),      # White
            "shadow": (0, 0, 0),          # Black
            "panel": (0, 0, 0),           # Black
            "accent": (0, 255, 255),      # Yellow
            "price": (0, 255, 0),         # Green
            "border": (255, 255, 255),    # White
        }

    def draw_overlay(
        self, 
        image: np.ndarray, 
        card_info: Dict, 
        price_data: Optional[PriceData] = None
    ) -> np.ndarray:
        """
        Draw info panel on image
        
        Args:
            image: Input image
            card_info: Card metadata from identification
            price_data: Price information (optional)
            
        Returns:
            Image with overlay
        """
        output = image.copy()
        h, w = output.shape[:2]
        
        # Panel configuration
        panel_height = int(h * 0.15)  # Bottom 15% of image
        panel_y = h - panel_height
        
        # 1. Draw semi-transparent panel
        overlay = output.copy()
        cv2.rectangle(overlay, (0, panel_y), (w, h), self.colors["panel"], -1)
        alpha = 0.85
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        
        # 2. Draw top border for panel
        cv2.line(output, (0, panel_y), (w, panel_y), self.colors["accent"], 2)
        
        # 3. Prepare text
        card_name = card_info.get("name", "Unknown Card")
        set_name = card_info.get("set", "Unknown Set")
        confidence = card_info.get("confidence", 0.0)
        
        price_text = "Price: N/A"
        if price_data:
            price_text = f"Market: {price_data.display_price}"
            
        # 4. Draw Text
        # Card Name (Large, Top Left of Panel)
        self._draw_text(
            output, 
            card_name, 
            (20, panel_y + 40), 
            scale=1.2, 
            color=self.colors["text"]
        )
        
        # Set Name (Medium, Below Name)
        self._draw_text(
            output, 
            f"{set_name} | Conf: {confidence:.0%}", 
            (20, panel_y + 80), 
            scale=0.7, 
            color=(200, 200, 200)
        )
        
        # Price (Large, Right side)
        # Calculate text size to align right
        (p_w, p_h), _ = cv2.getTextSize(price_text, self.font, 1.2, self.thickness)
        price_x = w - p_w - 20
        
        self._draw_text(
            output, 
            price_text, 
            (price_x, panel_y + 55), 
            scale=1.2, 
            color=self.colors["price"]
        )
        
        # Source attribution (Small, Bottom Right)
        if price_data:
            source_text = f"Source: {price_data.source}"
            (s_w, s_h), _ = cv2.getTextSize(source_text, self.font, 0.5, 1)
            self._draw_text(
                output,
                source_text,
                (w - s_w - 20, h - 10),
                scale=0.5,
                color=(150, 150, 150),
                thickness=1
            )

        return output

    def _draw_text(
        self, 
        img: np.ndarray, 
        text: str, 
        pos: Tuple[int, int], 
        scale: float = 1.0, 
        color: Tuple[int, int, int] = (255, 255, 255),
        thickness: Optional[int] = None
    ):
        """Draw text with shadow for better visibility"""
        if thickness is None:
            thickness = self.thickness
            
        x, y = pos
        
        # Draw shadow
        cv2.putText(
            img, 
            text, 
            (x + 2, y + 2), 
            self.font, 
            scale, 
            self.colors["shadow"], 
            thickness + 1, 
            cv2.LINE_AA
        )
        
        # Draw text
        cv2.putText(
            img, 
            text, 
            (x, y), 
            self.font, 
            scale, 
            color, 
            thickness, 
            cv2.LINE_AA
        )
