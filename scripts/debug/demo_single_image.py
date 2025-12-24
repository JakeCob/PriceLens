#!/usr/bin/env python3
"""
PriceLens Demo: Single Image Processing
Combines Identification, Price API, and Overlay Rendering.
"""

import sys
import argparse
import logging
from pathlib import Path
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.identification.feature_matcher import FeatureMatcher
from src.api.service import PriceService
from src.overlay.renderer import OverlayRenderer
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def process_image(image_path: str, output_path: str):
    """
    Process a single image: Identify -> Get Price -> Render Overlay
    """
    logger.info(f"Processing image: {image_path}")
    
    # 1. Load Image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Could not load image: {image_path}")
        return False

    # 2. Initialize Components
    logger.info("Initializing components...")
    matcher = FeatureMatcher()
    matcher.load_database("data/features/base_set_features.pkl")
    
    price_service = PriceService()
    renderer = OverlayRenderer()

    # 3. Identify Card
    logger.info("Identifying card...")
    matches = matcher.identify(image, top_k=1)
    
    if not matches:
        logger.warning("No card identified!")
        return False
        
    top_match = matches[0]
    card_id = top_match["card_id"]
    card_name = top_match["name"]
    confidence = top_match["confidence"]
    
    logger.info(f"Identified: {card_name} ({card_id}) - Conf: {confidence:.2f}")

    # 4. Fetch Price
    logger.info(f"Fetching price for {card_id}...")
    price_data = price_service.get_price(card_id)
    
    if price_data:
        logger.info(f"Price found: {price_data.display_price}")
    else:
        logger.warning("Price not found")

    # 5. Render Overlay
    logger.info("Rendering overlay...")
    result_image = renderer.draw_overlay(image, top_match, price_data)

    # 6. Save Result
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(str(output_file), result_image)
    logger.info(f"Result saved to: {output_file}")
    
    return True


if __name__ == "__main__":
    setup_logging(level="INFO")
    
    parser = argparse.ArgumentParser(description="PriceLens Single Image Demo")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="output/result.jpg", help="Path to output image")
    
    args = parser.parse_args()
    
    success = process_image(args.image, args.output)
    sys.exit(0 if success else 1)
