#!/usr/bin/env python3
"""
Build ORB feature database for me1 and me2 card sets only.
This creates a pickle file that the FeatureMatcher can load for fast identification.
"""

import pickle
import logging
from pathlib import Path
import cv2
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_features():
    # ORB detector - balanced features for speed + accuracy
    orb = cv2.ORB_create(nfeatures=1000)
    
    # Card database
    card_db_path = Path("data/card_database")
    sets_to_index = ["me1", "me2"]
    
    features_db = {}
    
    for set_name in sets_to_index:
        set_path = card_db_path / set_name
        if not set_path.exists():
            logger.warning(f"Set folder not found: {set_path}")
            continue
        
        logger.info(f"Processing set: {set_name}")
        
        for img_path in set_path.glob("*.jpg"):
            # Parse card ID from filename (e.g., Honchkrow_me2-58.jpg -> me2-58)
            filename = img_path.stem  # e.g., "Honchkrow_me2-58"
            parts = filename.rsplit("_", 1)
            if len(parts) == 2:
                card_name = parts[0].replace("_", " ")
                card_id = parts[1]  # e.g., "me2-58"
            else:
                card_name = filename
                card_id = f"{set_name}-{filename}"

            # Load and compute ORB features
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Failed to load: {img_path}")
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # CLAHE preprocessing (must match feature_matcher.py)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
            keypoints, descriptors = orb.detectAndCompute(gray, None)

            if descriptors is None or len(keypoints) < 10:
                logger.warning(f"Insufficient features for: {img_path}")
                continue
            
            features_db[card_id] = {
                "descriptors": descriptors,
                "metadata": {
                    "name": card_name,
                    "set": set_name,
                    "image_path": str(img_path),
                    "card_id": card_id
                }
            }
            
            logger.debug(f"Indexed: {card_id} ({card_name}) - {len(keypoints)} keypoints")
    
    # Save to pickle
    output_path = Path("data/me1_me2_features.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        pickle.dump(features_db, f)
    
    logger.info(f"Saved {len(features_db)} cards to {output_path}")
    return output_path

if __name__ == "__main__":
    build_features()
