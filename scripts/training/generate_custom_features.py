"""
Generate features for custom card sets (ME1, ME2) from local images.
"""
import cv2
import pickle
import logging
import re
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import setup_logging

setup_logging(level="INFO")
logger = logging.getLogger("feature_gen")

def generate_features(set_id: str, input_dir: Path, output_file: Path):
    """
    Generate features for a set of images
    """
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return

    logger.info(f"Generating features for set {set_id} from {input_dir}")
    
    # Initialize ORB
    orb = cv2.ORB_create(nfeatures=1000)
    
    database = {}
    
    files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    logger.info(f"Found {len(files)} images")
    
    for file_path in tqdm(files):
        try:
            # Parse filename: Name_id.jpg -> Name, id
            # Example: Bulbasaur_me1-1.jpg
            stem = file_path.stem
            
            # Split by last underscore to separate name and ID
            if "_" in stem:
                parts = stem.rsplit("_", 1)
                name = parts[0].replace("-", " ") # Clean up name
                card_id = parts[1]
            else:
                # Fallback
                name = stem
                card_id = f"{set_id}-{stem}"
            
            # Read image
            img = cv2.imread(str(file_path))
            if img is None:
                logger.warning(f"Could not read {file_path}")
                continue
                
            # Compute features
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            
            if descriptors is None:
                logger.warning(f"No features found in {file_path}")
                continue
                
            # Store in database
            database[card_id] = {
                "descriptors": descriptors,
                "metadata": {
                    "id": card_id,
                    "name": name,
                    "set": set_id,
                    "image_path": str(file_path)
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            
    # Save to pickle
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(database, f)
        
    logger.info(f"Saved {len(database)} cards to {output_file}")

if __name__ == "__main__":
    base_dir = Path("data/card_database")
    feature_dir = Path("data/features")
    
    # Generate for ME1
    generate_features("me1", base_dir / "me1", feature_dir / "me1_features.pkl")
    
    # Generate for ME2
    generate_features("me2", base_dir / "me2", feature_dir / "me2_features.pkl")
