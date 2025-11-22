#!/usr/bin/env python3
"""
Extract ORB+BEBLID features from card database images.
Pre-computes features for fast identification.
"""

import sys
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract and store card features using ORB+BEBLID"""

    def __init__(self, n_features: int = 1000):
        """
        Initialize feature extractor

        Args:
            n_features: Number of ORB features to detect
        """
        self.n_features = n_features

        # Initialize ORB detector (also computes ORB descriptors)
        self.detector = cv2.ORB_create(nfeatures=n_features)

        # Note: Using ORB descriptors instead of BEBLID due to opencv-contrib issues
        # ORB descriptors are binary and work well for card matching
        logger.info(
            f"Feature extractor initialized with ORB (n_features={n_features})"
        )

    def compute_features(
        self, image: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Compute ORB keypoints and BEBLID descriptors for an image

        Args:
            image: Input image (BGR format)

        Returns:
            Tuple of (keypoints, descriptors)
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect keypoints and compute ORB descriptors in one step
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)

        return keypoints, descriptors

    def serialize_keypoints(self, keypoints: List[cv2.KeyPoint]) -> List[Dict]:
        """
        Serialize keypoints for storage

        Args:
            keypoints: List of OpenCV KeyPoint objects

        Returns:
            List of serializable dictionaries
        """
        serialized = []
        for kp in keypoints:
            serialized.append({
                "pt": kp.pt,
                "size": kp.size,
                "angle": kp.angle,
                "response": kp.response,
                "octave": kp.octave,
                "class_id": kp.class_id,
            })
        return serialized


def extract_database_features(
    database_path: str = "data/card_database",
    output_path: str = "data/features",
) -> bool:
    """
    Extract features from all cards in the database

    Args:
        database_path: Path to card database directory
        output_path: Path to save extracted features

    Returns:
        True if successful
    """
    db_path = Path(database_path)
    output_path = Path(output_path)

    # Load card index
    index_file = db_path / "index.json"
    if not index_file.exists():
        logger.error(f"Index file not found: {index_file}")
        return False

    with open(index_file, "r") as f:
        index_data = json.load(f)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize feature extractor
    extractor = FeatureExtractor(n_features=1000)

    # Process each set
    all_features = {}
    total_cards = 0
    successful_cards = 0

    logger.info("=" * 60)
    logger.info("EXTRACTING CARD FEATURES")
    logger.info("=" * 60)

    for set_code, set_data in index_data.get("sets", {}).items():
        logger.info(f"\nProcessing set: {set_data['name']} ({set_code})")
        set_folder = db_path / set_data["folder"]

        for card in set_data.get("cards", []):
            total_cards += 1
            card_id = card["id"]
            card_name = card["name"]
            image_path = db_path / card["image_path"]

            logger.info(f"  Processing: {card_name} ({card_id})")

            # Load image
            if not image_path.exists():
                logger.warning(f"    ❌ Image not found: {image_path}")
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"    ❌ Failed to load image: {image_path}")
                continue

            # Extract features
            try:
                keypoints, descriptors = extractor.compute_features(image)

                if descriptors is None or len(keypoints) == 0:
                    logger.warning(f"    ⚠️  No features detected")
                    continue

                # Serialize keypoints for storage
                serialized_kp = extractor.serialize_keypoints(keypoints)

                # Store features
                all_features[card_id] = {
                    "keypoints": serialized_kp,
                    "descriptors": descriptors,
                    "metadata": {
                        "id": card_id,
                        "name": card_name,
                        "set": set_data["name"],
                        "set_code": set_code,
                        "rarity": card.get("rarity"),
                        "type": card.get("type"),
                    },
                }

                logger.info(
                    f"    ✓ Extracted {len(keypoints)} keypoints, "
                    f"{descriptors.shape[0]} descriptors"
                )
                successful_cards += 1

            except Exception as e:
                logger.error(f"    ❌ Error extracting features: {e}")
                continue

    # Save features
    if successful_cards > 0:
        features_file = output_path / "base_set_features.pkl"
        logger.info(f"\nSaving features to: {features_file}")

        with open(features_file, "wb") as f:
            pickle.dump(all_features, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Update index.json to mark features as computed
        for set_code, set_data in index_data.get("sets", {}).items():
            for card in set_data.get("cards", []):
                if card["id"] in all_features:
                    card["features_computed"] = True

        with open(index_file, "w") as f:
            json.dump(index_data, f, indent=4)

        logger.info("=" * 60)
        logger.info("FEATURE EXTRACTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total cards processed: {total_cards}")
        logger.info(f"Successfully extracted: {successful_cards}")
        logger.info(f"Failed: {total_cards - successful_cards}")
        logger.info(f"Features saved to: {features_file}")
        logger.info(f"Index updated with features_computed flag")

        # Show file size
        file_size = features_file.stat().st_size
        logger.info(f"Feature file size: {file_size / 1024:.1f} KB")

        return True
    else:
        logger.error("No features extracted!")
        return False


if __name__ == "__main__":
    setup_logging(level="INFO")

    success = extract_database_features()
    sys.exit(0 if success else 1)
