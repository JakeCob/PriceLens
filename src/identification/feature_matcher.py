#!/usr/bin/env python3
"""
Feature-based card identification using ORB descriptors and FLANN matching.
"""

import sys
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.identification.identifier_base import IdentifierBase

logger = logging.getLogger(__name__)


class FeatureMatcher(IdentifierBase):
    """
    Card identification using ORB feature matching with FLANN matcher
    """

    def __init__(
        self,
        n_features: int = 1000,
        match_threshold: float = 0.75,
        min_matches: int = 10,
    ):
        """
        Initialize feature matcher

        Args:
            n_features: Number of ORB features to detect
            match_threshold: Low's ratio test threshold (typically 0.7-0.8)
            min_matches: Minimum good matches required for valid identification
        """
        self.n_features = n_features
        self.match_threshold = match_threshold
        self.min_matches = min_matches

        # Initialize ORB detector
        self.detector = cv2.ORB_create(nfeatures=n_features)

        # Initialize FLANN matcher for binary descriptors (ORB)
        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH,
            table_number=12,  # 12 hash tables
            key_size=20,  # Hash key length
            multi_probe_level=2,  # Number of bits to flip for multi-probe
        )
        search_params = dict(checks=50)  # Number of times to check the tree

        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

        # Card database
        self.card_features: Dict[str, np.ndarray] = {}
        self.card_metadata: Dict[str, Dict] = {}

        logger.info(
            f"FeatureMatcher initialized (n_features={n_features}, "
            f"threshold={match_threshold}, min_matches={min_matches})"
        )

    def load_database(self, database_path: str) -> None:
        """
        Load pre-computed card features from pickle file

        Args:
            database_path: Path to features pickle file
        """
        db_path = Path(database_path)

        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {database_path}")

        logger.info(f"Loading card database from: {database_path}")

        with open(db_path, "rb") as f:
            data = pickle.load(f)

        # Extract descriptors and metadata
        for card_id, card_data in data.items():
            self.card_features[card_id] = card_data["descriptors"]
            self.card_metadata[card_id] = card_data["metadata"]

        logger.info(f"Loaded {len(self.card_features)} cards from database")

    def compute_features(
        self, image: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Compute ORB features for an image

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

        # Detect and compute ORB features
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)

        return keypoints, descriptors

    def identify(self, image: np.ndarray, top_k: int = 3) -> List[Dict]:
        """
        Identify a card from its image

        Args:
            image: Card image (BGR format)
            top_k: Number of top matches to return

        Returns:
            List of matches sorted by confidence, each containing:
                - card_id: Database card ID
                - name: Card name
                - confidence: Match confidence (0-1)
                - num_matches: Number of good feature matches
                - metadata: Card metadata
        """
        # Compute features for input image
        keypoints, descriptors = self.compute_features(image)

        if descriptors is None or len(keypoints) == 0:
            logger.warning("No features detected in input image")
            return []

        logger.debug(f"Detected {len(keypoints)} keypoints in query image")

        # Match against each card in database
        match_results = []

        for card_id, card_descriptors in self.card_features.items():
            try:
                # Match using FLANN (k=2 for Lowe's ratio test)
                matches = self.matcher.knnMatch(descriptors, card_descriptors, k=2)

                # Apply Lowe's ratio test
                good_matches = self._apply_ratio_test(matches)

                if len(good_matches) < self.min_matches:
                    continue

                # Calculate confidence score
                confidence = min(1.0, len(good_matches) / 100.0)  # Normalize to 0-1

                match_results.append({
                    "card_id": card_id,
                    "name": self.card_metadata[card_id]["name"],
                    "set": self.card_metadata[card_id].get("set", "Unknown"),
                    "num_matches": len(good_matches),
                    "confidence": confidence,
                    "metadata": self.card_metadata[card_id],
                })

            except Exception as e:
                logger.warning(f"Error matching against {card_id}: {e}")
                continue

        # Sort by number of matches (primary) and confidence (secondary)
        match_results.sort(
            key=lambda x: (x["num_matches"], x["confidence"]), reverse=True
        )

        logger.debug(f"Found {len(match_results)} potential matches")

        # Return top-K matches
        return match_results[:top_k]

    def _apply_ratio_test(self, matches: List) -> List:
        """
        Apply Lowe's ratio test to filter good matches

        Args:
            matches: List of matches from knnMatch (k=2)

        Returns:
            List of good matches that pass ratio test
        """
        good_matches = []

        for match_pair in matches:
            # Need at least 2 matches for ratio test
            if len(match_pair) < 2:
                continue

            m, n = match_pair

            # Lowe's ratio test: best match distance should be significantly
            # better than second-best match
            if m.distance < self.match_threshold * n.distance:
                good_matches.append(m)

        return good_matches


# Example usage
if __name__ == "__main__":
    from src.utils.logging_config import setup_logging

    setup_logging(level="INFO")

    # Initialize matcher
    matcher = FeatureMatcher(n_features=1000, match_threshold=0.75, min_matches=10)

    # Load database
    matcher.load_database("data/features/base_set_features.pkl")

    # Test with a database card (should get high confidence)
    logger.info("Testing identification on database card...")
    test_image = cv2.imread("data/card_database/base_set/Charizard_base1-4.jpg")

    if test_image is not None:
        matches = matcher.identify(test_image, top_k=3)

        logger.info(f"\nTop matches:")
        for i, match in enumerate(matches, 1):
            logger.info(
                f"  {i}. {match['name']} ({match['card_id']}) - "
                f"{match['num_matches']} matches, "
                f"confidence: {match['confidence']:.2f}"
            )
    else:
        logger.error("Could not load test image")
