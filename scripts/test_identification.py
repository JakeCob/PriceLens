#!/usr/bin/env python3
"""
Test card identification on all database cards.
Should achieve 100% accuracy on cards in the database.
"""

import sys
import time
from pathlib import Path

import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.identification.feature_matcher import FeatureMatcher
from src.utils.logging_config import setup_logging
import logging

logger = logging.getLogger(__name__)


def test_identification():
    """Test identification on all database cards"""
    logger.info("=" * 60)
    logger.info("TESTING CARD IDENTIFICATION")
    logger.info("=" * 60)

    # Initialize matcher
    matcher = FeatureMatcher(n_features=1000, match_threshold=0.75, min_matches=10)

    # Load database
    matcher.load_database("data/features/base_set_features.pkl")

    # Test cards
    test_cards = [
        ("base1-1", "Alakazam", "data/card_database/base1/Alakazam_base1-1.jpg"),
        ("base1-2", "Blastoise", "data/card_database/base1/Blastoise_base1-2.jpg"),
        ("base1-3", "Chansey", "data/card_database/base1/Chansey_base1-3.jpg"),
        ("base1-4", "Charizard", "data/card_database/base1/Charizard_base1-4.jpg"),
        ("base1-5", "Clefairy", "data/card_database/base1/Clefairy_base1-5.jpg"),
    ]

    correct = 0
    total = len(test_cards)
    total_time = 0
    confidences = []

    logger.info(f"\nTesting {total} cards...")
    logger.info("")

    for card_id, name, image_path in test_cards:
        logger.info(f"Testing: {name} ({card_id})")

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"  ❌ Could not load image: {image_path}")
            continue

        # Time the identification
        start_time = time.time()
        matches = matcher.identify(image, top_k=3)
        elapsed = (time.time() - start_time) * 1000  # Convert to ms

        total_time += elapsed

        # Check top match
        if matches and len(matches) > 0:
            top_match = matches[0]

            if top_match["card_id"] == card_id:
                logger.info(
                    f"  ✓ Correct! {top_match['name']} - "
                    f"{top_match['num_matches']} matches, "
                    f"confidence: {top_match['confidence']:.2f}, "
                    f"time: {elapsed:.1f}ms"
                )
                correct += 1
                confidences.append(top_match['confidence'])
            else:
                logger.error(
                    f"  ❌ Wrong match: {top_match['name']} ({top_match['card_id']}) - "
                    f"{top_match['num_matches']} matches, "
                    f"confidence: {top_match['confidence']:.2f}"
                )
        else:
            logger.error("  ❌ No matches found")

        logger.info("")

    # Summary
    accuracy = (correct / total) * 100 if total > 0 else 0
    avg_time = total_time / total if total > 0 else 0
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    logger.info(f"Average confidence: {avg_confidence:.2f}")
    logger.info(f"Average time per card: {avg_time:.1f}ms")
    logger.info(f"Total time: {total_time:.1f}ms")
    logger.info("")

    if accuracy == 100:
        logger.info("✅ ALL CARDS IDENTIFIED CORRECTLY!")
    else:
        logger.warning(f"⚠️  {total - correct} card(s) misidentified")

    return accuracy == 100


if __name__ == "__main__":
    setup_logging(level="INFO")
    success = test_identification()
    sys.exit(0 if success else 1)
