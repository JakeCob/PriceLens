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

try:
    import easyocr
except ImportError:
    easyocr = None

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

try:
    import torch
    from src.identification.embedding_generator import EmbeddingGenerator
except ImportError:
    torch = None
    EmbeddingGenerator = None

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.identification.identifier_base import IdentifierBase

logger = logging.getLogger(__name__)


class FeatureMatcher(IdentifierBase):
    """
    Enhanced card identification using:
    1. ORB Feature Matching (Fast)
    2. ChromaDB Vector Search (Semantic)
    3. EasyOCR (Fallback)
    """

    def __init__(
        self,
        n_features: int = 1000,
        match_threshold: float = 0.75,
        min_matches: int = 10,
        use_ocr: bool = True,
        use_vector_db: bool = True,
    ):
        """
        Initialize feature matcher
        
        Args:
            n_features: Number of ORB features
            match_threshold: Lowe's ratio test threshold
            min_matches: Minimum good matches
            use_ocr: Enable OCR fallback
            use_vector_db: Enable ChromaDB vector search
        """
        self.n_features = n_features
        self.match_threshold = match_threshold
        self.min_matches = min_matches
        
        # Initialize ORB
        self.detector = cv2.ORB_create(nfeatures=n_features)
        
        # Initialize FLANN
        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH,
            table_number=12,
            key_size=20,
            multi_probe_level=2,
        )
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Initialize OCR
        self.ocr_reader = None
        if use_ocr and easyocr:
            try:
                logger.info("Initializing EasyOCR...")
                self.ocr_reader = easyocr.Reader(['en'], gpu=True)
            except Exception as e:
                logger.warning(f"Failed to init OCR: {e}")

        # Initialize ChromaDB and Embedder
        self.chroma_client = None
        self.collection = None
        self.embedder = None
        
        if use_vector_db and chromadb and EmbeddingGenerator:
            try:
                logger.info("Initializing ChromaDB and Embedder...")
                self.chroma_client = chromadb.PersistentClient(path="data/chromadb")
                self.collection = self.chroma_client.get_or_create_collection(
                    name="pokemon_cards",
                    metadata={"hnsw:space": "cosine"}
                )
                
                device = "cuda" if torch and torch.cuda.is_available() else "cpu"
                self.embedder = EmbeddingGenerator(device=device)
                
            except Exception as e:
                logger.warning(f"Failed to init ChromaDB/Embedder: {e}")

        # Card database
        self.card_features: Dict[str, np.ndarray] = {}
        self.card_metadata: Dict[str, Dict] = {}

        logger.info(
            f"FeatureMatcher initialized (OCR={bool(self.ocr_reader)}, "
            f"VectorDB={bool(self.collection)})"
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
        
        # Vector Search Fallback/Enhancement
        if (not match_results or match_results[0]["confidence"] < 0.6) and self.collection and self.embedder:
            logger.info("Low confidence ORB match. Attempting Vector Search...")
            try:
                embedding = self.embedder.generate(image)
                results = self.collection.query(
                    query_embeddings=[embedding.tolist()],
                    n_results=top_k
                )
                
                if results and results['ids'] and len(results['ids'][0]) > 0:
                    ids = results['ids'][0]
                    distances = results['distances'][0]
                    metadatas = results['metadatas'][0]
                    
                    for i, card_id in enumerate(ids):
                        # Convert distance to confidence (cosine distance is 0-2, usually small for matches)
                        # 0 distance = 1.0 confidence
                        dist = distances[i]
                        confidence = max(0.0, 1.0 - dist)
                        
                        # Check if already in match_results
                        existing = next((m for m in match_results if m["card_id"] == card_id), None)
                        if existing:
                            # Boost confidence if both methods agree
                            existing["confidence"] = max(existing["confidence"], confidence)
                            existing["method"] = "hybrid"
                        else:
                            match_results.append({
                                "card_id": card_id,
                                "name": metadatas[i]["name"],
                                "set": metadatas[i]["set"],
                                "num_matches": 0,
                                "confidence": confidence,
                                "metadata": metadatas[i],
                                "method": "vector"
                            })
                            
                    # Re-sort
                    match_results.sort(key=lambda x: x["confidence"], reverse=True)
                    
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")

        # OCR Fallback
        if (not match_results or match_results[0]["confidence"] < 0.4) and self.ocr_reader:
            logger.info("Low confidence match. Attempting OCR...")
            ocr_results = self._identify_with_ocr(image)
            if ocr_results:
                match_results.extend(ocr_results)
                # Re-sort
                match_results.sort(key=lambda x: x["confidence"], reverse=True)

        # Return top-K matches
        return match_results[:top_k]

    def _identify_with_ocr(self, image: np.ndarray) -> List[Dict]:
        """
        Attempt to identify card using OCR text reading
        """
        try:
            # Read text
            result = self.ocr_reader.readtext(image)
            detected_text = " ".join([text[1] for text in result]).lower()
            
            logger.debug(f"OCR Text: {detected_text}")
            
            # Simple keyword matching against database
            # In a real system, this would use a search index or vector DB
            potential_matches = []
            
            for card_id, metadata in self.card_metadata.items():
                card_name = metadata["name"].lower()
                if card_name in detected_text:
                    # Calculate a simple confidence based on name length match
                    confidence = 0.6  # Base confidence for OCR match
                    
                    potential_matches.append({
                        "card_id": card_id,
                        "name": metadata["name"],
                        "set": metadata.get("set", "Unknown"),
                        "num_matches": 0,  # No visual matches
                        "confidence": confidence,
                        "metadata": metadata,
                        "method": "ocr"
                    })
            
            return potential_matches
            
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return []

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
    test_image = cv2.imread("data/card_database/base1/Charizard_base1-4.jpg")

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
