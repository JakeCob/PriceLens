#!/usr/bin/env python3
"""
FAISS-based approximate nearest neighbor matcher for fast card identification.

Uses FAISS IndexBinaryIVF for binary ORB descriptors, providing 10-50x speedup
over sequential FLANN matching.
"""

import logging
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)


class FAISSMatcher:
    """
    Fast card identification using FAISS binary index.
    
    Builds an IVF index over all card descriptors and uses voting
    to identify which card best matches a query image's descriptors.
    """
    
    def __init__(
        self,
        n_probe: int = 8,
        n_neighbors: int = 5,
        distance_threshold: int = 64,  # Hamming distance threshold for 256-bit descriptors
    ):
        """
        Initialize FAISS matcher.
        
        Args:
            n_probe: Number of clusters to search (higher = more accurate, slower)
            n_neighbors: Number of neighbors to retrieve per query descriptor
            distance_threshold: Max Hamming distance to consider a valid match
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not installed. Run: pip install faiss-cpu")
        
        self.n_probe = n_probe
        self.n_neighbors = n_neighbors
        self.distance_threshold = distance_threshold
        
        self.index: Optional[faiss.IndexBinaryIVF] = None
        self.descriptor_to_card: List[str] = []
        self.card_metadata: Dict[str, Dict] = {}
        self.n_cards = 0
        
        logger.info(f"FAISSMatcher initialized (n_probe={n_probe}, n_neighbors={n_neighbors})")
    
    def build_index(
        self, 
        card_features: Dict[str, np.ndarray],
        card_metadata: Dict[str, Dict],
        n_lists: int = 32,
    ) -> None:
        """
        Build FAISS index from card features.
        
        Args:
            card_features: Dict mapping card_id to ORB descriptors (N x 32 uint8)
            card_metadata: Dict mapping card_id to metadata
            n_lists: Number of IVF clusters (sqrt(N) is typical)
        """
        logger.info(f"Building FAISS index for {len(card_features)} cards...")
        
        self.card_metadata = card_metadata
        self.n_cards = len(card_features)
        
        # Stack all descriptors and track which card each belongs to
        all_descriptors = []
        self.descriptor_to_card = []
        
        for card_id, descriptors in card_features.items():
            if descriptors is None or len(descriptors) == 0:
                continue
            
            # Ensure descriptors are uint8 and contiguous
            descs = np.ascontiguousarray(descriptors.astype(np.uint8))
            all_descriptors.append(descs)
            self.descriptor_to_card.extend([card_id] * len(descs))
        
        if not all_descriptors:
            raise ValueError("No valid descriptors found in card features")
        
        # Stack into single array
        all_descriptors = np.vstack(all_descriptors)
        n_descriptors = len(all_descriptors)
        
        logger.info(f"Total descriptors: {n_descriptors:,}")
        
        # ORB descriptors are 256-bit (32 bytes)
        d = all_descriptors.shape[1] * 8  # Bits
        
        # Create quantizer and IVF index for binary descriptors
        # Adjust n_lists based on dataset size
        n_lists = min(n_lists, n_descriptors // 39 + 1)  # At least 39 vectors per list
        n_lists = max(1, n_lists)
        
        quantizer = faiss.IndexBinaryFlat(d)
        self.index = faiss.IndexBinaryIVF(quantizer, d, n_lists)
        
        # Train the index
        logger.info(f"Training FAISS index with {n_lists} clusters...")
        self.index.train(all_descriptors)
        
        # Add all descriptors
        self.index.add(all_descriptors)
        
        # Set search parameters
        self.index.nprobe = self.n_probe
        
        logger.info(f"FAISS index built: {n_descriptors:,} descriptors, {n_lists} clusters")
    
    def load_database(self, database_path: str) -> None:
        """
        Load card features from pickle file and build index.
        
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
        card_features = {}
        card_metadata = {}
        
        for card_id, card_data in data.items():
            card_features[card_id] = card_data["descriptors"]
            card_metadata[card_id] = card_data["metadata"]
        
        logger.info(f"Loaded {len(card_features)} cards from database")
        
        # Build the FAISS index
        self.build_index(card_features, card_metadata)
    
    def identify(
        self, 
        query_descriptors: np.ndarray,
        top_k: int = 3,
        min_votes: int = 5,
    ) -> List[Dict]:
        """
        Identify card from query descriptors using batch ANN search + voting.
        
        Args:
            query_descriptors: ORB descriptors from query image (N x 32 uint8)
            top_k: Number of top matches to return
            min_votes: Minimum votes required to be a valid match
            
        Returns:
            List of matches sorted by confidence
        """
        if self.index is None:
            logger.warning("FAISS index not built")
            return []
        
        if query_descriptors is None or len(query_descriptors) == 0:
            logger.warning("No query descriptors provided")
            return []
        
        # Ensure descriptors are contiguous uint8
        query = np.ascontiguousarray(query_descriptors.astype(np.uint8))
        
        # Batch search for all query descriptors at once
        distances, indices = self.index.search(query, self.n_neighbors)
        
        # Vote: count which card_ids appear most with good matches
        card_votes: Counter = Counter()
        card_distances: Dict[str, List[int]] = {}
        
        for i, (dists, idxs) in enumerate(zip(distances, indices)):
            for dist, idx in zip(dists, idxs):
                if idx < 0:  # Invalid index
                    continue
                if dist > self.distance_threshold:
                    continue
                
                card_id = self.descriptor_to_card[idx]
                card_votes[card_id] += 1
                
                if card_id not in card_distances:
                    card_distances[card_id] = []
                card_distances[card_id].append(dist)
        
        if not card_votes:
            logger.debug("No matches found within distance threshold")
            return []
        
        # Convert votes to match results
        results = []
        max_possible_votes = len(query_descriptors) * self.n_neighbors
        
        for card_id, votes in card_votes.most_common(top_k * 2):  # Get extra for filtering
            if votes < min_votes:
                continue
            
            # Calculate confidence based on vote proportion and average distance
            vote_confidence = min(1.0, votes / 100)
            avg_distance = np.mean(card_distances[card_id])
            distance_confidence = 1.0 - (avg_distance / 256)  # Normalize to 0-1
            
            # Combined confidence
            confidence = 0.6 * vote_confidence + 0.4 * distance_confidence
            
            metadata = self.card_metadata.get(card_id, {})
            
            results.append({
                "card_id": card_id,
                "name": metadata.get("name", "Unknown"),
                "set": metadata.get("set", "Unknown"),
                "num_matches": votes,
                "confidence": confidence,
                "metadata": metadata,
                "method": "faiss",
            })
        
        # Sort by confidence
        results.sort(key=lambda x: x["confidence"], reverse=True)
        
        if results:
            logger.debug(f"FAISS matched: {results[0]['name']} with {results[0]['num_matches']} votes")
        
        return results[:top_k]


# Export availability flag
__all__ = ["FAISSMatcher", "FAISS_AVAILABLE"]
