"""Base class for card identifiers"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Optional


class IdentifierBase(ABC):
    """Abstract base class for card identification implementations"""

    @abstractmethod
    def identify(self, image: np.ndarray, top_k: int = 3) -> List[Dict]:
        """
        Identify a card from its image

        Args:
            image: Card image
            top_k: Number of top matches to return

        Returns:
            List of matches with card_id, name, confidence
        """
        pass

    @abstractmethod
    def load_database(self, database_path: str) -> None:
        """
        Load card database

        Args:
            database_path: Path to database file
        """
        pass