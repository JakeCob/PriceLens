"""
Cache Warmer
Iterates through all known cards and fetches their prices to populate the cache.
Run this in the background to ensure instant lookups for the user.
"""
import time
import pickle
import logging
from pathlib import Path
from tqdm import tqdm

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import setup_logging
from src.api.service import PriceService

setup_logging(level="INFO")
logger = logging.getLogger("cache_warmer")

def warm_cache():
    service = PriceService()
    
    # Load all card IDs from features
    card_ids = []
    features_dir = Path("data/features")
    
    for pkl_file in features_dir.glob("*.pkl"):
        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
                card_ids.extend(data.keys())
        except Exception as e:
            logger.error(f"Error reading {pkl_file}: {e}")
            
    logger.info(f"Found {len(card_ids)} cards to warm up.")
    
    # Filter for ME1/ME2 only (prioritize them)
    priority_ids = [cid for cid in card_ids if cid.startswith("me1") or cid.startswith("me2")]
    logger.info(f"Prioritizing {len(priority_ids)} ME1/ME2 cards.")
    
    for card_id in tqdm(priority_ids):
        try:
            # Check if already cached
            if service.cache.get(card_id):
                continue
                
            # Fetch (this will cache it)
            # We use the service directly, which handles the API call and caching
            price = service.get_price(card_id)
            
            if price:
                logger.info(f"Cached {card_id}: ${price.market}")
            else:
                logger.warning(f"Failed to fetch {card_id}")
                
            # Be nice to the API
            time.sleep(1.0)
            
        except Exception as e:
            logger.error(f"Error processing {card_id}: {e}")

if __name__ == "__main__":
    warm_cache()
