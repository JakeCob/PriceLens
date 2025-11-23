"""
Generate a custom price template JSON file from existing feature databases.
"""
import json
import pickle
import logging
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import setup_logging

setup_logging(level="INFO")
logger = logging.getLogger("price_gen")

def generate_template(output_path: str = "data/custom_prices.json"):
    output_file = Path(output_path)
    
    # Load existing prices if any
    prices = {}
    if output_file.exists():
        try:
            with open(output_file, "r") as f:
                prices = json.load(f)
            logger.info(f"Loaded {len(prices)} existing prices")
        except Exception:
            pass
            
    # Scan features
    features_dir = Path("data/features")
    count = 0
    
    for pkl_file in features_dir.glob("*.pkl"):
        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
                
            for card_id, card_data in data.items():
                if card_id not in prices:
                    metadata = card_data.get("metadata", {})
                    name = metadata.get("name", "Unknown")
                    
                    prices[card_id] = {
                        "price": 0.00,
                        "currency": "USD",
                        "name": name,  # Helpful comment
                        "set": metadata.get("set", "Unknown")
                    }
                    count += 1
                    
        except Exception as e:
            logger.error(f"Error reading {pkl_file}: {e}")
            
    # Save
    with open(output_file, "w") as f:
        json.dump(prices, f, indent=2)
        
    logger.info(f"Added {count} new cards to {output_path}")
    logger.info(f"Total cards in price file: {len(prices)}")

if __name__ == "__main__":
    generate_template()
