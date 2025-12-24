#!/usr/bin/env python3
"""
Download Full Card Set Data
Fetches metadata and images for an entire Pokemon TCG set (e.g., Base Set).
"""

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/download_set.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def _normalize_release_date(raw_date: Optional[str]) -> str:
    """Convert API release date formats to YYYY-MM-DD strings."""
    if not raw_date:
        return ""

    cleaned = raw_date.strip()
    if "/" in cleaned:
        parts = cleaned.split("/")
        if len(parts) == 3:
            year, month, day = parts
            if len(year) == 4:
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    return cleaned


def _load_or_init_index(index_file: Path) -> Dict:
    """Load the master card index, ensuring required keys exist."""
    base_structure: Dict = {
        "version": "1.0",
        "last_updated": None,
        "total_sets": 0,
        "total_cards": 0,
        "sets": {},
    }

    if index_file.exists():
        try:
            with open(index_file, "r") as f:
                index_data = json.load(f)
        except Exception:
            logger.warning("Could not parse existing index.json, starting fresh.")
            index_data = base_structure
    else:
        index_data = base_structure

    # Ensure required keys exist
    index_data.setdefault("version", "1.0")
    index_data.setdefault("sets", {})
    index_data.setdefault("total_sets", 0)
    index_data.setdefault("total_cards", 0)
    index_data.setdefault("last_updated", None)

    return index_data


def _update_index_metadata(index_data: Dict):
    """Update aggregate metadata for the index."""
    sets = index_data.get("sets", {})
    index_data["total_sets"] = len(sets)
    index_data["total_cards"] = sum(len(set_data.get("cards", [])) for set_data in sets.values())
    index_data["last_updated"] = datetime.now(timezone.utc).isoformat()


def _make_session(timeout: int = 30) -> requests.Session:
    """Create a requests session with retries/backoff."""
    retry_strategy = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.request_timeout = timeout
    return session


def download_set(set_id: str, output_dir: str, base_url: str, headers: Dict, timeout: int = 30):
    """
    Download all cards for a given set ID.
    """
    logger.info(f"Starting download for set: {set_id}")
    session = _make_session(timeout=timeout)
    
    # 1. Fetch Set Data from API
    api_url = f"{base_url}/cards"
    params = {
        "q": f"set.id:{set_id}",
        "pageSize": 250,  # Base set is 102, so 250 is plenty
        "orderBy": "number"
    }
    
    try:
        logger.info("Fetching card list from API...")
        response = session.get(api_url, params=params, timeout=timeout, headers=headers)
        response.raise_for_status()
        data = response.json()
        cards = data.get("data", [])
        
        logger.info(f"Found {len(cards)} cards in set {set_id}")
        
    except Exception as e:
        logger.error(f"Failed to fetch set data: {e}")
        return False

    # 2. Prepare Output Directory and index
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    index_file = output_path / "index.json"
    index_data = _load_or_init_index(index_file)

    # Migrate legacy schema where sets were stored at the root level
    legacy_entry = None
    if set_id in index_data and isinstance(index_data[set_id], dict):
        logger.info("Migrating legacy index entry for %s into sets[...]", set_id)
        legacy_entry = index_data.pop(set_id)

    existing_set_entry = index_data["sets"].get(set_id) or legacy_entry or {}
    existing_cards = {
        card["id"]: card for card in existing_set_entry.get("cards", [])
    }

    set_folder = set_id
    base_dir = output_path / set_folder
    base_dir.mkdir(parents=True, exist_ok=True)
    
    first_card = cards[0] if cards else None
    set_name = first_card["set"]["name"] if first_card else set_id
    release_date = _normalize_release_date(first_card["set"].get("releaseDate")) if first_card else ""

    # 3. Download Images
    logger.info("Downloading card images...")
    
    success_count = 0
    processed_cards: List[Dict] = []
    
    for card in tqdm(cards, desc="Downloading"):
        card_id = card["id"]
        image_url = card["images"]["large"]
        card_number = card["number"]
        card_name = card["name"]
        
        # Sanitize filename
        safe_name = "".join([c for c in card_name if c.isalnum() or c in (' ', '-', '_')]).strip()
        filename = f"{safe_name}_{card_id}.jpg"
        filepath = base_dir / filename
        
        # Download Image
        if not filepath.exists():
            try:
                img_response = requests.get(image_url, timeout=30)
                img_response.raise_for_status()
                with open(filepath, "wb") as f:
                    f.write(img_response.content)
            except Exception as e:
                logger.error(f"Failed to download {card_name}: {e}")
                continue
        
        # Add to Index
        card_entry = {
            "id": card_id,
            "name": card_name,
            "number": card_number,
            "rarity": card.get("rarity", "Unknown"),
            "type": card.get("types", ["Unknown"])[0],
            "image_path": str(Path(set_folder) / filename),
            "features_computed": False
        }
        
        # Check if card already exists in index to preserve 'features_computed' status if true
        existing_entry = existing_cards.get(card_id)
        if existing_entry and existing_entry.get("features_computed"):
            card_entry["features_computed"] = True
            
        processed_cards.append(card_entry)
        success_count += 1

    # 4. Save Index
    # Sort cards by number (handles "102a" style suffixes)
    def sort_key(card_data):
        try:
            number_part = "".join(filter(str.isdigit, card_data.get("number", "")))
            return int(number_part)
        except Exception:
            return float("inf")

    processed_cards.sort(key=sort_key)

    index_data["sets"][set_id] = {
        "name": set_name,
        "code": set_id,
        "release_date": release_date,
        "total_cards": len(cards),
        "cards_in_database": len(processed_cards),
        "folder": set_folder,
        "cards": processed_cards,
    }

    _update_index_metadata(index_data)

    with open(index_file, "w") as f:
        json.dump(index_data, f, indent=4)
        
    logger.info(f"Successfully processed {success_count}/{len(cards)} cards.")
    logger.info(f"Index saved to {index_file}")
    return True

if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Download Pokemon Card Set(s)")
    parser.add_argument(
        "--set",
        dest="sets",
        action="append",
        help="Set ID (can be provided multiple times)",
    )
    parser.add_argument(
        "--sets",
        nargs="+",
        dest="sets_list",
        help="Space-separated set IDs",
    )
    parser.add_argument(
        "--output", default="data/card_database", help="Output directory"
    )

    args = parser.parse_args()

    # Merge set inputs and default to base1 if none provided
    merged_sets = (args.sets or []) + (args.sets_list or [])
    if not merged_sets:
        merged_sets = ["base1"]
    unique_sets = list(dict.fromkeys(merged_sets))  # preserve order, dedupe

    api_key = os.getenv("POKEMONTCG_API_KEY") or os.getenv("DEV_POKEMONTCG_IO_API_KEY")
    base_url = os.getenv("POKEMONTCG_BASE_URL") or os.getenv("DEV_POKEMONTCG_IO_BASE_URL") or (
        "https://dev.pokemontcg.io/v2" if api_key else "https://api.pokemontcg.io/v2"
    )
    headers = {"X-Api-Key": api_key} if api_key else {}

    logger.info("Using base URL: %s", base_url)
    if api_key:
        logger.info("API key detected; using authenticated requests")
    else:
        logger.info("No API key detected; using public rate limits")

    for set_id in unique_sets:
        download_set(set_id, args.output, base_url=base_url, headers=headers, timeout=60)
