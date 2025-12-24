#!/usr/bin/env python3
"""
Download sample Pokemon card images for testing
"""

import requests
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_sample_cards():
    """Download sample Pokemon card images from Pokemon TCG API"""
    print("Downloading sample Pokemon card images...")

    # Use Pokemon TCG API to get card images
    api_url = "https://api.pokemontcg.io/v2/cards"

    # Get some popular cards
    params = {
        "q": "set.id:base1",  # Base Set
        "pageSize": 5,
        "orderBy": "number",
    }

    try:
        response = requests.get(api_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        cards = data.get("data", [])
        output_dir = Path("test_images")
        output_dir.mkdir(exist_ok=True)

        print(f"\nDownloading {len(cards)} sample cards...\n")

        for card in cards:
            card_id = card["id"]
            card_name = card["name"]
            image_url = card["images"]["large"]

            # Download image
            img_response = requests.get(image_url, timeout=30)
            img_response.raise_for_status()

            # Save image
            safe_name = "".join(c if c.isalnum() else "_" for c in card_name)
            filename = output_dir / f"{safe_name}_{card_id}.jpg"

            with open(filename, "wb") as f:
                f.write(img_response.content)

            print(f"✓ Downloaded: {card_name} ({card_id})")

        print(f"\n✅ Downloaded {len(cards)} cards to test_images/")
        print("\nRun: python scripts/test_detector_images.py")

        return True

    except Exception as e:
        print(f"❌ Error downloading cards: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = download_sample_cards()
    sys.exit(0 if success else 1)
