"""
Remove ME1 and ME2 entries from custom_prices.json since the API supports them.
"""
import json
from pathlib import Path

def cleanup_custom_prices():
    file_path = Path("data/custom_prices.json")
    if not file_path.exists():
        print("File not found.")
        return

    with open(file_path, "r") as f:
        prices = json.load(f)
    
    print(f"Total entries before: {len(prices)}")
    
    # Filter out me1 and me2
    new_prices = {
        k: v for k, v in prices.items() 
        if not k.startswith("me1-") and not k.startswith("me2-")
    }
    
    print(f"Total entries after: {len(new_prices)}")
    
    with open(file_path, "w") as f:
        json.dump(new_prices, f, indent=2)
        
    print("Cleanup complete.")

if __name__ == "__main__":
    cleanup_custom_prices()
