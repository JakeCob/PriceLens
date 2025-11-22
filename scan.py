#!/usr/bin/env python3
"""
PriceLens CLI Scanner
Simple wrapper to scan a card image from the command line.
"""

import sys
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="PriceLens Card Scanner")
    parser.add_argument("image", help="Path to the card image file")
    parser.add_argument("--output", "-o", default="output/result.jpg", help="Path to save the result")
    
    args = parser.parse_args()
    
    # Check if image exists
    if not Path(args.image).exists():
        print(f"❌ Error: Image not found at {args.image}")
        sys.exit(1)
        
    print(f"🔍 Scanning {args.image}...")
    
    # Run the demo script
    cmd = [
        sys.executable, 
        "scripts/demo_single_image.py",
        "--image", args.image,
        "--output", args.output
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"\n✅ Scan Complete!")
        print(f"🖼️  Result saved to: {args.output}")
        print(f"   (You can download this file to view it)")
    else:
        print("\n❌ Scan Failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
