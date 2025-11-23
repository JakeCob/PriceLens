#!/usr/bin/env python3
"""
Test Web API
Sends a request to the running API to verify it works.
"""

import requests
import sys
import time
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_api():
    # 1. Start the server in a subprocess
    logger.info("Starting server...")
    server_process = subprocess.Popen(
        [sys.executable, "run_web.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to start
    time.sleep(5)
    
    try:
        # 2. Prepare test image
        image_path = "data/card_database/base1/Charizard_base1-4.jpg"
        if not Path(image_path).exists():
            logger.error(f"Test image not found: {image_path}")
            return False

        # 3. Send Request
        url = "http://localhost:8000/analyze"
        logger.info(f"Sending POST request to {url}...")
        
        with open(image_path, "rb") as f:
            files = {"file": f}
            response = requests.post(url, files=files)
            
        # 4. Verify Response
        if response.status_code == 200:
            data = response.json()
            if data["success"]:
                logger.info("✅ API Test Passed!")
                logger.info(f"Card: {data['card']['name']}")
                if data['price']:
                    logger.info(f"Price: ${data['price']['market']}")
                else:
                    logger.warning("Price: N/A (API failed or card not found)")
                return True
            else:
                logger.error(f"API returned success=False: {data}")
        else:
            logger.error(f"API failed with status {response.status_code}: {response.text}")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
    finally:
        # 5. Cleanup
        logger.info("Stopping server...")
        server_process.terminate()
        server_process.wait()

    return False

if __name__ == "__main__":
    success = test_api()
    sys.exit(0 if success else 1)
