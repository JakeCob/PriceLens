"""
PriceLens Web API
FastAPI backend for the PriceLens web interface.
"""

import base64
import logging
import shutil
from io import BytesIO
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from src.identification.feature_matcher import FeatureMatcher
from src.api.service import PriceService
from src.overlay.renderer import OverlayRenderer
from src.utils.logging_config import setup_logging

# Setup logging
setup_logging(level="INFO")
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="PriceLens API",
    description="Pokemon Card Price Overlay API",
    version="0.1.0"
)

# CORS (Allow all for local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Components (Lazy loading or startup event recommended, but global for MVP)
logger.info("Initializing PriceLens components...")
matcher = FeatureMatcher()
# Check if features exist
features_path = Path("data/features/base_set_features.pkl")
if features_path.exists():
    matcher.load_database(str(features_path))
else:
    logger.warning(f"Features DB not found at {features_path}. Identification will fail.")

price_service = PriceService()
renderer = OverlayRenderer()


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze uploaded image: Identify -> Price -> Overlay
    """
    try:
        # 1. Read Image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # 2. Identify Card
        matches = matcher.identify(image, top_k=1)
        
        if not matches:
            return JSONResponse(content={
                "success": False,
                "message": "No card identified. Please try a clearer image."
            })
            
        top_match = matches[0]
        card_id = top_match["card_id"]
        
        # 3. Get Price
        price_data = price_service.get_price(card_id)
        
        # 4. Render Overlay
        result_image = renderer.draw_overlay(image, top_match, price_data)
        
        # 5. Encode Result to Base64
        _, buffer = cv2.imencode(".jpg", result_image)
        jpg_as_text = base64.b64encode(buffer).decode("utf-8")
        
        # 6. Prepare Response
        response_data = {
            "success": True,
            "card": {
                "name": top_match["name"],
                "set": top_match["set"],
                "id": card_id,
                "confidence": top_match["confidence"]
            },
            "price": price_data.to_dict() if price_data else None,
            "image": f"data:image/jpeg;base64,{jpg_as_text}"
        }
        
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Serve Static Files (Frontend)
# Mount static files at root for simple access (index.html at /)
app.mount("/", StaticFiles(directory="src/web/static", html=True), name="static")
