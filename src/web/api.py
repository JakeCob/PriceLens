"""
PriceLens Web API
FastAPI backend for the PriceLens web interface.
"""

import base64
import logging
import shutil
import time
import yaml
from io import BytesIO
from pathlib import Path
from typing import Dict
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from src.identification.feature_matcher import FeatureMatcher
from src.api.service import PriceService
from src.overlay.renderer import OverlayRenderer
from src.detection.yolo_detector import YOLOCardDetector
from src.core.event_bus import event_bus
from src.web.price_preloader import PricePreloader
from src.utils.logging_config import setup_logging

# Setup logging
setup_logging(level="INFO")
logger = logging.getLogger(__name__)

# Global instances
price_preloader = None
detection_cache = {}  # tracking_id -> {card_id, name, set, price, confidence, bbox, timestamp}
CACHE_TTL = 5  # seconds - clear cache entries not seen for this long


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global price_preloader
    
    # Startup
    logger.info("Starting up PriceLens API...")
    price_preloader = PricePreloader(price_service, refresh_interval_hours=1)
    await price_preloader.start()
    logger.info("Price preloader started (will refresh hourly)")
    
    yield
    
    # Shutdown
    logger.info("Shutting down PriceLens API...")
    if price_preloader:
        await price_preloader.stop()


# Initialize FastAPI with lifespan
app = FastAPI(
    title="PriceLens API",
    description="Pokemon Card Price Overlay API",
    version="0.1.0",
    lifespan=lifespan
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

# Load me1 and me2 feature database (318 cards)
me1_me2_features = Path("data/me1_me2_features.pkl")
if me1_me2_features.exists():
    try:
        matcher.load_database(str(me1_me2_features))
        logger.info(f"Loaded {len(matcher.card_features)} cards from me1_me2_features.pkl")
    except Exception as e:
        logger.error(f"Failed to load me1_me2_features.pkl: {e}")
else:
    logger.warning("me1_me2_features.pkl not found. Run scripts/build_me1_me2_features.py first.")

price_service = PriceService()
renderer = OverlayRenderer()

# Load configuration
config_path = Path("config.yaml")
config = {}
if config_path.exists():
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {config_path}")
else:
    logger.warning(f"Config file not found at {config_path}, using defaults")

# Initialize YOLO detector for live detection
logger.info("Initializing YOLO detector...")
try:
    detection_config = config.get('detection', {})
    detector = YOLOCardDetector(
        model_path=detection_config.get('model_path', 'models/yolo11m.pt'),
        conf_threshold=detection_config.get('confidence_threshold', 0.65),
        iou_threshold=detection_config.get('iou_threshold', 0.45),
        use_card_specific_model=detection_config.get('use_card_specific_model', False)
    )
    logger.info(f"YOLO detector initialized successfully (model: {detection_config.get('model_path', 'models/yolo11m.pt')})")
except Exception as e:
    logger.error(f"Failed to initialize YOLO detector: {e}")
    detector = None


@app.post("/detect-live")
async def detect_live(file: UploadFile = File(...)):
    """
    Live detection endpoint - optimized for speed
    Returns bounding boxes + card names without rendering
    Used by camera mode for real-time detection
    """
    try:
        # 1. Read Image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        # 2. Check if detector is available
        if detector is None:
            return JSONResponse(content={
                "success": False,
                "detections": [],
                "message": "Detector not initialized"
            })
        
        logger.warning(f"=== API /detect-live called ===")
        logger.warning(f"Frame received: {type(image)}, shape: {image.shape if hasattr(image, 'shape') else 'N/A'}")

        # 3. Detect cards using YOLO
        detections = detector.detect(image)
        logger.warning(f"Detector returned {len(detections)} detections")
        tracked = detector.track_cards(detections)
        
        # 4. Extract and identify each card region
        results = []
        current_time = time.time()
        
        if tracked:
            card_regions = detector.extract_card_regions(image, tracked)
            
            for det, region in zip(tracked, card_regions):
                # Check cache first
                if det.card_id in detection_cache:
                    cached = detection_cache[det.card_id]
                    # Check if cache is still valid
                    if time.time() - cached["timestamp"] < CACHE_TTL:
                        logger.debug(f"Using cached identification for {det.card_id}")
                        results.append({
                            "bbox": det.bbox,
                            "name": cached["name"],
                            "card_id": cached["card_id"],
                            "set": cached["set"],
                            "confidence": cached["confidence"],
                            "price": cached["price"]
                        })
                        continue
                
                # Quick identification (ORB only)
                matches = matcher.identify(region, top_k=1)
                
                if matches:
                    top = matches[0]
                    card_id = top["card_id"]
                    
                    # Get price from preloaded cache (instant!)
                    cached_price = None
                    if price_preloader:
                        cached_price = price_preloader.get_price(card_id)
                    
                    # Update cache with new identification
                    detection_cache[det.card_id] = {
                        "name": top["name"],
                        "card_id": card_id,
                        "set": top.get("set", "Unknown"),
                        "confidence": top["confidence"],
                        "price": cached_price,
                        "bbox": det.bbox,
                        "timestamp": current_time
                    }
                    logger.info(f"Cached new detection: {top['name']} (card_id={det.card_id})")
                    
                    results.append({
                        "bbox": det.bbox,
                        "name": top["name"],
                        "card_id": card_id,
                        "set": top.get("set", "Unknown"),
                        "confidence": top["confidence"],
                        "price": cached_price  # Use real price from API/cache
                    })
                else:
                    # Identification failed, but we still want to show the detection box
                    logger.debug(f"Identification failed for {det.card_id}, returning 'Scanning...'")
                    results.append({
                        "bbox": det.bbox,
                        "name": "Scanning...",
                        "card_id": "unknown",
                        "set": "...",
                        "confidence": 0.0,
                        "price": None
                    })
        
        # Clean up stale cache entries (not seen in >CACHE_TTL seconds)
        stale_ids = [tid for tid, data in detection_cache.items() 
                     if current_time - data['timestamp'] > CACHE_TTL]
        for tid in stale_ids:
            logger.debug(f"Removing stale cache entry: {detection_cache[tid]['name']} (tracking_id={tid})")
            del detection_cache[tid]
        
        return JSONResponse(content={
            "success": True,
            "detections": results
        })

    except Exception as e:
        logger.error(f"Live detection error: {e}")
        return JSONResponse(content={
            "success": False,
            "detections": [],
            "message": str(e)
        })


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
        
        # Emit event
        event_bus.emit("card.identified", {
            "card_id": card_id,
            "name": top_match["name"],
            "price": price_data.market if price_data else None,
            "confidence": top_match["confidence"]
        })
        
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/refresh-prices")
async def refresh_prices():
    """
    Manually trigger price refresh
    Returns immediately, refresh happens in background
    """
    if not price_preloader:
        raise HTTPException(status_code=503, detail="Price preloader not initialized")
    
    if price_preloader.is_refreshing:
        return JSONResponse(content={
            "success": False,
            "message": "Refresh already in progress"
        })
    
    # Start refresh in background
    asyncio.create_task(price_preloader.refresh_all_prices())
    
    return JSONResponse(content={
        "success": True,
        "message": "Price refresh started"
    })


@app.get("/price-status")
async def price_status():
    """Get price cache status and progress"""
    if not price_preloader:
        return JSONResponse(content={"error": "Preloader not initialized"})
    
    stats = price_preloader.get_cache_stats()
    progress = price_preloader.get_progress()
    
    return JSONResponse(content={
        "stats": stats,
        "progress": progress
    })


# Serve Static Files (Frontend)
# Mount static files at root for simple access (index.html at /)
app.mount("/", StaticFiles(directory="src/web/static", html=True), name="static")
