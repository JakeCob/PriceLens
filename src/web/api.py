"""
PriceLens Web API
FastAPI backend for the PriceLens web interface.
"""

import base64
import logging
import os
import shutil
import time
import yaml
import asyncio
from datetime import datetime
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
from src.preprocessing.enhancer import ImageEnhancer

# Setup logging
setup_logging(level="INFO")
logger = logging.getLogger(__name__)

# Global instances
price_preloader = None
detection_cache = {}  # tracking_id -> {card_id, name, set, price, confidence, bbox, timestamp}
CACHE_TTL = 5  # seconds - clear cache entries not seen for this long
price_fetch_inflight = set()  # card_id currently being fetched (avoid duplicate background fetches)

# Cache-content fingerprinting:
# The detector/tracker can keep the same tracking id + bbox when you swap the card in the same position.
# Without a content check, we keep showing the previous card until TTL expires.
def _region_fingerprint(region: np.ndarray) -> np.ndarray:
    """
    Create a cheap fingerprint of a card crop to detect changes.
    Returns a small uint8 grayscale thumbnail (16x16) flattened.
    """
    try:
        if region is None or region.size == 0:
            return np.zeros((16 * 16,), dtype=np.uint8)
        thumb = cv2.resize(region, (16, 16), interpolation=cv2.INTER_AREA)
        if len(thumb.shape) == 3:
            thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
        return thumb.reshape(-1).astype(np.uint8)
    except Exception:
        return np.zeros((16 * 16,), dtype=np.uint8)


def _fingerprint_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference (0-255). Higher means more different."""
    try:
        if a is None or b is None or a.size == 0 or b.size == 0:
            return 255.0
        if a.shape != b.shape:
            return 255.0
        return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))
    except Exception:
        return 255.0


async def _maybe_fetch_price_background(card_id: str) -> None:
    """
    Fire-and-forget price fetch. This keeps /detect-live fast:
    - First frame may return price=None
    - Subsequent frames can pick up the cached value
    """
    global price_fetch_inflight, price_preloader
    if not card_id or card_id == "unknown":
        return
    if card_id in price_fetch_inflight:
        return
    price_fetch_inflight.add(card_id)
    try:
        # Keep live path responsive: use a short timeout for on-demand fetches.
        price_data = await asyncio.to_thread(price_service.get_price, card_id, 10.0)
        if price_data and price_preloader:
            # Update preloader cache so live path becomes instant next frames
            price_preloader.cache[card_id] = {
                "price": price_data.market,
                "timestamp": datetime.now(),
                "data": price_data,
            }
    except Exception as e:
        logger.debug(f"Background price fetch failed for {card_id}: {e}")
    finally:
        price_fetch_inflight.discard(card_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global price_preloader
    
    # Startup
    logger.info("Starting up PriceLens API...")
    # Price preloader can be tuned via env:
    # - PRICE_PRELOAD_EAGER=1                 (block startup until initial refresh completes)
    # - PRICE_PRELOAD_STARTUP_DELAY_SECONDS=2 (delay before background refresh)
    # - PRICE_PRELOAD_BATCH_SIZE=10
    # - PRICE_PRELOAD_BATCH_DELAY_SECONDS=0.2
    # Daily refresh by default; loads stale prices instantly on startup.
    refresh_hours = int(os.getenv("PRICE_REFRESH_INTERVAL_HOURS", "24"))
    # Start refresh quickly; can be tuned via env in PricePreloader
    price_preloader = PricePreloader(price_service, refresh_interval_hours=refresh_hours)
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

# Load configuration first (needed for enhancer)
config_path = Path("config.yaml")
config = {}
if config_path.exists():
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {config_path}")
else:
    logger.warning(f"Config file not found at {config_path}, using defaults")

# Initialize Image Enhancer for preprocessing (before matcher and detector)
logger.info("Initializing Image Enhancer...")
image_enhancer = None
preprocessing_config = config.get('preprocessing', {})
if preprocessing_config.get('enabled', True):
    try:
        image_enhancer = ImageEnhancer(preprocessing_config)
        logger.info(f"Image Enhancer initialized (mode={preprocessing_config.get('mode', 'balanced')})")
    except Exception as e:
        logger.warning(f"Failed to initialize Image Enhancer: {e}")
        image_enhancer = None
else:
    logger.info("Image preprocessing disabled in config")

# Initialize Components (Lazy loading or startup event recommended, but global for MVP)
logger.info("Initializing PriceLens components...")
matcher = FeatureMatcher(enhancer=image_enhancer)  # Pass enhancer for identification preprocessing

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

# Initialize YOLO detector for live detection
logger.info("Initializing YOLO detector...")
try:
    detection_config = config.get('detection', {})
    detector = YOLOCardDetector(
        model_path=detection_config.get('model_path', 'models/yolo11n.pt'),
        conf_threshold=detection_config.get('confidence_threshold', 0.45),  # Lowered from 0.65 - cards often detected at 0.5-0.6
        iou_threshold=detection_config.get('iou_threshold', 0.45),
        use_card_specific_model=detection_config.get('use_card_specific_model', False),
        enhancer=image_enhancer  # Pass enhancer for pre-detection preprocessing
    )
    logger.info(f"YOLO detector initialized successfully (model: {detection_config.get('model_path', 'models/yolo11n.pt')})")
except Exception as e:
    logger.error(f"Failed to initialize YOLO detector: {e}")
    detector = None

def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0
        
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0

def find_spatial_match(bbox, cache, threshold=0.6):
    """
    Find a cached result that spatially matches the current bbox.
    Prefer confirmed (non-scanning) matches and highest IoU.
    """
    best = None
    best_iou = 0.0
    for cached in cache.values():
        iou = compute_iou(bbox, cached["bbox"])
        if iou <= threshold:
            continue
        if iou > best_iou:
            best = cached
            best_iou = iou
        # If IoU ties, prefer confirmed over scanning
        elif iou == best_iou and best is not None:
            if best.get("is_scanning_state", False) and not cached.get("is_scanning_state", False):
                best = cached
                best_iou = iou
    return best

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
                # Check cache first (Direct Match)
                cached = None
                if det.card_id in detection_cache:
                    cached = detection_cache[det.card_id]
                else:
                    # Fallback: Spatial Match (Robust to tracker ID changes)
                    cached = find_spatial_match(det.bbox, detection_cache)
                    if cached:
                         logger.debug(f"Found spatial match for {det.card_id} -> Reusing {cached['card_id']}")

                if cached:
                    # Determine appropriate TTL
                    is_scanning = cached.get("is_scanning_state", False)
                    # Use shorter TTL (2.0s) for "Scanning..." to allow re-trying
                    # Use longer TTL (5.0s) for confirmed matches
                    current_ttl = 2.0 if is_scanning else CACHE_TTL
                    
                    # Check if cache is still valid
                    if time.time() - cached["timestamp"] < current_ttl:
                        # If the content inside the bbox has changed (new card swapped in same place),
                        # bypass the cache and re-identify immediately.
                        # This reduces the "stuck on last card" latency.
                        new_fp = _region_fingerprint(region)
                        old_fp = cached.get("fingerprint")
                        fp_delta = _fingerprint_distance(new_fp, old_fp) if old_fp is not None else 255.0
                        # Threshold tuned empirically; cards swapped in-place tend to be >> 20.
                        if not is_scanning and fp_delta > 22.0:
                            logger.info(
                                f"Cache bypass: content changed for {det.card_id} "
                                f"(prev={cached.get('card_id')}, delta={fp_delta:.1f})"
                            )
                            # Explicitly bypass cached handling below (and avoid grace reuse)
                            cached = None
                        else:
                            logger.debug(f"Using cached identification for {det.card_id} (scanning={is_scanning})")
                            # IMPORTANT:
                            # - For confirmed matches, refresh timestamp so the cache doesn't expire while the card is
                            #   still on-screen (prevents flicker back to Scanning/Unknown after ~CACHE_TTL seconds).
                            # - For "Scanning..." state, keep the original timestamp so it naturally expires quickly
                            #   and triggers a re-identification attempt.
                            cached["bbox"] = det.bbox
                            if not is_scanning:
                                cached["timestamp"] = current_time
                                # If the first identification happened before the price was available,
                                # refresh the cached price from the preloader so the UI updates without
                                # needing a re-identification.
                                cid = cached.get("card_id")
                                if cid and cid != "unknown":
                                    if price_preloader:
                                        latest_price = price_preloader.get_price(cid)
                                        if latest_price is None:
                                            asyncio.create_task(_maybe_fetch_price_background(cid))
                                        else:
                                            cached["price"] = latest_price
                                # Refresh fingerprint for confirmed cached hits
                                cached["fingerprint"] = new_fp
                            # Also key by the current tracker id so we get direct hits on subsequent frames
                            detection_cache[det.card_id] = cached
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
                # If we had a recent confirmed match, hold it through brief dips in confidence.
                confirmed_grace_s = 3.0
                last_confirmed = None
                if cached and not cached.get("is_scanning_state", False) and cached.get("card_id") != "unknown":
                    last_confirmed = cached
                elif cached and cached.get("card_id") != "unknown" and not cached.get("is_scanning_state", False):
                    last_confirmed = cached
                
                if matches:
                    top = matches[0]
                    
                    # THRESHOLD: OCR matches get 0.80, ORB fallback needs >= 0.50
                    # This prevents false positives from ORB while allowing accurate OCR
                    if top["confidence"] < 0.50:
                         logger.info(f"Rejected low confidence match: {top['name']} ({top['confidence']:.2f})")
                         # If we recently had a confirmed ID for this tracked card, reuse it instead of
                         # downgrading to "Scanning..." (prevents UI flicker to Unknown).
                         # But do NOT reuse the last confirmed if the content clearly changed (new card swapped).
                         new_fp = _region_fingerprint(region)
                         old_fp = last_confirmed.get("fingerprint") if last_confirmed else None
                         fp_delta = _fingerprint_distance(new_fp, old_fp) if old_fp is not None else 255.0
                         if last_confirmed and fp_delta <= 22.0 and (time.time() - last_confirmed["timestamp"] < (CACHE_TTL + confirmed_grace_s)):
                             last_confirmed["timestamp"] = current_time
                             last_confirmed["bbox"] = det.bbox
                             last_confirmed["fingerprint"] = new_fp
                             detection_cache[det.card_id] = last_confirmed
                             results.append({
                                 "bbox": det.bbox,
                                 "name": last_confirmed["name"],
                                 "card_id": last_confirmed["card_id"],
                                 "set": last_confirmed["set"],
                                 "confidence": last_confirmed["confidence"],
                                 "price": last_confirmed["price"],
                             })
                             continue
                         
                         # CACHE THE FAILURE (Short TTL) to prevent re-running ORB every frame
                         # This fixes the 2 FPS lag when scanning unsupported cards
                         detection_cache[det.card_id] = {
                            "name": "Scanning...",
                            "card_id": "unknown",
                            "set": "...",
                            "confidence": top["confidence"],
                            "price": None,
                            "bbox": det.bbox,
                            "timestamp": current_time,
                            "fingerprint": new_fp,
                            "is_scanning_state": True # Marker for short TTL
                         }
                         
                         results.append({
                            "bbox": det.bbox,
                            "name": "Scanning...",
                            "card_id": "unknown",
                            "set": "...",
                            "confidence": top["confidence"],
                            "price": None
                        })
                         continue

                    card_id = top["card_id"]
                    
                    # Get price from preloaded cache (instant!)
                    cached_price = None
                    if price_preloader:
                        cached_price = price_preloader.get_price(card_id)
                    # If missing, trigger a background fetch so it can populate on subsequent frames
                    if cached_price is None:
                        asyncio.create_task(_maybe_fetch_price_background(card_id))
                    
                    # Update cache with new identification
                    new_fp = _region_fingerprint(region)
                    detection_cache[det.card_id] = {
                        "name": top["name"],
                        "card_id": card_id,
                        "set": top.get("set", "Unknown"),
                        "confidence": top["confidence"],
                        "price": cached_price,
                        "bbox": det.bbox,
                        "timestamp": current_time,
                        "fingerprint": new_fp,
                        "is_scanning_state": False
                    }
                    logger.info(f"Cached new detection: {top['name']} (card_id={det.card_id}, conf={top['confidence']:.2f})")
                    
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
                    # Same stability rule: if we have a recent confirmed match, keep it.
                    new_fp = _region_fingerprint(region)
                    old_fp = last_confirmed.get("fingerprint") if last_confirmed else None
                    fp_delta = _fingerprint_distance(new_fp, old_fp) if old_fp is not None else 255.0
                    if last_confirmed and fp_delta <= 22.0 and (time.time() - last_confirmed["timestamp"] < (CACHE_TTL + confirmed_grace_s)):
                        last_confirmed["timestamp"] = current_time
                        last_confirmed["bbox"] = det.bbox
                        last_confirmed["fingerprint"] = new_fp
                        detection_cache[det.card_id] = last_confirmed
                        results.append({
                            "bbox": det.bbox,
                            "name": last_confirmed["name"],
                            "card_id": last_confirmed["card_id"],
                            "set": last_confirmed["set"],
                            "confidence": last_confirmed["confidence"],
                            "price": last_confirmed["price"],
                        })
                        continue
                    
                    # Cache the failure too
                    detection_cache[det.card_id] = {
                        "name": "Scanning...",
                        "card_id": "unknown",
                        "set": "...",
                        "confidence": 0.0,
                        "price": None,
                        "bbox": det.bbox,
                        "timestamp": current_time,
                        "fingerprint": new_fp,
                        "is_scanning_state": True
                    }
                    
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
