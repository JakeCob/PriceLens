# Live Camera Feature - Implementation Walkthrough

## Overview
We have successfully implemented the **Live Camera Scanning** feature for PriceLens. This allows users to point their camera at Pokemon cards and get instant identification and pricing overlays.

## Features Implemented

### 1. Live Camera UI
- **Toggle Mode**: Switch between "Upload Photo" and "Live Camera".
- **Real-time Overlay**: Bounding boxes and labels drawn on a canvas over the video feed.
- **Controls**: Start/Stop camera, Capture frame.
- **Responsive**: Works on desktop and mobile (uses rear camera).

### 2. High-Performance Backend
- **`/detect-live` Endpoint**: Optimized for speed (returns JSON, no server-side rendering).
- **YOLO Integration**: Uses the YOLO11n model for fast object detection.
- **Multi-Card Support**: Detects and identifies multiple cards simultaneously.

### 3. Intelligent Price Loading
- **Startup Preloader**: Loads prices for all ~420 known cards in the background on startup.
- **Instant Pricing**: Prices are served from memory cache (0ms latency).
- **Auto-Refresh**: Prices automatically refresh every hour.
- **Manual Refresh**: API endpoint `/refresh-prices` available.

## How to Run

### Start the Server
```bash
python run_web.py
```
*Note: If you get "Address already in use", run `pkill -f "python run_web.py"` first.*

### Usage
1. Open **http://localhost:8080**
2. Click **"📹 Live Camera"**
3. Click **"Start Camera"**
4. Point at cards!

*Note: On first startup, prices may show as "Fetching..." for ~10 minutes while the background loader runs. After that, they are instant.*

## Future Improvements
See `docs/LIVE_CAMERA_TODO.md` for a detailed list of planned improvements, including:
- Reducing false positives (faces/objects)
- Improving low-light detection
- Adding "Best Lighting" user guidance

## Files Changed
- `src/web/static/index.html`: Added camera UI
- `src/web/static/camera.js`: Camera logic & rendering
- `src/web/api.py`: Backend endpoints & lifecycle management
- `src/web/price_preloader.py`: Background price fetching system
- `src/web/static/style.css`: Camera styles
