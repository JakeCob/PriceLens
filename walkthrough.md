# Walkthrough: Enhanced Features Implementation

I have implemented the missing "Enhanced" features from the V2 plan.

## 1. New Dependencies
You need to install the new dependencies to use these features:
```bash
pip install -r requirements.txt
```
New packages: `easyocr`, `chromadb`, `filterpy`.

## 2. Implemented Features

### Phase 1.3: Frame Interpolation
- **File**: `src/detection/frame_interpolator.py`
- **Usage**: Uses Kalman Filters to predict card positions between detection frames, enabling smoother tracking and higher effective FPS.

### Phase 3.1: Smart Cache
- **File**: `src/api/smart_cache.py`
- **Usage**: A two-level cache (Memory + SQLite).
- **Integration**: `PriceService` now uses `SmartCache` automatically. It persists price data to `data/cache/cache.db`.

### Phase 2.2: Enhanced Identification
- **File**: `src/identification/feature_matcher.py`
- **Updates**:
    - Added `EasyOCR` support for reading text when visual matching confidence is low (<0.4).
    - Added `ChromaDB` initialization (stubbed for now) for future vector search.
    - **Note**: These features gracefully degrade if libraries are missing.

### Phase 4: Core Architecture
- **Files**: `src/core/event_bus.py`, `src/core/plugin_manager.py`
- **Usage**:
    - `EventBus`: Decouples components. You can now `subscribe` to events like `card.detected`.
    - `PluginManager`: Allows loading external plugins to extend functionality.

## 3. Verification
I created a script `scripts/verify_missing_features.py` to test these components. Run it after installing dependencies:
```bash
python scripts/verify_missing_features.py
```

## Next Steps
- Run `pip install -r requirements.txt`
- Run the verification script.
- Start the web app (`python run_web.py`) and verify that price fetching and detection still work (now with caching and interpolation available).
