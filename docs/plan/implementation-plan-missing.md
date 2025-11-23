# Implementation Plan: Missing Features

This plan addresses the skipped "Enhanced" features from the original V2 plan.

## User Review Required
> [!IMPORTANT]
> This plan introduces new heavy dependencies: `easyocr` (PyTorch based), `chromadb` (Vector DB), and `filterpy`. Ensure your environment has sufficient disk space and memory.

## Proposed Changes

### 1. Dependencies
#### [MODIFY] [pyproject.toml](file:///root/Programming%20Projects/Personal/PriceLens/pyproject.toml)
- Add `easyocr`, `chromadb`, `filterpy` to dependencies.

### 2. Phase 1.3: Frame Interpolation
#### [NEW] [src/detection/frame_interpolator.py](file:///root/Programming%20Projects/Personal/PriceLens/src/detection/frame_interpolator.py)
- Implement `FrameInterpolator` class using Kalman Filters (`filterpy`).
- Methods: `predict()`, `update()`.

### 3. Phase 3.1: Smart Cache
#### [NEW] [src/api/smart_cache.py](file:///root/Programming%20Projects/Personal/PriceLens/src/api/smart_cache.py)
- Implement `SmartCache` with SQLite backend and LRU memory cache.
- Replaces simple pickle cache.

#### [MODIFY] [src/api/service.py](file:///root/Programming%20Projects/Personal/PriceLens/src/api/service.py)
- Integrate `SmartCache` into `PriceService`.

### 4. Phase 2.2: Enhanced Identification
#### [MODIFY] [src/identification/feature_matcher.py](file:///root/Programming%20Projects/Personal/PriceLens/src/identification/feature_matcher.py)
- Add `EasyOCR` fallback for text reading.
- Add `ChromaDB` integration for vector similarity search (optional/stubbed if DB not ready).

### 5. Phase 4: Core Architecture
#### [NEW] [src/core/event_bus.py](file:///root/Programming%20Projects/Personal/PriceLens/src/core/event_bus.py)
- Implement `EventBus` for pub/sub communication.

#### [NEW] [src/core/plugin_manager.py](file:///root/Programming%20Projects/Personal/PriceLens/src/core/plugin_manager.py)
- Implement `PluginManager` and base `Plugin` class.

## Verification Plan

### Automated Tests
- Run `pytest` to ensure no regressions.
- Create specific tests for `FrameInterpolator` and `SmartCache`.

### Manual Verification
- **Interpolation**: Run detection on video sample, verify smooth tracking.
- **Cache**: Check SQLite file creation and persistence after restart.
- **OCR**: Test with a card image that fails visual matching but has clear text.
