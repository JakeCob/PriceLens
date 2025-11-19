# PriceLens Implementation Plan
**Generated:** November 19, 2025  
**Status:** Project is ~10-15% complete (foundation only)  
**Target:** Functional MVP with card detection, identification, and price overlay

---

## Executive Summary

PriceLens currently has excellent documentation (1,888-line CLAUDE.md guide) and solid architecture, but minimal actual implementation. This plan provides a phased approach to reach a working MVP within 6-8 weeks.

### Current State Analysis
- ✅ **Complete:** Project structure, config system, base classes, documentation, Docker setup
- ⚠️ **Incomplete:** All core functionality (detection, identification, API, rendering)
- ❌ **Missing:** Models, card database, actual implementation files

### Key Challenges
1. **Feature matching complexity** - Need high-quality reference images and careful parameter tuning
2. **Real-time performance** - Must maintain 30 FPS with GPU acceleration
3. **API costs** - Price APIs may require paid subscriptions
4. **Database size** - Full Pokemon card database requires ~5GB storage

---

## Phase 1: Environment Setup & Basic Detection (Week 1)

### Milestone: See a card detected on webcam with bounding box

#### 1.1 Development Environment Setup
**Priority:** Critical  
**Estimated Time:** 30 minutes

**Tasks:**
- [ ] Create virtual environment using `uv venv`
- [ ] Install dependencies: `uv pip install -r requirements.txt`
- [ ] Install dev dependencies: `uv pip install -r requirements-dev.txt`
- [ ] Verify all imports work (OpenCV, PyTorch, Ultralytics)
- [ ] Test GPU availability with `torch.cuda.is_available()`

**Verification:**
```bash
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from ultralytics import YOLO; print('YOLO OK')"
```

**Blockers:**
- CUDA/GPU drivers may need installation
- WSL might need X11 forwarding for camera access

---

#### 1.2 Download YOLO11 Model
**Priority:** Critical  
**Estimated Time:** 10 minutes

**Tasks:**
- [ ] Run `python scripts/download_models.py`
- [ ] Verify model saved to `models/yolo11n.pt` (~6MB)
- [ ] Test model loads successfully
- [ ] Create `models/README.md` documentation

**Verification:**
```python
from ultralytics import YOLO
model = YOLO('models/yolo11n.pt')
print(f"Model loaded: {model}")
```

**Notes:**
- YOLO11n (nano) chosen for speed over accuracy
- Can upgrade to YOLO11s/m later if needed

---

#### 1.3 Implement YOLOCardDetector
**Priority:** Critical  
**Estimated Time:** 4 hours

**File:** `src/detection/yolo_detector.py`

**Requirements:**
- Inherit from `DetectorBase` abstract class
- Implement `detect(frame)` method returning list of detections
- Implement `extract_card_regions(frame, detections)` method
- Filter detections by aspect ratio (0.5-0.9 for Pokemon cards)
- Add confidence threshold filtering
- Support GPU/CPU detection

**Key Methods:**
```python
class YOLOCardDetector(DetectorBase):
    def __init__(self, model_path, conf_threshold=0.5, iou_threshold=0.45)
    def detect(self, frame: np.ndarray) -> List[dict]
    def extract_card_regions(self, frame, detections) -> List[np.ndarray]
    def _validate_card_shape(self, bbox) -> bool  # Aspect ratio check
```

**Testing Strategy:**
- Unit tests with synthetic images
- Integration test with webcam
- Test with physical Pokemon cards

**Success Criteria:**
- Detects rectangular objects with >95% accuracy
- Runs at >30 FPS on GPU
- False positive rate <10%

---

#### 1.4 Create Basic Camera Test Script
**Priority:** Critical  
**Estimated Time:** 1 hour

**File:** `scripts/test_detection.py`

**Purpose:** Verify webcam + YOLO detection works end-to-end

**Script Structure:**
```python
#!/usr/bin/env python3
"""Test script for card detection"""
import cv2
from src.detection.yolo_detector import YOLOCardDetector

def main():
    detector = YOLOCardDetector("models/yolo11n.pt")
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        detections = detector.detect(frame)
        
        # Draw bounding boxes
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.imshow('Detection Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
```

**Deliverable:** Working demo showing real-time card detection

---

## Phase 2: Card Database & Feature Matching (Week 2-3)

### Milestone: Identify specific Pokemon cards by name

#### 2.1 Build Card Database Script
**Priority:** High  
**Estimated Time:** 3 hours

**File:** `scripts/build_card_database.py`

**Requirements:**
- Connect to Pokemon TCG API (https://api.pokemontcg.io/v2)
- Download card images for specified sets
- Save metadata (card_id, name, set, rarity) to JSON
- Organize by set: `data/card_database/{set_id}/{card_id}.jpg`
- Create master index: `data/card_database/index.json`

**Initial Target Sets:**
- Base Set (102 cards)
- Jungle (64 cards)
- Fossil (62 cards)
- **Total: ~230 cards, ~150MB**

**API Endpoints:**
```
GET https://api.pokemontcg.io/v2/cards?q=set.id:base1
GET https://api.pokemontcg.io/v2/cards/{id}
```

**Rate Limiting:** 1000 requests/hour (no API key needed)

**Script Usage:**
```bash
python scripts/build_card_database.py --sets base1,jungle,fossil
python scripts/build_card_database.py --all  # Download all sets (5GB!)
```

---

#### 2.2 Pre-compute Card Features
**Priority:** High  
**Estimated Time:** 2 hours

**File:** `scripts/compute_features.py`

**Purpose:** Pre-compute ORB+BEBLID features for all database cards

**Process:**
1. Load all card images from database
2. Compute ORB keypoints (1000 features per card)
3. Compute BEBLID descriptors
4. Save to pickle file: `data/features/card_features.pkl`

**Data Structure:**
```python
{
    'features': {
        'base1-4': np.ndarray,  # Charizard descriptors
        'base1-58': np.ndarray, # Pikachu descriptors
        ...
    },
    'metadata': {
        'base1-4': {'name': 'Charizard', 'set_name': 'Base Set', ...},
        ...
    }
}
```

**Estimated Size:** ~50MB for 230 cards

**Performance:** Should take <5 minutes to process 230 cards

---

#### 2.3 Implement FeatureMatcher
**Priority:** High  
**Estimated Time:** 6 hours

**File:** `src/identification/feature_matcher.py`

**Requirements:**
- Use ORB for keypoint detection (fast, license-free)
- Use BEBLID descriptors (14% better than ORB)
- FLANN-based matcher for speed
- Lowe's ratio test (threshold: 0.75)
- Homography verification for geometric consistency
- Return top-K matches with confidence scores

**Key Methods:**
```python
class FeatureMatcher(IdentifierBase):
    def __init__(self, n_features=1000, match_threshold=0.75, min_matches=10)
    def load_database(self, database_path)
    def compute_features(self, image) -> Tuple[keypoints, descriptors]
    def match_card(self, image, top_k=3) -> List[Dict]
    def _apply_ratio_test(self, matches) -> List
    def _verify_homography(self, matches) -> bool
```

**Matching Pipeline:**
```
Input Image → Grayscale → ORB Keypoints → BEBLID Descriptors
              ↓
Compare with Database → FLANN Matching → Lowe's Ratio Test
              ↓
Filter by Min Matches → Homography Check → Rank by Confidence
              ↓
Return Top-K Matches
```

**Performance Target:**
- Match time: <50ms per card
- Accuracy: >90% for front-facing cards
- False positive rate: <5%

**Edge Cases to Handle:**
- No features detected (blurry/dark images)
- Multiple similar cards (different editions)
- Holographic vs non-holographic versions
- Rotated/tilted cards

---

#### 2.4 Test Feature Matching Accuracy
**Priority:** High  
**Estimated Time:** 2 hours

**File:** `tests/test_identification.py`

**Test Strategy:**
1. **Unit Tests:**
   - Test feature computation
   - Test ratio test filtering
   - Test homography verification

2. **Integration Tests:**
   - Test with known cards from database
   - Test with cards not in database (should return no match)
   - Test with rotated/scaled cards

3. **Accuracy Benchmark:**
   - Create test set of 20 cards with multiple photos each
   - Measure accuracy: correct_matches / total_tests
   - Target: >90% accuracy

**Benchmark Script:**
```python
def benchmark_accuracy(test_images_dir, ground_truth_labels):
    matcher = FeatureMatcher()
    matcher.load_database('data/features/card_features.pkl')
    
    correct = 0
    total = 0
    
    for image_path, true_label in test_images_dir.items():
        matches = matcher.match_card(cv2.imread(image_path), top_k=1)
        if matches and matches[0]['card_id'] == true_label:
            correct += 1
        total += 1
    
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.2%}")
```

---

## Phase 3: Price API Integration (Week 4)

### Milestone: Display real-time prices for identified cards

#### 3.1 Research Price API Options
**Priority:** Medium  
**Estimated Time:** 2 hours

**Options:**

1. **PokemonPriceTracker.com API**
   - Pros: Comprehensive price data (TCGPlayer, eBay, CardMarket)
   - Cons: Requires API key, likely paid subscription
   - Rate limit: Unknown

2. **Pokemon TCG API (Fallback)**
   - Pros: Free, no API key needed
   - Cons: No price data (only card info)
   - Rate limit: 1000 req/hour

3. **TCGPlayer API**
   - Pros: Direct source of truth
   - Cons: Requires approved developer account
   - Rate limit: Unknown

4. **Scraping Alternative (Last Resort)**
   - Use BeautifulSoup to scrape TCGPlayer prices
   - Not recommended (TOS violation, fragile)

**Decision:** Start with Pokemon TCG API for card info, add price API later

---

#### 3.2 Implement Price API Client
**Priority:** Medium  
**Estimated Time:** 4 hours

**File:** `src/api/price_api.py`

**Requirements:**
- Create `PriceCache` class with TTL (5-minute default)
- Create `RateLimiter` class (60 req/min)
- Create `PokemonPriceAPI` class (primary)
- Create `PokemonTCGAPI` class (fallback)
- Handle API errors gracefully
- Support batch requests

**Key Classes:**
```python
class PriceCache:
    def __init__(self, ttl=300)
    def get(self, key) -> Optional[Dict]
    def set(self, key, value)
    def clear()

class RateLimiter:
    def __init__(self, rate_limit=60)
    def wait()  # Blocks if rate limit exceeded

class PokemonPriceAPI:
    def __init__(self, api_key, cache_ttl=300)
    def get_card_price(self, card_id) -> Optional[Dict]
    def get_batch_prices(self, card_ids) -> Dict[str, Dict]

class PokemonTCGAPI:  # Fallback (no prices)
    def get_card_info(self, card_id) -> Optional[Dict]
```

**Response Format:**
```python
{
    'card_id': 'base1-4',
    'name': 'Charizard',
    'set': 'Base Set',
    'prices': {
        'tcgplayer_market': 125.50,
        'tcgplayer_low': 100.00,
        'tcgplayer_high': 175.00
    },
    'graded_prices': {
        'psa10': 450.00,
        'psa9': 275.00
    },
    'last_updated': '2025-11-19T22:35:00'
}
```

---

#### 3.3 Mock Price Data for Development
**Priority:** Medium  
**Estimated Time:** 1 hour

**File:** `src/api/mock_price_api.py`

**Purpose:** Develop overlay system without API dependency

**Implementation:**
```python
class MockPriceAPI:
    """Mock API for development without real API keys"""
    
    MOCK_DATA = {
        'base1-4': {  # Charizard
            'name': 'Charizard',
            'prices': {'tcgplayer_market': 125.50},
            'graded_prices': {'psa10': 450.00}
        },
        'base1-58': {  # Pikachu
            'name': 'Pikachu',
            'prices': {'tcgplayer_market': 8.50},
        }
    }
    
    def get_card_price(self, card_id):
        return self.MOCK_DATA.get(card_id, None)
```

**Usage:** Use mock API until real API keys obtained

---

## Phase 4: Overlay Rendering (Week 5)

### Milestone: Beautiful overlays with card info and prices

#### 4.1 Design Overlay UI
**Priority:** High  
**Estimated Time:** 2 hours

**Considerations:**
- Panel position (avoid covering cards)
- Readability (contrast, font size)
- Information hierarchy (name > price > details)
- Color coding (green for profit, red for loss)
- Semi-transparency for context

**Layout Design:**
```
┌─────────────────────────────┐
│ 📹 Camera Feed              │
│                             │
│    ┌─────────┐             │
│    │  Card   │  ┌──────────┐
│    │  Image  │  │ Charizard│
│    │         │  │ Base Set │
│    └─────────┘  │ $125.50  │
│                 │ PSA 10:  │
│                 │ $450.00  │
│                 └──────────┘
│  FPS: 32  Cards: 2         │
└─────────────────────────────┘
```

**Color Scheme:**
- Bounding box: Green (high conf), Orange (low conf)
- Panel background: Dark gray (50, 50, 50)
- Text: White
- Price up: Green
- Price down: Red

---

#### 4.2 Implement OverlayRenderer
**Priority:** High  
**Estimated Time:** 5 hours

**File:** `src/overlay/renderer.py`

**Requirements:**
- Draw bounding boxes around detected cards
- Draw semi-transparent info panels
- Support multiple cards simultaneously
- Auto-position panels (avoid overlaps)
- Show FPS counter
- Show confidence scores (optional)

**Key Methods:**
```python
class OverlayRenderer:
    def __init__(self, panel_opacity=0.8, font_scale=0.6)
    def render_frame(self, frame, detections, identifications, prices)
    def _draw_bounding_box(self, frame, bbox, confidence)
    def _draw_info_panel(self, frame, bbox, card_info, price_data)
    def _draw_minimal_panel(self, frame, bbox, card_info)  # No price
    def _draw_text(self, frame, text, x, y, scale, bold)
    def _calculate_panel_position(self, bbox, frame_shape) -> (x, y)
```

**Panel Content Priority:**
1. Card name (bold, larger font)
2. Set name (smaller)
3. Market price (if available)
4. Price range (low-high)
5. Graded prices (PSA 10)

**Performance:** Overlay rendering should add <5ms per frame

---

#### 4.3 Implement UI Components
**Priority:** Medium  
**Estimated Time:** 2 hours

**File:** `src/overlay/components.py`

**Reusable Components:**
```python
def draw_rounded_rectangle(frame, top_left, bottom_right, color, radius)
def draw_shadow(frame, bbox)  # Drop shadow effect
def draw_trend_indicator(frame, position, trend)  # ↑ or ↓
def draw_confidence_bar(frame, position, confidence)
def format_price(price: float) -> str  # $125.50
def truncate_text(text, max_length) -> str
```

**Purpose:** Modular, reusable UI elements

---

#### 4.4 Create Overlay Test Suite
**Priority:** Medium  
**Estimated Time:** 2 hours

**File:** `tests/test_overlay.py`

**Tests:**
- Test panel rendering on synthetic images
- Test multi-card rendering (6 cards)
- Test panel position calculation (edge cases)
- Test text rendering and truncation
- Performance test: rendering time <5ms

**Visual Regression Tests:**
- Generate reference images
- Compare rendered output pixel-by-pixel
- Flag any UI changes

---

## Phase 5: Integration & Main Loop (Week 5-6)

### Milestone: Complete end-to-end application

#### 5.1 Complete Main Application
**Priority:** Critical  
**Estimated Time:** 4 hours

**File:** `src/main.py` (currently has TODOs)

**Pipeline:**
```
Camera → Frame Capture → Card Detection → Card Extraction
    ↓
Card Identification → Price Lookup (cached/async)
    ↓
Overlay Rendering → Display → User Input
```

**Main Application Class:**
```python
class PokemonCardPriceOverlay:
    def __init__(self, config_path)
    def _init_detector()
    def _init_identifier()
    def _init_price_api()
    def _init_renderer()
    def process_frame(self, frame) -> np.ndarray
    def run()  # Main loop
```

**Process Frame Logic:**
```python
def process_frame(self, frame):
    # 1. Detect cards
    detections = self.detector.detect(frame)
    
    # 2. Extract card regions
    card_regions = self.detector.extract_card_regions(frame, detections)
    
    # 3. Identify cards
    identifications = {}
    for idx, region in enumerate(card_regions):
        matches = self.identifier.match_card(region, top_k=1)
        if matches:
            identifications[idx] = matches[0]
    
    # 4. Fetch prices (async, cached)
    prices = {}
    for card_info in identifications.values():
        card_id = card_info['card_id']
        prices[card_id] = self.price_api.get_card_price(card_id)
    
    # 5. Render overlays
    output = self.renderer.render_frame(
        frame, detections, identifications, prices
    )
    
    return output
```

**Performance Monitoring:**
- Track FPS (rolling average of 30 frames)
- Track processing time per component
- Log performance warnings if <20 FPS

---

#### 5.2 Add Keyboard Controls
**Priority:** Medium  
**Estimated Time:** 1 hour

**Controls:**
- `q` - Quit application
- `s` - Save screenshot
- `p` - Pause/unpause
- `r` - Reset cache
- `d` - Toggle debug mode (show detection details)
- `c` - Toggle confidence display
- `h` - Show help overlay

**Implementation:**
```python
key = cv2.waitKey(1) & 0xFF

if key == ord('q'):
    break
elif key == ord('s'):
    cv2.imwrite(f'screenshot_{timestamp}.jpg', frame)
elif key == ord('p'):
    self.paused = not self.paused
# ... etc
```

---

#### 5.3 Performance Optimization
**Priority:** High  
**Estimated Time:** 3 hours

**Optimization Strategies:**

1. **Frame Skipping**
   - Process every Nth frame for identification
   - Always process detection (fast)
   - Config: `performance.frame_skip`

2. **Async Price Fetching**
   - Use threading for API calls
   - Don't block frame processing
   ```python
   with ThreadPoolExecutor(max_workers=3) as executor:
       futures = {executor.submit(fetch_price, id): id for id in ids}
   ```

3. **GPU Acceleration**
   - Ensure YOLO runs on GPU
   - Move tensors to CUDA
   ```python
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   model.to(device)
   ```

4. **Feature Matching Optimization**
   - Use FLANN instead of brute force
   - Limit database size for common cards
   - Pre-filter by card color/type

5. **Memory Management**
   - Clear old cache entries
   - Limit frame buffer size
   - Release resources properly

**Target Performance:**
- Detection: <30ms per frame
- Identification: <50ms per card
- Rendering: <5ms per frame
- **Total: 30+ FPS with 1-2 cards**

---

#### 5.4 Error Handling & Logging
**Priority:** High  
**Estimated Time:** 2 hours

**Error Scenarios:**
1. Camera not available
2. Model file missing
3. Card database not found
4. API request failed
5. Out of memory
6. GPU not available

**Error Handling Strategy:**
```python
try:
    frame = self.process_frame(frame)
except DetectionError as e:
    logger.warning(f"Detection failed: {e}")
    # Fall back to edge detection
except IdentificationError as e:
    logger.warning(f"Identification failed: {e}")
    # Show detection only
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Continue with next frame
```

**Logging Levels:**
- DEBUG: Frame-by-frame details
- INFO: Startup, card identified, price fetched
- WARNING: Failed identification, API errors
- ERROR: Critical failures

---

## Phase 6: Testing & Refinement (Week 6)

### Milestone: Stable, tested application

#### 6.1 Comprehensive Testing
**Priority:** High  
**Estimated Time:** 8 hours

**Test Coverage:**

1. **Unit Tests** (`tests/test_*.py`)
   - Detection: `test_detection.py`
   - Identification: `test_identification.py`
   - API: `test_api.py`
   - Overlay: `test_overlay.py`
   - Config: `test_config.py`
   - Utils: `test_utils.py`

2. **Integration Tests** (`tests/test_integration.py`)
   - Full pipeline with test fixtures
   - Multi-card scenarios
   - Edge cases (no cards, obstructed cards)

3. **Performance Tests** (`scripts/benchmark.py`)
   - FPS benchmarking
   - Latency measurement
   - Memory profiling

**Test Coverage Target:** >80%

**Run Tests:**
```bash
pytest tests/ -v --cov=src --cov-report=html
```

---

#### 6.2 Create Test Fixtures
**Priority:** Medium  
**Estimated Time:** 2 hours

**Fixtures Needed:**
- `tests/fixtures/single_card.jpg` - One card, clear
- `tests/fixtures/multiple_cards.jpg` - 4-6 cards
- `tests/fixtures/rotated_card.jpg` - Tilted card
- `tests/fixtures/blurry_card.jpg` - Low quality
- `tests/fixtures/no_cards.jpg` - Background only
- `tests/fixtures/mock_database/` - 10 test cards

**Creating Fixtures:**
- Take photos with webcam
- Use synthetic data generation
- Download from Pokemon TCG API

---

#### 6.3 Real-World Testing
**Priority:** Critical  
**Estimated Time:** 4 hours

**Test Scenarios:**

1. **Lighting Conditions**
   - Bright overhead light
   - Dim room
   - Backlit
   - Mixed lighting

2. **Card Conditions**
   - Mint condition cards
   - Played/damaged cards
   - Holographic vs non-holographic
   - Different editions (1st edition, shadowless)

3. **Camera Angles**
   - Straight on (ideal)
   - 15° tilt
   - 30° tilt
   - Cards overlapping

4. **Multiple Cards**
   - 1 card (easy)
   - 2-3 cards (medium)
   - 4-6 cards (challenging)

**Success Criteria:**
- >90% identification accuracy in good conditions
- >70% accuracy in poor conditions
- Stable 30 FPS with 1-2 cards
- >20 FPS with 4-6 cards

---

#### 6.4 Documentation Updates
**Priority:** Medium  
**Estimated Time:** 2 hours

**Documentation Needs:**

1. **README.md Updates**
   - Add installation status
   - Update quick start guide
   - Add troubleshooting section
   - Add demo GIF/video

2. **API Documentation** (`docs/api_reference.md`)
   - Document all public methods
   - Add usage examples
   - Document configuration options

3. **Development Guide** (`docs/development_guide.md`)
   - How to add new card sets
   - How to train custom YOLO model
   - How to add new price sources

4. **Deployment Guide** (`docs/deployment.md`)
   - Docker deployment
   - Performance tuning
   - Production considerations

---

## Phase 7: Polish & Enhancement (Week 7+)

### Optional Enhancements (Post-MVP)

#### 7.1 Train Custom YOLO Model
**Priority:** Low  
**Estimated Time:** 1 week

**Purpose:** Improve detection accuracy specifically for Pokemon cards

**Process:**
1. Collect training data (1000+ images of Pokemon cards)
2. Annotate bounding boxes (use LabelImg or Roboflow)
3. Fine-tune YOLO11 on card dataset
4. Evaluate on test set
5. Compare with base model

**Expected Improvement:** 5-10% better accuracy

---

#### 7.2 Add Price Trend Tracking
**Priority:** Low  
**Estimated Time:** 4 hours

**Features:**
- Store historical prices in SQLite
- Show price change % (↑ 15% this week)
- Plot mini price chart on overlay
- Alert on significant price changes

---

#### 7.3 Collection Management
**Priority:** Low  
**Estimated Time:** 1 week

**Features:**
- Scan and save cards to collection
- Track collection value
- Export to CSV
- Integration with TCGPlayer collection

---

#### 7.4 Mobile App Support
**Priority:** Low  
**Estimated Time:** 2-3 weeks

**Options:**
- React Native app
- Flutter app
- Progressive Web App (PWA)

**Challenges:**
- Mobile GPU support
- Camera API differences
- Smaller screen space

---

#### 7.5 Card Condition Assessment
**Priority:** Low  
**Estimated Time:** 2 weeks

**Features:**
- Detect surface scratches
- Detect edge wear
- Estimate condition grade (Near Mint, Lightly Played, etc.)
- Adjust price based on condition

**Approach:** Use CNN classifier trained on graded card images

---

## Risk Assessment & Mitigation

### High-Risk Items

**Risk 1: Poor Feature Matching Accuracy**
- **Impact:** High (core functionality)
- **Probability:** Medium
- **Mitigation:**
  - Start with small, high-quality database
  - Tune parameters carefully
  - Fall back to manual card selection
  - Consider adding QR codes to cards

**Risk 2: API Costs/Availability**
- **Impact:** Medium (prices unavailable)
- **Probability:** Medium
- **Mitigation:**
  - Use free Pokemon TCG API first
  - Add mock data for development
  - Implement multiple API sources
  - Consider web scraping (last resort)

**Risk 3: Performance Below 30 FPS**
- **Impact:** Medium (poor UX)
- **Probability:** Low (with GPU)
- **Mitigation:**
  - Use YOLO11n (nano) model
  - Implement frame skipping
  - Optimize feature matching
  - Require GPU for production use

**Risk 4: WSL Camera Access Issues**
- **Impact:** High (can't test)
- **Probability:** Medium
- **Mitigation:**
  - Use USB camera passthrough
  - Test on native Linux
  - Use video file for development
  - Dual boot for testing

---

## Success Metrics

### MVP Definition (Minimum Viable Product)
- ✅ Detects Pokemon cards at 30 FPS
- ✅ Identifies cards from database with >85% accuracy
- ✅ Displays card name and set
- ✅ Shows price (even if mock data)
- ✅ Handles 2-3 cards simultaneously
- ✅ Works in good lighting conditions

### Full Success Criteria
- ✅ >90% identification accuracy
- ✅ Real price data from API
- ✅ Handles 4-6 cards
- ✅ Works in various lighting
- ✅ Graded card prices (PSA)
- ✅ Price trends
- ✅ Stable performance on consumer hardware

---

## Timeline Summary

| Phase | Duration | Milestone |
|-------|----------|-----------|
| Phase 1: Setup & Detection | Week 1 | Card detection working |
| Phase 2: Database & Matching | Week 2-3 | Card identification working |
| Phase 3: Price API | Week 4 | Prices displayed |
| Phase 4: Overlay | Week 5 | Beautiful UI |
| Phase 5: Integration | Week 5-6 | Full app working |
| Phase 6: Testing | Week 6 | Stable, tested |
| Phase 7: Enhancements | Week 7+ | Polish & features |

**Total Time to MVP:** 6 weeks  
**Total Time to Full Release:** 7-8 weeks

---

## Next Immediate Actions

### Today (November 19, 2025)
1. ✅ Review this plan
2. [ ] Set up Python environment with `uv`
3. [ ] Download YOLO11 model
4. [ ] Test webcam access

### This Week
1. [ ] Implement `YOLOCardDetector`
2. [ ] Create detection test script
3. [ ] Verify end-to-end detection works

### Week 2
1. [ ] Build card database (10-20 cards)
2. [ ] Implement `FeatureMatcher`
3. [ ] Test identification accuracy

---

## Resource Requirements

### Development Environment
- Python 3.10+
- 16GB RAM (8GB minimum)
- NVIDIA GPU (GTX 1060 or better)
- 10GB free disk space
- Webcam (720p minimum)

### External Services
- Pokemon TCG API (free, 1000 req/hour)
- PokemonPriceTracker API (paid, ~$20/month)
- OR TCGPlayer API (free, requires approval)

### Time Investment
- Part-time (10 hrs/week): 12 weeks
- Full-time (40 hrs/week): 6 weeks

---

## Conclusion

PriceLens is a well-architected project with excellent documentation but minimal implementation. This plan provides a clear, phased approach to reach MVP within 6-8 weeks. The key is to start simple (detection only), then layer on complexity (identification, prices, UI) incrementally.

**Recommended Start:** Begin with Phase 1 immediately - set up environment and get basic detection working within 1 day.

---

*Plan generated by GitHub Copilot*  
*Last updated: November 19, 2025*
