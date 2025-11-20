# PriceLens Implementation Plan V2 (Enhanced)
**Generated:** November 20, 2025  
**Status:** Project is ~10-15% complete (foundation only)  
**Target:** Enhanced MVP with modern architecture and advanced features

---

## Executive Summary

This enhanced plan incorporates modern best practices, performance optimizations, and advanced features to create a production-ready Pokemon card price overlay system. Timeline: 8-10 weeks to enhanced MVP.

### What's New in V2
- ✨ Event-driven architecture for better modularity
- 🚀 Performance optimizations (model quantization, smart caching)
- 🌐 Web dashboard and API
- 🔌 Plugin system for extensibility
- 📊 Analytics and collection management
- 🧪 Modern testing infrastructure
- 🔄 CI/CD pipeline
- 📱 Mobile-ready web interface

---

## Phase 0: Foundation Improvements (Week 1)

### 0.1 Modernize Dependencies
**Priority:** Critical  
**Time:** 2 hours

**Tasks:**
- [ ] Update `requirements.txt` to latest stable versions
- [ ] Add `pyproject.toml` for modern packaging
- [ ] Add `ruff` for faster linting
- [ ] Set up pre-commit hooks

**Updated Dependencies:**
```toml
ultralytics = ">=8.3.24"  # Latest YOLO
torch = ">=2.5.0"         # Better performance
opencv-python = ">=4.10.0"
fastapi = ">=0.115.0"
pydantic = ">=2.9.0"
```

---

### 0.2 Development Infrastructure
**Priority:** High  
**Time:** 3 hours

**Tasks:**
- [ ] Add `.pre-commit-config.yaml`
- [ ] Create GitHub Actions CI/CD workflow
- [ ] Set up pytest with coverage
- [ ] Add property-based testing with Hypothesis

**CI/CD Pipeline:**
```yaml
# .github/workflows/ci.yml
- Test on Python 3.10, 3.11, 3.12
- Run linting (ruff)
- Type checking (mypy)
- Test coverage >80%
- Benchmark performance
```

---

## Phase 1: Core Detection with Improvements (Week 1-2)

### 1.1 Environment Setup
**Priority:** Critical  
**Time:** 30 minutes

Same as original plan, plus:
- [ ] Set up Docker dev container
- [ ] Configure VS Code/Cursor settings
- [ ] Install pre-commit hooks

---

### 1.2 Enhanced YOLO Detector
**Priority:** Critical  
**Time:** 6 hours (2 hours more than original)

**File:** `src/detection/yolo_detector.py`

**Enhancements over original:**
- Model quantization support (INT8)
- ONNX export for cross-platform
- Multi-scale detection
- Card tracking across frames

**Key Methods:**
```python
class YOLOCardDetector(DetectorBase):
    def __init__(self, model_path, quantize=False, export_onnx=False)
    def detect(self, frame) -> List[Detection]
    def track_cards(self, frame) -> List[TrackedCard]  # NEW
    def export_optimized_model(self, format='onnx')    # NEW
```

**Performance Target:**
- 60+ FPS with quantized model (vs 30 FPS baseline)
- <20ms inference time

---

### 1.3 Frame Interpolation System
**Priority:** High  
**Time:** 4 hours

**File:** `src/detection/frame_interpolator.py`

**Purpose:** Predict card positions between frames to reduce computation

```python
class FrameInterpolator:
    def __init__(self):
        self.kalman_filters = {}  # Track card positions
    
    def predict_position(self, card_id, frame_idx):
        # Use Kalman filter for smooth tracking
        pass
    
    def should_reidentify(self, card_id) -> bool:
        # Only re-identify if card moved significantly
        pass
```

**Benefits:**
- 2-3x FPS improvement
- Smoother overlays
- Reduced API calls

---

## Phase 2: Advanced Identification (Week 2-3)

### 2.1 Card Database (Same as original)
**Priority:** High  
**Time:** 3 hours

---

### 2.2 Enhanced Feature Matching
**Priority:** High  
**Time:** 8 hours (2 hours more)

**File:** `src/identification/feature_matcher.py`

**Enhancements:**
- OCR for set numbers (EasyOCR)
- Embeddings-based search with ChromaDB
- Multi-method fallback (ORB → SIFT → OCR)

```python
class EnhancedFeatureMatcher(IdentifierBase):
    def __init__(self):
        self.orb_matcher = ORBMatcher()
        self.ocr_reader = easyocr.Reader(['en'])
        self.embedding_db = chromadb.Client()
    
    def identify(self, image, method='hybrid'):
        # Try ORB first (fast)
        # Fall back to OCR if needed
        # Use embeddings for similarity search
        pass
```

---

### 2.3 State Management System
**Priority:** High  
**Time:** 4 hours

**File:** `src/core/state_manager.py`

**Purpose:** Track card states across frames

```python
class CardState(Enum):
    DETECTED = "detected"
    IDENTIFYING = "identifying"
    IDENTIFIED = "identified"
    FETCHING_PRICE = "fetching_price"
    READY = "ready"
    LOST = "lost"

class StateManager:
    def update_card_state(self, card_id, new_state)
    def get_cards_in_state(self, state) -> List[Card]
    def cleanup_lost_cards(self)
```

**Benefits:**
- Avoid redundant API calls
- Smooth UI transitions
- Better error handling

---

## Phase 3: Smart API & Caching (Week 3-4)

### 3.1 Multi-Level Cache System
**Priority:** High  
**Time:** 5 hours

**File:** `src/api/smart_cache.py`

**Architecture:**
```python
class SmartCache:
    def __init__(self):
        self.memory_cache = LRUCache(maxsize=100)  # Hot cache
        self.disk_cache = SqliteCache(ttl=3600)    # Persistent
    
    def get(self, key):
        # Try memory → disk → API
        pass
```

**Performance:**
- <1ms for hot cards (vs 100ms API call)
- Persistent across sessions

---

### 3.2 Price API with Analytics
**Priority:** High  
**Time:** 6 hours

**Files:** 
- `src/api/price_api.py` (enhanced)
- `src/api/price_history.py` (new)

**Features:**
- Price history storage (SQLite)
- Trend detection (↑15% this week)
- Price alerts

```python
class PriceHistoryTracker:
    def store_price(self, card_id, price, timestamp)
    def get_price_trend(self, card_id, days=30)
    def detect_price_spike(self, threshold=0.2)
```

---

## Phase 4: Event-Driven Architecture (Week 4-5)

### 4.1 Event Bus System
**Priority:** High  
**Time:** 6 hours

**File:** `src/core/event_bus.py`

**Purpose:** Decouple components for better modularity

```python
class EventBus:
    def __init__(self):
        self.handlers = defaultdict(list)
    
    def emit(self, event: str, data: Any):
        for handler in self.handlers[event]:
            handler(data)
    
    def on(self, event: str, handler: Callable):
        self.handlers[event].append(handler)

# Events:
# - frame.captured
# - cards.detected
# - cards.identified
# - prices.fetched
# - frame.rendered
```

---

### 4.2 Plugin System
**Priority:** Medium  
**Time:** 5 hours

**File:** `src/core/plugin_manager.py`

**Purpose:** Enable extensibility

```python
class Plugin(ABC):
    @abstractmethod
    def on_card_detected(self, card): pass
    
    @abstractmethod
    def on_card_identified(self, card): pass

# Built-in plugins:
class CardHistoryPlugin(Plugin):
    # Track all cards seen
    
class PriceAlertPlugin(Plugin):
    # Alert on valuable cards
    
class CollectionTrackerPlugin(Plugin):
    # Mark owned cards
```

---

## Phase 5: Modern UI & Rendering (Week 5-6)

### 5.1 Advanced Overlay Renderer
**Priority:** High  
**Time:** 6 hours

**File:** `src/overlay/renderer.py`

**Enhancements:**
- Price trend indicators (↑↓)
- Mini price charts
- Smooth animations
- Theme support

---

### 5.2 Web Dashboard (NEW)
**Priority:** High  
**Time:** 12 hours

**Files:**
- `src/web/app.py` - FastAPI backend
- `src/web/static/` - Frontend (HTML/JS)

**Features:**
```python
@app.websocket("/ws/stream")
async def stream_video():
    # Live video streaming
    
@app.get("/api/cards/recent")
async def get_recent_cards():
    # API for card history
    
@app.get("/api/collection/value")
async def get_collection_value():
    # Portfolio analytics
```

**Benefits:**
- Access from any device
- Remote monitoring
- Share sessions
- Better analytics

---

## Phase 6: Data & Analytics (Week 6-7)

### 6.1 Database Layer
**Priority:** High  
**Time:** 6 hours

**File:** `src/database/models.py`

**Schema:**
```python
class CardScan(Base):
    id: int
    card_id: str
    card_name: str
    price: float
    scanned_at: datetime
    confidence: float
    session_id: str

class Collection(Base):
    id: int
    card_id: str
    acquired_date: datetime
    purchase_price: float
    current_price: float
```

---

### 6.2 Collection Manager
**Priority:** Medium  
**Time:** 8 hours

**File:** `src/features/collection_manager.py`

**Features:**
```python
class CollectionManager:
    def add_to_collection(self, card)
    def remove_from_collection(self, card_id)
    def calculate_total_value(self) -> float
    def export_to_csv(self, path)
    def get_top_gainers(self, n=10)
    def get_portfolio_summary(self)
```

---

## Phase 7: Testing & Quality (Week 7-8)

### 7.1 Comprehensive Test Suite
**Priority:** Critical  
**Time:** 12 hours

**Test Types:**
1. **Unit Tests** - All components
2. **Integration Tests** - Full pipeline
3. **Property-Based Tests** - Edge cases with Hypothesis
4. **Performance Tests** - Benchmarking
5. **Visual Regression** - UI consistency

**Target:** >85% code coverage

---

### 7.2 Performance Benchmarking
**Priority:** High  
**Time:** 4 hours

**File:** `scripts/benchmark_v2.py`

**Metrics:**
```python
def benchmark_suite():
    # Detection speed (target: <20ms)
    # Identification accuracy (target: >92%)
    # End-to-end latency (target: <80ms)
    # Memory usage (target: <500MB)
    # CPU/GPU utilization
```

---

## Phase 8: Polish & Deployment (Week 8-10)

### 8.1 Multi-Platform Docker
**Priority:** Medium  
**Time:** 6 hours

**Files:**
- `docker/Dockerfile.cpu` - Lightweight
- `docker/Dockerfile.gpu` - CUDA support
- `docker/Dockerfile.arm` - Raspberry Pi

---

### 8.2 Desktop Packaging
**Priority:** Low  
**Time:** 4 hours

**Tool:** PyInstaller

```bash
pyinstaller --onefile \
    --add-data "models:models" \
    --add-data "config.yaml:." \
    --windowed \
    --name PriceLens \
    src/main.py
```

---

### 8.3 Documentation
**Priority:** High  
**Time:** 8 hours

**Updates:**
- Interactive MkDocs site
- API documentation (auto-generated)
- Video tutorials
- Deployment guides

---

## Phase 9: Advanced Features (Optional - Week 10+)

### 9.1 Graded Card Detection
**Time:** 1 week

Detect PSA/BGS slabs using OCR

### 9.2 Card Condition Assessment
**Time:** 2 weeks

ML model to grade card condition

### 9.3 Mobile App Prototype
**Time:** 2 weeks

Streamlit-based mobile interface

---

## Enhanced Success Metrics

| Metric | V1 Target | V2 Target | Current |
|--------|-----------|-----------|---------|
| **Detection FPS** | 30 | 60 | TBD |
| **Detection Accuracy** | >95% | >97% | TBD |
| **Identification Accuracy** | >90% | >92% | TBD |
| **End-to-End Latency** | <100ms | <80ms | TBD |
| **Multi-Card Support** | 4-6 | 6-10 | TBD |
| **Code Coverage** | >80% | >85% | TBD |
| **Platforms** | Desktop | Desktop+Web | TBD |

---

## Updated Timeline

| Phase | Duration | Key Deliverable |
|-------|----------|-----------------|
| 0: Foundation | Week 1 | Modern dev infrastructure |
| 1: Detection | Week 1-2 | Optimized detection (60 FPS) |
| 2: Identification | Week 2-3 | Enhanced matching + OCR |
| 3: API & Cache | Week 3-4 | Smart caching + analytics |
| 4: Architecture | Week 4-5 | Event system + plugins |
| 5: UI & Web | Week 5-6 | Web dashboard |
| 6: Data | Week 6-7 | Database + collection mgmt |
| 7: Testing | Week 7-8 | 85%+ coverage |
| 8: Deploy | Week 8-10 | Multi-platform release |

**Total Time to Enhanced MVP:** 8-10 weeks

---

## Key Improvements Over V1

### Architecture
- ✅ Event-driven design (vs monolithic)
- ✅ Plugin system (extensible)
- ✅ State management (better tracking)

### Performance
- ✅ 2x faster (60 vs 30 FPS)
- ✅ Smart caching (10x faster for hot data)
- ✅ Model quantization (4x inference speed)

### Features
- ✅ Web dashboard (mobile access)
- ✅ Collection management
- ✅ Price trends & alerts
- ✅ OCR fallback

### Developer Experience
- ✅ Modern dependencies
- ✅ CI/CD pipeline
- ✅ Pre-commit hooks
- ✅ Comprehensive tests

### User Experience
- ✅ Smoother animations
- ✅ Better error handling
- ✅ Multi-platform support
- ✅ Analytics dashboard

---

## Next Steps (Updated)

### This Week
1. [ ] Update dependencies to latest versions
2. [ ] Set up pre-commit hooks and CI/CD
3. [ ] Implement event bus system
4. [ ] Create smart cache with LRU
5. [ ] Download and quantize YOLO model

### Next Week
1. [ ] Implement enhanced YOLO detector
2. [ ] Add frame interpolation
3. [ ] Build card database
4. [ ] Start web dashboard

---

## Resource Requirements (Updated)

### Development
- Python 3.10+ (3.11+ recommended)
- 16GB RAM minimum
- NVIDIA GPU (GTX 1660 or better for 60 FPS)
- 15GB free disk space (models + database)

### External Services
- Pokemon TCG API (free)
- PokemonPriceTracker API (~$20/month)
- GitHub Actions (free for public repos)
- Optional: Redis for shared cache

---

## Risk Mitigation (Updated)

### New Risks in V2

**Risk: Increased Complexity**
- Mitigation: Phased rollout, comprehensive testing

**Risk: Web Security**
- Mitigation: API authentication, rate limiting, CORS

**Risk: Database Scaling**
- Mitigation: Use SQLite initially, PostgreSQL for production

---

## Conclusion

This enhanced plan transforms PriceLens from a basic MVP into a production-ready application with modern architecture, better performance, and advanced features. The modular design enables future extensions while maintaining code quality.

**Recommended Start:** Phase 0 (modernize dependencies and setup CI/CD) to establish a solid foundation.

---

*Implementation Plan V2 - Enhanced*  
*Created: November 20, 2025*  
*Estimated Total Development Time: 8-10 weeks*
