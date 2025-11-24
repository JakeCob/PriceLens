# PriceLens Implementation Plan V3 (Enhanced + Mobile)
**Generated:** November 24, 2025  
**Status:** Phase 1-3 Complete (~40% overall)  
**Target:** Production-ready web + native mobile app

---

## What's New in V3

Building on V2's foundation with:
- 📱 **Flutter Mobile App** - Native iOS/Android with device camera
- 🌍 **Multi-language Support** - Detect cards in Japanese, Korean, etc.
- 💰 **Multi-currency Pricing** - PHP, USD, EUR, JPY support
- 📊 **Running Total** - Live sum of all detected cards
- 🔊 **Audio Feedback** - Cash register sound on detection
- 📦 **Product Detection** - Packs, ETBs, UPCs, bundles
- 🏷️ **Enhanced Filters** - Set info, rarity badges, condition

---

## Phase 10: Enhanced UX Features (Week 10-11)

### 10.1 Running Total Display
**Priority:** High  
**Time:** 4 hours

**Files:**
- `src/web/static/camera.js` - Add total display
- `src/web/api.py` - Track session total

**Features:**
```javascript
class SessionTracker {
    constructor() {
        this.detectedCards = new Set();
        this.runningTotal = 0;
    }
    
    addCard(cardId, price) {
        if (!this.detectedCards.has(cardId)) {
            this.detectedCards.add(cardId);
            this.runningTotal += price;
            this.updateDisplay();
        }
    }
}
```

**UI:**
- Floating total badge: `💰 Total: $245.50`
- "Clear Session" button
- Export detected cards list

---

### 10.2 Audio Feedback System
**Priority:** Medium  
**Time:** 3 hours

**Files:**
- `src/web/static/sounds/` - Audio assets
- `src/web/static/camera.js` - Audio manager

**Sounds:**
- Cash register "cha-ching" on new card
- Subtle beep on re-detection (same card)
- Different pitch for high-value cards (>$50)

```javascript
class AudioManager {
    play(type) {
        const sounds = {
            'new_card': '/static/sounds/cash_register.mp3',
            'high_value': '/static/sounds/jackpot.mp3',
            'duplicate': '/static/sounds/beep.mp3'
        };
        new Audio(sounds[type]).play();
    }
}
```

---

### 10.3 Multi-Currency Support
**Priority:** High  
**Time:** 6 hours

**Files:**
- `src/api/currency_converter.py` - Exchange rate API
- `src/web/api.py` - Currency parameter
- `src/web/static/camera.js` - Currency selector

**Features:**
```python
class CurrencyConverter:
    def __init__(self):
        self.rates = self.fetch_rates()  # ExchangeRate-API
    
    def convert(self, usd_price: float, to_currency: str) -> float:
        return usd_price * self.rates[to_currency]
```

**Supported Currencies:**
- USD (default)
- PHP (Philippine Peso)
- EUR (Euro)
- GBP (British Pound)
- JPY (Japanese Yen)
- CAD (Canadian Dollar)

**UI:**
- Currency dropdown selector
- Auto-save preference (localStorage)
- Show both USD and selected currency

---

### 10.4 Set Information Display
**Priority:** Medium  
**Time:** 3 hours

**Files:**
- `src/web/static/camera.js` - Enhanced overlay
- `data/set_metadata.json` - Set logos/icons

**Display:**
```
┌─────────────────────────┐
│ Charizard ex           │
│ 🔥 ME1 #006            │ ← Set badge
│ Mythical Island        │ ← Set name
│ ★★★★ Ultra Rare        │ ← Rarity
│ $45.99 (₱2,599)        │ ← Multi-currency
└─────────────────────────┘
```

---

## Phase 11: Multi-Language Card Detection (Week 11-12)

### 11.1 Language Detection
**Priority:** High  
**Time:** 8 hours

**Files:**
- `src/identification/language_detector.py`
- `src/identification/feature_matcher.py` - Language filtering

**Supported Languages:**
- English
- Japanese
- Korean
- Chinese (Simplified/Traditional)
- French, German, Italian, Spanish

**Approach:**
```python
class LanguageDetector:
    def detect_language(self, card_region: np.ndarray) -> str:
        # 1. OCR text extraction
        text = self.ocr_reader.readtext(card_region)
        
        # 2. Language classification
        # Japanese: Hiragana/Katakana/Kanji
        # Korean: Hangul
        # etc.
        
        return detected_language
```

**Database:**
- Separate feature DBs per language
- `data/features/japanese_features.pkl`
- `data/features/korean_features.pkl`

---

### 11.2 Multi-Language Card Database
**Priority:** High  
**Time:** 6 hours

**Tasks:**
- Download Japanese card images
- Download Korean card images
- Extract features for each language
- Update matcher to search correct DB

---

## Phase 12: Product Detection (Week 12-13)

### 12.1 YOLO Model Fine-tuning
**Priority:** High  
**Time:** 12 hours

**New Classes:**
- Booster Pack
- Elite Trainer Box (ETB)
- Booster Bundle
- Ultra Premium Collection (UPC)
- Build & Battle Box
- Pre-release Kit

**Training:**
- Collect 500+ images per class
- Annotate with LabelImg
- Fine-tune YOLO11m on product dataset
- Achieve >90% mAP

---

### 12.2 Product Price Fetching
**Priority:** High  
**Time:** 6 hours

**Files:**
- `src/api/product_prices.py`
- `data/products/product_database.json`

**Sources:**
- TCGPlayer sealed products
- eBay sold listings
- Local retailer APIs

**Database:**
```json
{
  "products": [
    {
      "name": "Mythical Island Booster Pack",
      "type": "booster_pack",
      "set": "ME1",
      "prices": {
        "USD": 4.99,
        "PHP": 280
      }
    }
  ]
}
```

---

### 12.3 Bundle Detection UI
**Priority:** Medium  
**Time:** 4 hours

**Features:**
- Different overlay colors for products vs cards
- Pack icon: 📦
- ETB icon: 🎁
- Product hover: Show contents (e.g., "10 packs")

---

## Phase 13: Flutter Mobile App (Week 13-16)

### 13.1 Project Setup
**Priority:** Critical  
**Time:** 6 hours

**Tasks:**
```bash
flutter create pricelens_mobile
cd pricelens_mobile
flutter pub add camera
flutter pub add http
flutter pub add provider
flutter pub add audioplayers
```

**Structure:**
```
lib/
├── main.dart
├── screens/
│   ├── camera_screen.dart
│   ├── history_screen.dart
│   └── settings_screen.dart
├── services/
│   ├── detection_service.dart
│   ├── audio_service.dart
│   └── storage_service.dart
├── models/
│   └── card_detection.dart
└── widgets/
    ├── overlay_painter.dart
    └── total_display.dart
```

---

### 13.2 Camera Integration
**Priority:** Critical  
**Time:** 8 hours

**Files:**
- `lib/screens/camera_screen.dart`

**Features:**
```dart
class CameraScreen extends StatefulWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack([
        CameraPreview(controller),
        OverlayPainter(detections),
        TotalDisplay(runningTotal),
        FloatingActionButton(
          onPressed: captureFrame,
          child: Icon(Icons.camera),
        ),
      ]),
    );
  }
}
```

**Camera Features:**
- Real-time preview (30 FPS)
- Frame capture every 500ms
- Auto-focus on card region
- Flash toggle
- Camera flip (front/back)

---

### 13.3 API Integration
**Priority:** Critical  
**Time:** 6 hours

**Files:**
- `lib/services/detection_service.dart`

```dart
class DetectionService {
  Future<List<CardDetection>> detectCards(Uint8List imageBytes) async {
    final response = await http.post(
      Uri.parse('http://your-api.com/detect-live'),
      headers: {'Content-Type': 'image/jpeg'},
      body: imageBytes,
    );
    
    return parseDetections(response.body);
  }
}
```

**Optimizations:**
- Image compression before upload
- Request batching
- Offline caching

---

### 13.4 Native Features
**Priority:** High  
**Time:** 8 hours

**Features:**

**1. Device Camera Optimization**
- Use native resolution
- Hardware acceleration
- Low-latency mode

**2. Haptic Feedback**
```dart
void onCardDetected() {
  HapticFeedback.mediumImpact();
  audioService.play('cash_register');
}
```

**3. Background Mode**
- Keep camera active in background
- Battery optimization warnings

**4. Storage**
```dart
class StorageService {
  Future<void> saveSession(Session session) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('last_session', jsonEncode(session));
  }
}
```

---

### 13.5 Mobile UI/UX
**Priority:** High  
**Time:** 10 hours

**Screens:**

**1. Camera Screen**
- Full-screen camera view
- Floating total: `💰 ₱5,250`
- Detected cards counter: `8 cards`
- Settings gear icon
- History button

**2. Settings Screen**
- Currency selector
- Sound toggle
- Language preference
- API endpoint config
- Clear history

**3. History Screen**
- List of all detected cards
- Total per session
- Export to CSV
- Share via WhatsApp/Messenger

**Design:**
- Material Design 3
- Dark mode support
- Smooth animations
- Gesture controls (swipe to clear)

---

### 13.6 Deployment
**Priority:** High  
**Time:** 6 hours

**Tasks:**

**iOS:**
```yaml
# ios/Runner/Info.plist
<key>NSCameraUsageDescription</key>
<string>PriceLens needs camera access to detect Pokemon cards</string>
```
- TestFlight beta release
- App Store submission

**Android:**
```xml
<!-- AndroidManifest.xml -->
<uses-permission android:name="android.permission.CAMERA"/>
<uses-feature android:name="android.hardware.camera"/>
```
- Google Play internal testing
- Production release

---

## Phase 14: Portfolio Management & SaaS Features (Week 16-18)

> **💡 Reddit Community Feedback:** *"Do card collectors actually want bulk scanning + portfolio value tracking without switching between eBay/TCGPlayer? If you add export + portfolio valuation + grading estimates, I could definitely see people paying for it."*

### 14.1 Bulk Scanning & Session Management
**Priority:** Very High (Product-Market Fit)  
**Time:** 8 hours

**Files:**
- `src/web/session_manager.py`
- `src/web/static/sessions.js`
- `src/database/models.py`

**Features:**
```python
class BulkScanSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.cards = []
        self.total_value = 0.0
        self.started_at = datetime.now()
        
    def add_card(self, card_data):
        """Add card without switching apps"""
        self.cards.append(card_data)
        self.total_value += card_data['price']
        self.save_to_db()
```

**UI:**
- "Start Bulk Scan" button
- Live counter: `📦 42 cards scanned`
- Session history: `"Binder 1 - Nov 24 (158 cards, $2,450)"`
- Resume interrupted sessions
- Merge sessions

---

### 14.2 Portfolio Valuation Dashboard
**Priority:** Very High (Monetization)  
**Time:** 12 hours

**Files:**
- `src/portfolio/analytics.py`
- `src/web/templates/portfolio.html`
- `src/web/api.py` - Portfolio endpoints

**Features:**
```python
class PortfolioAnalyzer:
    def get_total_value(self, user_id: str) -> float:
        """Calculate total portfolio value"""
        
    def get_price_history(self, days=30) -> List[Tuple]:
        """Track portfolio value over time"""
        
    def get_top_gainers(self, n=10) -> List[Card]:
        """Cards that increased in value"""
        
    def get_marketplace_comparison(self, card_id) -> Dict:
        """Compare TCGPlayer vs eBay vs CardMarket"""
```

**Dashboard:**
```
╔══════════════════════════════════════╗
║  💼 Portfolio Value: $12,450.75     ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ║
║  📈 +$450 (3.8%) this month          ║
║  🎯 Top Card: Charizard ex ($850)   ║
║  📊 Total Cards: 1,247               ║
╚══════════════════════════════════════╝

📊 Value Breakdown by Set:
- Mythical Island: $3,250 (26%)
- Base Set: $2,100 (17%)
- Evolving Skies: $1,800 (14%)

⚡ Top Movers (30 days):
1. Umbreon VMAX +$45 (+38%)
2. Rayquaza VMAX +$32 (+28%)
3. Mew ex +$28 (+25%)
```

---

### 14.3 Multi-Source Price Comparison
**Priority:** High (No more switching apps!)  
**Time:** 10 hours

**Files:**
- `src/api/marketplace_aggregator.py`
- `src/scrapers/tcgplayer_scraper.py`
- `src/scrapers/ebay_scraper.py`

**Sources:**
- TCGPlayer (Market Price)
- eBay (Sold Listings - Last 30 days)
- CardMarket (EU pricing)
- StockX (Sealed products)
- Local card shops (if API available)

**UI Display:**
```
Charizard ex ME1 #006
┌─────────────────────────┐
│ 💰 Best Price: $42.50   │ ← Highlighted
│ TCGPlayer: $45.99       │
│ eBay Avg: $48.20        │
│ CardMarket: €41.00      │
└─────────────────────────┘
📊 30-day trend: ↗ +$3.50
```

**Benefits:**
- One-stop price check
- See all marketplaces at once
- Historical sold prices
- Arbitrage opportunities

---

### 14.4 Advanced Export Functionality
**Priority:** High (Sharing & Integration)  
**Time:** 6 hours

**Files:**
- `src/export/exporters.py`
- `src/web/api.py` - Export endpoints

**Export Formats:**

**1. CSV Export**
```csv
Card Name,Set,Number,Rarity,Condition,Price USD,Price PHP,TCGPlayer,eBay,Last Updated
Charizard ex,ME1,006,Ultra Rare,NM,45.99,2599,45.99,48.20,2025-11-24
```

**2. Excel (.xlsx)**
- Multiple sheets: Cards, Summary, Price History
- Conditional formatting (price changes)
- Charts & graphs

**3. PDF Report**
- Professional portfolio report
- Card images included
- QR code for digital verification
- Suitable for insurance claims

**4. Integration Exports**
- TCGPlayer Collection (import format)
- Dragon Shield Card Manager
- Pokellector format
- TCG Collector App

**5. Share Options**
- Share link (view-only portfolio)
- QR code for mobile sharing
- Social media cards (Twitter/FB preview)

---

### 14.5 Card Condition & Grading Estimates
**Priority:** Medium (Value-add feature)  
**Time:** 16 hours

**Files:**
- `src/grading/condition_analyzer.py`
- `src/ml/grading_model.py`

**Approach:**
```python
class ConditionAnalyzer:
    def estimate_grade(self, card_image: np.ndarray) -> Dict:
        """
        Analyze card condition using CV
        Returns: {
            'grade': 8.5,  # PSA/BGS equivalent
            'confidence': 0.82,
            'issues': ['minor edge wear', 'slight centering'],
            'estimated_value_graded': 120.50,
            'raw_value': 45.99
        }
        """
        # Check for:
        # - Centering (top/bottom, left/right)
        # - Surface condition (scratches, dents)
        # - Edge wear
        # - Corner sharpness
```

**ML Model Training:**
- Dataset: 10,000+ graded cards (PSA/BGS images)
- Features: Edge detection, scratch detection, centering metrics
- Model: ResNet-50 or EfficientNet
- Accuracy target: 75% within ±0.5 grade

**UI Display:**
```
📷 Condition Analysis
┌──────────────────────────┐
│ Estimated Grade: PSA 8   │
│ Confidence: 82%          │
│                          │
│ ✅ Centering: Excellent  │
│ ⚠️  Edges: Minor wear    │
│ ✅ Corners: Sharp        │
│ ✅ Surface: Clean        │
│                          │
│ Raw: $45                 │
│ PSA 8: $85 (+89%)        │
│ PSA 9: $155 (+244%)      │
│ PSA 10: $420 (+833%)     │
└──────────────────────────┘

💡 Grading Recommendation:
Worth grading if centering improves slightly.
Expected return: +$40 profit after fees.
```

---

### 14.6 SaaS Monetization Features
**Priority:** High (Revenue)  
**Time:** 8 hours

**Pricing Tiers:**

**Free Tier:**
- 50 cards/month
- Basic price lookup
- CSV export only
- Ads displayed

**Pro ($4.99/month):**
- Unlimited scans
- Portfolio tracking
- All export formats
- No ads
- Price alerts

**Premium ($9.99/month):**
- All Pro features
- Grading estimates
- Multi-marketplace comparison
- Historical price charts
- API access
- Priority support

**Features:**
```python
class SubscriptionManager:
    def check_scan_limit(self, user_id: str) -> bool:
        """Enforce tier limits"""
        
    def unlock_feature(self, user_id: str, feature: str) -> bool:
        """Check feature access"""
        
    def send_upgrade_prompt(self, user_id: str):
        """Suggest upgrade at limit"""
```

**Payment Integration:**
- Stripe for subscriptions
- PayPal for one-time purchases
- Google Play / App Store billing (mobile)

**Analytics:**
```python
# Track for product decisions
- Most used features
- Conversion funnel
- Churn reasons
- Feature adoption rates
```

---

### 14.7 Collaboration & Sharing
**Priority:** Medium (Viral growth)  
**Time:** 6 hours

**Features:**

**1. Public Portfolios**
- Share portfolio link: `pricelens.app/u/jakecob`
- View-only access
- Social proof: "1,247 cards, $12,450"

**2. Compare Collections**
- Compare your portfolio vs friends
- "Trading opportunities" suggestion
- Shared want lists

**3. Marketplace Integration**
- "List for sale" button → Auto-post to eBay/Mercari
- Generate listing with photos & description
- Track sold items

---

## Updated Timeline

| Phase | Duration | Key Deliverable |
|-------|----------|-----------------|
| 0-9 | Week 1-9 | ✅ **Complete** (Web app + detection) |
| 10 | Week 10-11 | Running total, audio, currency |
| 11 | Week 11-12 | Multi-language detection |
| 12 | Week 12-13 | Product detection (packs/ETBs) |
| 13 | Week 13-16 | Flutter mobile app (iOS/Android) |
| 14 | Week 16-18 | **Portfolio & SaaS features** 💰 |

**Total Time to SaaS MVP:** 18 weeks (~4.5 months)

---

| Phase | Duration | Key Deliverable |
|-------|----------|-----------------|
| 0-9 | Week 1-9 | ✅ **Complete** (Web app + detection) |
| 10 | Week 10-11 | Running total, audio, currency |
| 11 | Week 11-12 | Multi-language detection |
| 12 | Week 12-13 | Product detection (packs/ETBs) |
| 13 | Week 13-16 | Flutter mobile app (iOS/Android) |

**Total Time to V3:** 16 weeks (~4 months)

---

## Feature Priority Matrix

| Feature | Impact | Effort | Priority |
|---------|--------|--------|----------|
| Running Total | High | Low | 🔥 NOW |
| Multi-Currency | High | Medium | 🔥 NOW |
| Audio Feedback | Medium | Low | ✅ Soon |
| Set Display | Medium | Low | ✅ Soon |
| Multi-Language | Medium | High | 📅 Later |
| Product Detection | High | High | 📅 Later |
| Flutter App | Very High | Very High | 🎯 Major |

---

## Next Steps

### This Week
1. [ ] Implement running total display
2. [ ] Add cash register sound effect
3. [ ] Integrate currency converter API
4. [ ] Show set information on overlay

### Next Week
1. [ ] Start Flutter project setup
2. [ ] Research multi-language card databases
3. [ ] Collect product images for YOLO training
4. [ ] Design mobile UI mockups

---

## Technical Dependencies

### New APIs
- **ExchangeRate-API** (Free tier: 1500 requests/month)
- **Pokemon TCG API** (Sealed products endpoint)

### New Libraries
```toml
# Python
forex-python = ">=1.8"
playsound = ">=1.3"

# Flutter
dependencies:
  camera: ^0.10.0
  http: ^1.1.0
  audioplayers: ^5.0.0
  provider: ^6.0.0
  shared_preferences: ^2.0.0
```

---

*Implementation Plan V3*  
*Created: November 24, 2025*  
*Estimated Total Development Time: 16 weeks*
