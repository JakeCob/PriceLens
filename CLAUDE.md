# POKEMON CARD PRICE OVERLAY - CLAUDE CODE PROJECT GUIDE

## 📋 PROJECT OVERVIEW

### Purpose
Real-time computer vision system that detects Pokemon trading cards through a camera feed and overlays live market price information directly onto the video stream.

### Core Value Proposition
- **For Collectors**: Instant price lookups without manual searching
- **For Traders**: Quick market value assessment during trades
- **For Sellers**: Real-time pricing for inventory management
- **For Buyers**: Price verification during purchases

### Technical Achievement
Combines state-of-the-art YOLO11 object detection with feature matching algorithms and real-time price APIs to create an augmented reality pricing experience.

---

## 🎯 PROJECT GOALS

### Primary Objectives
1. **Detect** Pokemon cards in live camera feed with >95% accuracy
2. **Identify** specific card names and sets using computer vision
3. **Fetch** real-time market prices from multiple sources
4. **Overlay** price information seamlessly on video stream
5. **Maintain** 30 FPS performance on consumer hardware

### Success Metrics
- Detection accuracy: >95% for front-facing cards
- Identification accuracy: >90% for cards in database
- Processing latency: <100ms end-to-end
- Frame rate: ≥30 FPS
- Multi-card support: 4-6 cards simultaneously

---

## 🏗️ SYSTEM ARCHITECTURE

### High-Level Flow
```
Camera Feed → Frame Capture → Card Detection (YOLO11) → 
Card Identification (Feature Matching) → Price Lookup (API) → 
Overlay Rendering → Display
```

### Component Breakdown

#### 1. VIDEO INPUT LAYER
- **Technology**: OpenCV VideoCapture
- **Input**: Webcam/USB camera (720p minimum, 1080p recommended)
- **Frame Processing**: 30 FPS capture rate
- **Pre-processing**: Resize, normalize, color correction

#### 2. DETECTION PIPELINE
- **Model**: Ultralytics YOLO11n (nano variant for speed)
- **Task**: Object detection - locate rectangular card regions
- **Output**: Bounding boxes [x, y, width, height] + confidence scores
- **Fallback**: If YOLO fails, use Canny edge detection + contour finding

#### 3. IDENTIFICATION PIPELINE
Two-method approach (hybrid system):

**Method A: Deep Learning Classification** (Future enhancement)
- Fine-tuned YOLO11 classification head
- Direct card ID prediction
- Fast but requires extensive training data

**Method B: Feature Matching** (Primary implementation)
- ORB keypoint detection (1000 features)
- BEBLID descriptor computation (14% better than ORB)
- FLANN-based matching against card database
- Lowe's ratio test for filtering (threshold: 0.75)
- Homography verification for geometric consistency

#### 4. PRICE DATA LAYER
- **Primary API**: PokemonPriceTracker API
- **Fallback API**: Pokemon TCG API (free tier)
- **Caching Strategy**: 5-minute TTL cache
- **Async Fetching**: Non-blocking API calls
- **Rate Limiting**: 60 requests/minute maximum

#### 5. OVERLAY RENDERING
- **Technology**: OpenCV drawing functions + alpha blending
- **Components**: Bounding boxes, semi-transparent panels, text overlays
- **Information Displayed**: 
  - Card name + set
  - Market price (TCGPlayer, eBay, CardMarket)
  - Price trend indicators
  - Graded card prices (PSA 10/9/8)
  - Last updated timestamp

---

## 💻 TECHNICAL STACK

### Core Dependencies
```python
# Computer Vision & Machine Learning
opencv-python==4.8.1.78
opencv-contrib-python==4.8.1.78  # For BEBLID descriptor
ultralytics==8.3.0                # YOLO11
torch==2.1.0
torchvision==0.16.0

# Numerical Computing
numpy==1.24.3
scipy==1.11.3

# API & Web (Optional - for web version)
fastapi==0.104.1
uvicorn[standard]==0.24.0
websockets==12.0
aiohttp==3.9.1

# Price Data APIs
requests==2.31.0
python-dotenv==1.0.0

# Utilities
pillow==10.1.0
pyyaml==6.0.1

# Performance Optimization
numba==0.58.1  # JIT compilation for feature matching
```

### Development Tools
```bash
# Code Quality
black==23.11.0
flake8==6.1.0
mypy==1.7.0
pytest==7.4.3
pytest-cov==4.1.0

# Documentation
mkdocs==1.5.3
mkdocs-material==9.4.14
```

### Hardware Requirements

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| CPU | Intel i5-8th gen / AMD Ryzen 5 | Intel i7-10th gen / AMD Ryzen 7 | Intel i9 / AMD Ryzen 9 |
| RAM | 8GB | 16GB | 32GB |
| GPU | Integrated | NVIDIA GTX 1060 (6GB) | NVIDIA RTX 3060+ |
| Camera | 720p webcam | 1080p USB camera | 4K camera |
| Storage | 5GB available | 10GB SSD | 20GB NVMe SSD |

---

## 📂 PROJECT STRUCTURE

```
pokemon-card-price-overlay/
│
├── README.md                      # User-facing documentation
├── CLAUDE.md                      # This file - comprehensive guide
├── requirements.txt               # Python dependencies
├── requirements-dev.txt           # Development dependencies
├── setup.py                       # Package installation
├── .env.example                   # Environment variables template
├── .gitignore                     # Git ignore rules
├── config.yaml                    # Application configuration
│
├── src/                           # Main source code
│   ├── __init__.py
│   │
│   ├── detection/                 # Card detection module
│   │   ├── __init__.py
│   │   ├── yolo_detector.py      # YOLO11 implementation
│   │   ├── edge_detector.py      # Fallback edge-based detection
│   │   └── detector_base.py      # Abstract detector interface
│   │
│   ├── identification/            # Card identification module
│   │   ├── __init__.py
│   │   ├── feature_matcher.py    # ORB/BEBLID matching
│   │   ├── card_database.py      # Card database management
│   │   ├── classifier.py         # Future: DL classification
│   │   └── identifier_base.py    # Abstract identifier interface
│   │
│   ├── api/                       # Price API clients
│   │   ├── __init__.py
│   │   ├── price_api.py          # Main API client
│   │   ├── pokemonpricetracker.py # PokemonPriceTracker client
│   │   ├── pokemontcg_api.py     # Pokemon TCG API client
│   │   └── cache.py              # Caching layer
│   │
│   ├── overlay/                   # Rendering module
│   │   ├── __init__.py
│   │   ├── renderer.py           # Main overlay renderer
│   │   ├── components.py         # UI components (boxes, panels)
│   │   └── styles.py             # Color schemes and styles
│   │
│   ├── utils/                     # Utility functions
│   │   ├── __init__.py
│   │   ├── image_processing.py   # Image manipulation helpers
│   │   ├── performance.py        # Performance monitoring
│   │   ├── logging_config.py     # Logging setup
│   │   └── validators.py         # Input validation
│   │
│   ├── main.py                    # Main application entry point
│   └── config.py                  # Configuration management
│
├── models/                        # Model files
│   ├── yolo11n.pt                # Pre-trained YOLO11 nano
│   ├── pokemon_cards_yolo11.pt   # Fine-tuned model (future)
│   └── model_info.yaml           # Model metadata
│
├── data/                          # Data directory
│   ├── card_database/            # Reference card images
│   │   ├── base_set/             # Organized by set
│   │   ├── jungle/
│   │   ├── fossil/
│   │   └── index.json            # Card metadata index
│   │
│   ├── features/                 # Pre-computed feature descriptors
│   │   ├── base_set_features.pkl
│   │   └── feature_index.pkl
│   │
│   └── train/                    # Training data (optional)
│       ├── images/
│       └── labels/
│
├── tests/                         # Test suite
│   ├── __init__.py
│   ├── test_detection.py         # Detection tests
│   ├── test_identification.py    # Identification tests
│   ├── test_api.py               # API client tests
│   ├── test_overlay.py           # Rendering tests
│   ├── test_integration.py       # End-to-end tests
│   └── fixtures/                 # Test fixtures and sample data
│
├── notebooks/                     # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_yolo_training.ipynb
│   ├── 03_feature_matching_analysis.ipynb
│   └── 04_performance_benchmarking.ipynb
│
├── scripts/                       # Utility scripts
│   ├── download_models.py        # Download pre-trained models
│   ├── build_card_database.py    # Create card database
│   ├── compute_features.py       # Pre-compute card features
│   └── benchmark.py              # Performance benchmarking
│
├── docs/                          # Documentation
│   ├── architecture.md           # Architecture details
│   ├── api_reference.md          # API documentation
│   ├── development_guide.md      # Development guidelines
│   └── deployment.md             # Deployment instructions
│
└── docker/                        # Docker configuration
    ├── Dockerfile                # Main Dockerfile
    ├── docker-compose.yml        # Docker Compose setup
    └── .dockerignore             # Docker ignore rules
```

---

## 🔧 IMPLEMENTATION GUIDE

### Phase 1: Project Setup (Week 1)

#### Step 1.1: Environment Setup
```bash
# Create project directory
mkdir pokemon-card-price-overlay
cd pokemon-card-price-overlay

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Verify installations
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "from ultralytics import YOLO; print('YOLO OK')"
```

#### Step 1.2: Download YOLO11 Model
```python
# scripts/download_models.py
from ultralytics import YOLO
import os

def download_yolo11():
    """Download YOLO11 nano model"""
    print("Downloading YOLO11n model...")
    model = YOLO('yolo11n.pt')  # Downloads automatically
    
    # Save to models directory
    os.makedirs('models', exist_ok=True)
    model.save('models/yolo11n.pt')
    print("✓ Model downloaded to models/yolo11n.pt")

if __name__ == "__main__":
    download_yolo11()
```

#### Step 1.3: Configuration Setup
```yaml
# config.yaml
app:
  name: "Pokemon Card Price Overlay"
  version: "0.1.0"
  debug: true

camera:
  source: 0  # 0 for default webcam, or video file path
  width: 1280
  height: 720
  fps: 30

detection:
  model_path: "models/yolo11n.pt"
  confidence_threshold: 0.5
  iou_threshold: 0.45
  max_detections: 6

identification:
  method: "feature_matching"  # "feature_matching" or "classification"
  min_matches: 10
  match_ratio_threshold: 0.75
  
feature_matching:
  detector: "ORB"  # ORB, SIFT, AKAZE
  descriptor: "BEBLID"  # BEBLID, ORB, SIFT
  n_features: 1000
  match_method: "FLANN"  # FLANN or BF (Brute Force)

api:
  primary: "pokemonpricetracker"
  fallback: "pokemontcg"
  cache_ttl: 300  # 5 minutes
  rate_limit: 60  # requests per minute
  timeout: 5  # seconds

overlay:
  show_bounding_box: true
  show_confidence: true
  show_prices: true
  show_trends: true
  panel_opacity: 0.8
  font_scale: 0.6
  
performance:
  use_gpu: true  # Use GPU if available
  num_threads: 4
  frame_skip: 0  # Skip every N frames (0 = no skip)
```

### Phase 2: Core Detection (Week 2)

#### Step 2.1: YOLO Detector Implementation
```python
# src/detection/yolo_detector.py
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class YOLOCardDetector:
    """Card detection using YOLO11 object detection"""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45, device: str = 'auto'):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference ('cpu', 'cuda', 'auto')
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = self._get_device(device)
        
        logger.info(f"YOLO detector initialized on {self.device}")
    
    def _get_device(self, device: str) -> str:
        """Determine device for inference"""
        if device == 'auto':
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def detect(self, frame: np.ndarray) -> List[dict]:
        """
        Detect cards in frame
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            List of detections, each containing:
                - bbox: [x1, y1, x2, y2]
                - confidence: float
                - class_id: int
                - class_name: str
        """
        # Run inference
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )
        
        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Extract box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                # Validate detection is card-like (rectangular)
                width = x2 - x1
                height = y2 - y1
                aspect_ratio = width / height if height > 0 else 0
                
                # Pokemon cards are approximately 2.5" x 3.5" (aspect ~0.71)
                if 0.5 < aspect_ratio < 0.9:
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name,
                        'aspect_ratio': aspect_ratio
                    })
        
        logger.debug(f"Detected {len(detections)} cards")
        return detections
    
    def extract_card_regions(self, frame: np.ndarray, 
                            detections: List[dict]) -> List[np.ndarray]:
        """Extract card regions from frame"""
        card_regions = []
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Add small padding
            padding = 5
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)
            
            # Extract region
            card_region = frame[y1:y2, x1:x2].copy()
            card_regions.append(card_region)
        
        return card_regions


# Example usage
if __name__ == "__main__":
    detector = YOLOCardDetector("models/yolo11n.pt")
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect cards
        detections = detector.detect(frame)
        
        # Draw bounding boxes
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('Card Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
```

#### Step 2.2: Edge Detection Fallback
```python
# src/detection/edge_detector.py
import cv2
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


class EdgeCardDetector:
    """Fallback card detection using edge detection and contours"""
    
    def __init__(self, min_area: int = 5000, max_area: int = 200000):
        self.min_area = min_area
        self.max_area = max_area
    
    def detect(self, frame: np.ndarray) -> List[dict]:
        """Detect cards using edge detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Morphological closing to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue
            
            # Approximate polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if roughly rectangular (4 corners)
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / h if h > 0 else 0
                
                # Check aspect ratio
                if 0.5 < aspect_ratio < 0.9:
                    detections.append({
                        'bbox': [x, y, x+w, y+h],
                        'confidence': 0.7,  # Fixed confidence
                        'class_name': 'card',
                        'aspect_ratio': aspect_ratio
                    })
        
        return detections
```

### Phase 3: Card Identification (Week 3-4)

#### Step 3.1: Card Database Setup
```python
# scripts/build_card_database.py
import os
import json
import cv2
import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CardDatabaseBuilder:
    """Build card reference database from Pokemon TCG API"""
    
    def __init__(self, output_dir: str = "data/card_database"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.api_url = "https://api.pokemontcg.io/v2"
        
    def fetch_cards(self, set_id: str, limit: int = 100):
        """Fetch cards from Pokemon TCG API"""
        url = f"{self.api_url}/cards"
        params = {
            'q': f'set.id:{set_id}',
            'pageSize': limit
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        return response.json()['data']
    
    def download_card_image(self, card_data: dict, set_dir: Path):
        """Download single card image"""
        try:
            card_id = card_data['id']
            card_name = card_data['name']
            image_url = card_data['images']['large']
            
            # Download image
            response = requests.get(image_url)
            response.raise_for_status()
            
            # Save image
            image_path = set_dir / f"{card_id}.jpg"
            with open(image_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded: {card_name} ({card_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {card_id}: {e}")
            return False
    
    def build_set_database(self, set_id: str, set_name: str):
        """Build database for a specific card set"""
        logger.info(f"Building database for {set_name}...")
        
        # Create set directory
        set_dir = self.output_dir / set_id
        set_dir.mkdir(exist_ok=True)
        
        # Fetch cards
        cards = self.fetch_cards(set_id)
        logger.info(f"Found {len(cards)} cards in {set_name}")
        
        # Download images
        successful = 0
        metadata = []
        
        for card in cards:
            if self.download_card_image(card, set_dir):
                metadata.append({
                    'id': card['id'],
                    'name': card['name'],
                    'number': card.get('number', ''),
                    'rarity': card.get('rarity', ''),
                    'set_id': set_id,
                    'set_name': set_name,
                    'image_path': f"{set_id}/{card['id']}.jpg"
                })
                successful += 1
        
        # Save metadata
        metadata_path = set_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ Downloaded {successful}/{len(cards)} cards")
        
    def build_index(self):
        """Build master index of all cards"""
        index = []
        
        for set_dir in self.output_dir.iterdir():
            if set_dir.is_dir():
                metadata_path = set_dir / 'metadata.json'
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        cards = json.load(f)
                        index.extend(cards)
        
        # Save master index
        index_path = self.output_dir / 'index.json'
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
        
        logger.info(f"✓ Built index with {len(index)} total cards")


if __name__ == "__main__":
    builder = CardDatabaseBuilder()
    
    # Build database for popular sets
    sets = [
        ('base1', 'Base Set'),
        ('base2', 'Jungle'),
        ('base3', 'Fossil'),
        ('base4', 'Base Set 2'),
    ]
    
    for set_id, set_name in sets:
        builder.build_set_database(set_id, set_name)
    
    # Build master index
    builder.build_index()
    
    logger.info("✓ Card database build complete!")
```

#### Step 3.2: Feature Matcher Implementation
```python
# src/identification/feature_matcher.py
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FeatureMatcher:
    """Card identification using feature matching"""
    
    def __init__(self, 
                 n_features: int = 1000,
                 match_threshold: float = 0.75,
                 min_matches: int = 10):
        """
        Initialize feature matcher
        
        Args:
            n_features: Number of features to detect
            match_threshold: Lowe's ratio test threshold
            min_matches: Minimum matches for valid identification
        """
        self.n_features = n_features
        self.match_threshold = match_threshold
        self.min_matches = min_matches
        
        # Initialize ORB detector
        self.detector = cv2.ORB_create(nfeatures=n_features)
        
        # Initialize BEBLID descriptor
        # BEBLID is 14% better than ORB descriptor
        self.descriptor = cv2.xfeatures2d.BEBLID_create(0.75)
        
        # Initialize FLANN matcher for binary descriptors
        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH,
            table_number=12,
            key_size=20,
            multi_probe_level=2
        )
        search_params = dict(checks=50)
        
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Card database
        self.card_features = {}
        self.card_metadata = {}
        
        logger.info("Feature matcher initialized")
    
    def compute_features(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Compute features for an image
        
        Args:
            image: Input image (BGR)
            
        Returns:
            keypoints: List of keypoints
            descriptors: Feature descriptors
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect keypoints
        keypoints = self.detector.detect(gray, None)
        
        # Compute BEBLID descriptors
        keypoints, descriptors = self.descriptor.compute(gray, keypoints)
        
        return keypoints, descriptors
    
    def load_card_database(self, database_path: str):
        """Load pre-computed card features"""
        db_path = Path(database_path)
        
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {database_path}")
        
        with open(db_path, 'rb') as f:
            data = pickle.load(f)
            self.card_features = data['features']
            self.card_metadata = data['metadata']
        
        logger.info(f"Loaded {len(self.card_features)} cards from database")
    
    def match_card(self, image: np.ndarray, 
                   top_k: int = 3) -> List[Dict]:
        """
        Match card image against database
        
        Args:
            image: Card image to identify
            top_k: Return top K matches
            
        Returns:
            List of matches with card_id, name, confidence
        """
        # Compute features for query image
        keypoints, descriptors = self.compute_features(image)
        
        if descriptors is None or len(descriptors) == 0:
            logger.warning("No features detected in image")
            return []
        
        # Match against all cards in database
        match_results = []
        
        for card_id, card_desc in self.card_features.items():
            # Match descriptors using KNN (k=2 for Lowe's ratio test)
            matches = self.matcher.knnMatch(descriptors, card_desc, k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.match_threshold * n.distance:
                        good_matches.append(m)
            
            # Check if we have enough matches
            if len(good_matches) >= self.min_matches:
                # Calculate match confidence
                confidence = len(good_matches) / len(keypoints)
                
                match_results.append({
                    'card_id': card_id,
                    'name': self.card_metadata[card_id]['name'],
                    'set_name': self.card_metadata[card_id]['set_name'],
                    'num_matches': len(good_matches),
                    'confidence': confidence,
                    'metadata': self.card_metadata[card_id]
                })
        
        # Sort by number of matches (primary) and confidence (secondary)
        match_results.sort(
            key=lambda x: (x['num_matches'], x['confidence']),
            reverse=True
        )
        
        # Return top K matches
        return match_results[:top_k]
    
    def verify_with_homography(self, 
                               query_kp, query_desc,
                               train_kp, train_desc,
                               matches) -> bool:
        """
        Verify match using homography (geometric consistency)
        
        Returns:
            True if homography is valid, False otherwise
        """
        if len(matches) < 4:
            return False
        
        # Extract matched point coordinates
        src_pts = np.float32([query_kp[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([train_kp[m.trainIdx].pt for m in matches])
        
        src_pts = src_pts.reshape(-1, 1, 2)
        dst_pts = dst_pts.reshape(-1, 1, 2)
        
        # Find homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is None:
            return False
        
        # Count inliers
        inliers = np.sum(mask)
        inlier_ratio = inliers / len(matches)
        
        # Valid if >50% inliers
        return inlier_ratio > 0.5


# Pre-compute features for database
class DatabaseBuilder:
    """Build feature database from card images"""
    
    def __init__(self, card_dir: str, output_path: str):
        self.card_dir = Path(card_dir)
        self.output_path = Path(output_path)
        self.matcher = FeatureMatcher()
        
    def build(self):
        """Build feature database"""
        import json
        
        features_db = {}
        metadata_db = {}
        
        # Load card index
        index_path = self.card_dir / 'index.json'
        with open(index_path, 'r') as f:
            cards = json.load(f)
        
        logger.info(f"Processing {len(cards)} cards...")
        
        for card in cards:
            card_id = card['id']
            image_path = self.card_dir / card['image_path']
            
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                continue
            
            # Load image
            image = cv2.imread(str(image_path))
            
            # Compute features
            _, descriptors = self.matcher.compute_features(image)
            
            if descriptors is not None:
                features_db[card_id] = descriptors
                metadata_db[card_id] = card
                logger.info(f"Processed: {card['name']}")
        
        # Save database
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_path, 'wb') as f:
            pickle.dump({
                'features': features_db,
                'metadata': metadata_db
            }, f)
        
        logger.info(f"✓ Database saved to {self.output_path}")
        logger.info(f"✓ Total cards: {len(features_db)}")


if __name__ == "__main__":
    # Build feature database
    builder = DatabaseBuilder(
        card_dir="data/card_database",
        output_path="data/features/card_features.pkl"
    )
    builder.build()
```

### Phase 4: Price API Integration (Week 4)

#### Step 4.1: Price API Client
```python
# src/api/price_api.py
import requests
import time
from typing import Optional, Dict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class PriceCache:
    """Simple in-memory cache with TTL"""
    
    def __init__(self, ttl: int = 300):
        self.cache = {}
        self.ttl = ttl  # Time to live in seconds
    
    def get(self, key: str) -> Optional[Dict]:
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Dict):
        self.cache[key] = (value, time.time())
    
    def clear(self):
        self.cache.clear()


class PokemonPriceAPI:
    """Client for PokemonPriceTracker API"""
    
    def __init__(self, api_key: str, cache_ttl: int = 300):
        self.api_key = api_key
        self.base_url = "https://www.pokemonpricetracker.com/api/v2"
        self.cache = PriceCache(ttl=cache_ttl)
        self.rate_limiter = RateLimiter(rate_limit=60)  # 60 req/min
    
    def get_card_price(self, card_id: str) -> Optional[Dict]:
        """
        Get price data for a card
        
        Args:
            card_id: Pokemon TCG card ID (e.g., 'base1-4')
            
        Returns:
            Dict with price data or None
        """
        # Check cache first
        cached = self.cache.get(card_id)
        if cached:
            logger.debug(f"Cache hit for {card_id}")
            return cached
        
        # Rate limiting
        self.rate_limiter.wait()
        
        # Make API request
        url = f"{self.base_url}/cards"
        params = {'search': card_id, 'includeHistory': 'false'}
        headers = {'Authorization': f'Bearer {self.api_key}'}
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            
            if data and len(data) > 0:
                card_data = data[0]
                
                # Parse price data
                price_info = {
                    'card_id': card_id,
                    'name': card_data.get('name'),
                    'set': card_data.get('set'),
                    'prices': {
                        'tcgplayer_market': card_data.get('tcgplayer_market'),
                        'tcgplayer_low': card_data.get('tcgplayer_low'),
                        'tcgplayer_mid': card_data.get('tcgplayer_mid'),
                        'tcgplayer_high': card_data.get('tcgplayer_high'),
                    },
                    'graded_prices': card_data.get('graded_prices', {}),
                    'last_updated': datetime.now().isoformat()
                }
                
                # Cache result
                self.cache.set(card_id, price_info)
                
                return price_info
            
            return None
            
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
    
    def get_batch_prices(self, card_ids: list) -> Dict[str, Dict]:
        """Get prices for multiple cards"""
        results = {}
        
        for card_id in card_ids:
            price_data = self.get_card_price(card_id)
            if price_data:
                results[card_id] = price_data
        
        return results


class RateLimiter:
    """Simple rate limiter"""
    
    def __init__(self, rate_limit: int = 60):
        self.rate_limit = rate_limit  # requests per minute
        self.requests = []
    
    def wait(self):
        """Wait if necessary to respect rate limit"""
        now = time.time()
        
        # Remove old requests (>1 minute ago)
        self.requests = [t for t in self.requests if now - t < 60]
        
        # Check if we need to wait
        if len(self.requests) >= self.rate_limit:
            sleep_time = 60 - (now - self.requests[0])
            if sleep_time > 0:
                logger.debug(f"Rate limit reached, sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
        
        # Record this request
        self.requests.append(time.time())


# Fallback: Pokemon TCG API (free tier)
class PokemonTCGAPI:
    """Client for official Pokemon TCG API (no price data)"""
    
    def __init__(self):
        self.base_url = "https://api.pokemontcg.io/v2"
    
    def get_card_info(self, card_id: str) -> Optional[Dict]:
        """Get card information (no prices)"""
        url = f"{self.base_url}/cards/{card_id}"
        
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            
            data = response.json()['data']
            
            return {
                'card_id': card_id,
                'name': data.get('name'),
                'set': data.get('set', {}).get('name'),
                'number': data.get('number'),
                'rarity': data.get('rarity'),
                'image_url': data.get('images', {}).get('large')
            }
            
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
```

### Phase 5: Overlay Rendering (Week 5)

#### Step 5.1: Overlay Renderer
```python
# src/overlay/renderer.py
import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class OverlayRenderer:
    """Render price overlays on video frame"""
    
    def __init__(self, 
                 panel_opacity: float = 0.8,
                 font_scale: float = 0.6,
                 show_confidence: bool = True):
        self.panel_opacity = panel_opacity
        self.font_scale = font_scale
        self.show_confidence = show_confidence
        
        # Color scheme (BGR format)
        self.colors = {
            'box': (0, 255, 0),           # Green
            'box_low_conf': (0, 165, 255), # Orange
            'panel_bg': (50, 50, 50),     # Dark gray
            'text': (255, 255, 255),      # White
            'price_up': (0, 255, 0),      # Green
            'price_down': (0, 0, 255),    # Red
            'price_neutral': (255, 255, 255) # White
        }
    
    def render_frame(self, 
                    frame: np.ndarray,
                    detections: List[Dict],
                    identifications: Dict[int, Dict],
                    prices: Dict[str, Dict]) -> np.ndarray:
        """
        Render overlays on frame
        
        Args:
            frame: Input frame
            detections: List of detected cards
            identifications: Dict mapping detection index to card info
            prices: Dict mapping card_id to price data
            
        Returns:
            Frame with overlays
        """
        output = frame.copy()
        
        for idx, detection in enumerate(detections):
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            # Get identification if available
            card_info = identifications.get(idx)
            
            # Get price data if available
            price_data = None
            if card_info:
                card_id = card_info.get('card_id')
                price_data = prices.get(card_id)
            
            # Draw bounding box
            self._draw_bounding_box(output, bbox, confidence)
            
            # Draw info panel if we have identification
            if card_info and price_data:
                self._draw_info_panel(output, bbox, card_info, price_data)
            elif card_info:
                # Draw minimal info (no price)
                self._draw_minimal_panel(output, bbox, card_info)
        
        return output
    
    def _draw_bounding_box(self, 
                          frame: np.ndarray,
                          bbox: List[int],
                          confidence: float):
        """Draw bounding box around detected card"""
        x1, y1, x2, y2 = bbox
        
        # Choose color based on confidence
        if confidence > 0.7:
            color = self.colors['box']
            thickness = 2
        else:
            color = self.colors['box_low_conf']
            thickness = 1
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw confidence if enabled
        if self.show_confidence:
            conf_text = f"{confidence:.2f}"
            cv2.putText(frame, conf_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def _draw_info_panel(self,
                        frame: np.ndarray,
                        bbox: List[int],
                        card_info: Dict,
                        price_data: Dict):
        """Draw information panel with prices"""
        x1, y1, x2, y2 = bbox
        
        # Panel dimensions
        panel_width = 250
        panel_height = 150
        
        # Position panel (to the right of card)
        panel_x = x2 + 10
        panel_y = y1
        
        # Adjust if panel goes off screen
        if panel_x + panel_width > frame.shape[1]:
            panel_x = x1 - panel_width - 10
        
        if panel_y + panel_height > frame.shape[0]:
            panel_y = frame.shape[0] - panel_height - 10
        
        # Create semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            self.colors['panel_bg'],
            -1
        )
        
        # Blend with original frame
        cv2.addWeighted(overlay, self.panel_opacity, frame, 1 - self.panel_opacity, 0, frame)
        
        # Add text content
        text_x = panel_x + 10
        text_y = panel_y + 25
        line_height = 25
        
        # Card name
        name = card_info.get('name', 'Unknown')
        self._draw_text(frame, name, text_x, text_y, scale=0.6, bold=True)
        text_y += line_height
        
        # Set name
        set_name = card_info.get('set_name', '')
        self._draw_text(frame, set_name, text_x, text_y, scale=0.4)
        text_y += line_height
        
        # Market price
        prices = price_data.get('prices', {})
        market_price = prices.get('tcgplayer_market')
        
        if market_price:
            price_text = f"Market: ${market_price:.2f}"
            self._draw_text(frame, price_text, text_x, text_y, scale=0.5)
            text_y += line_height
        
        # Price range
        low_price = prices.get('tcgplayer_low')
        high_price = prices.get('tcgplayer_high')
        
        if low_price and high_price:
            range_text = f"Range: ${low_price:.2f} - ${high_price:.2f}"
            self._draw_text(frame, range_text, text_x, text_y, scale=0.4)
            text_y += line_height
        
        # PSA 10 price if available
        graded = price_data.get('graded_prices', {})
        psa10 = graded.get('psa10')
        
        if psa10:
            psa_text = f"PSA 10: ${psa10:.2f}"
            self._draw_text(frame, psa_text, text_x, text_y, scale=0.4)
    
    def _draw_minimal_panel(self,
                           frame: np.ndarray,
                           bbox: List[int],
                           card_info: Dict):
        """Draw minimal panel without price data"""
        x1, y1, x2, y2 = bbox
        
        name = card_info.get('name', 'Unknown')
        set_name = card_info.get('set_name', '')
        
        # Draw text above card
        text = f"{name} - {set_name}"
        
        # Text background
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - 10),
            (x1 + text_width + 10, y1),
            self.colors['panel_bg'],
            -1
        )
        
        cv2.putText(
            frame, text,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.colors['text'],
            1
        )
    
    def _draw_text(self, 
                   frame: np.ndarray,
                   text: str,
                   x: int,
                   y: int,
                   scale: float = 0.5,
                   bold: bool = False):
        """Helper to draw text"""
        thickness = 2 if bold else 1
        
        cv2.putText(
            frame, text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            self.colors['text'],
            thickness
        )
```

### Phase 6: Main Application (Week 5-6)

#### Step 6.1: Main Application Loop
```python
# src/main.py
import cv2
import yaml
import argparse
import logging
from pathlib import Path
import time
from collections import deque

from detection.yolo_detector import YOLOCardDetector
from identification.feature_matcher import FeatureMatcher
from api.price_api import PokemonPriceAPI
from overlay.renderer import OverlayRenderer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PokemonCardPriceOverlay:
    """Main application class"""
    
    def __init__(self, config_path: str = "config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self._init_detector()
        self._init_identifier()
        self._init_price_api()
        self._init_renderer()
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.frame_count = 0
        
        logger.info("Application initialized")
    
    def _init_detector(self):
        """Initialize card detector"""
        detection_config = self.config['detection']
        
        self.detector = YOLOCardDetector(
            model_path=detection_config['model_path'],
            conf_threshold=detection_config['confidence_threshold'],
            iou_threshold=detection_config['iou_threshold']
        )
        
        logger.info("Detector initialized")
    
    def _init_identifier(self):
        """Initialize card identifier"""
        id_config = self.config['identification']
        fm_config = self.config['feature_matching']
        
        self.identifier = FeatureMatcher(
            n_features=fm_config['n_features'],
            match_threshold=id_config['match_ratio_threshold'],
            min_matches=id_config['min_matches']
        )
        
        # Load card database
        db_path = "data/features/card_features.pkl"
        if Path(db_path).exists():
            self.identifier.load_card_database(db_path)
        else:
            logger.warning(f"Feature database not found: {db_path}")
        
        logger.info("Identifier initialized")
    
    def _init_price_api(self):
        """Initialize price API client"""
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv('POKEMON_PRICE_API_KEY')
        
        if api_key:
            self.price_api = PokemonPriceAPI(
                api_key=api_key,
                cache_ttl=self.config['api']['cache_ttl']
            )
            logger.info("Price API initialized")
        else:
            logger.warning("No API key found, prices will not be available")
            self.price_api = None
    
    def _init_renderer(self):
        """Initialize overlay renderer"""
        overlay_config = self.config['overlay']
        
        self.renderer = OverlayRenderer(
            panel_opacity=overlay_config['panel_opacity'],
            font_scale=overlay_config['font_scale'],
            show_confidence=overlay_config['show_confidence']
        )
        
        logger.info("Renderer initialized")
    
    def process_frame(self, frame):
        """Process a single frame"""
        start_time = time.time()
        
        # Detect cards
        detections = self.detector.detect(frame)
        
        # Extract card regions
        card_regions = self.detector.extract_card_regions(frame, detections)
        
        # Identify cards
        identifications = {}
        for idx, region in enumerate(card_regions):
            matches = self.identifier.match_card(region, top_k=1)
            if matches:
                identifications[idx] = matches[0]
        
        # Fetch prices (only for newly identified cards)
        prices = {}
        if self.price_api:
            for card_info in identifications.values():
                card_id = card_info['card_id']
                if card_id not in prices:
                    price_data = self.price_api.get_card_price(card_id)
                    if price_data:
                        prices[card_id] = price_data
        
        # Render overlays
        output_frame = self.renderer.render_frame(
            frame, detections, identifications, prices
        )
        
        # Calculate FPS
        elapsed = time.time() - start_time
        self.fps_counter.append(1.0 / elapsed if elapsed > 0 else 0)
        
        # Draw FPS
        avg_fps = sum(self.fps_counter) / len(self.fps_counter)
        cv2.putText(
            output_frame,
            f"FPS: {avg_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        
        return output_frame
    
    def run(self):
        """Run main application loop"""
        # Open camera
        camera_source = self.config['camera']['source']
        cap = cv2.VideoCapture(camera_source)
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['height'])
        cap.set(cv2.CAP_PROP_FPS, self.config['camera']['fps'])
        
        if not cap.isOpened():
            logger.error(f"Failed to open camera: {camera_source}")
            return
        
        logger.info("Starting video capture...")
        logger.info("Press 'q' to quit, 's' to save screenshot")
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    logger.error("Failed to read frame")
                    break
                
                # Process frame
                output_frame = self.process_frame(frame)
                
                # Display
                cv2.imshow('Pokemon Card Price Overlay', output_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    logger.info("Quit requested")
                    break
                elif key == ord('s'):
                    # Save screenshot
                    filename = f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(filename, output_frame)
                    logger.info(f"Screenshot saved: {filename}")
                
                self.frame_count += 1
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            logger.info(f"Processed {self.frame_count} frames")


def main():
    parser = argparse.ArgumentParser(
        description='Pokemon Card Price Overlay'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Create and run application
    app = PokemonCardPriceOverlay(config_path=args.config)
    app.run()


if __name__ == "__main__":
    main()
```

---

## 🧪 TESTING STRATEGY

### Unit Tests
```python
# tests/test_detection.py
import pytest
import cv2
import numpy as np
from src.detection.yolo_detector import YOLOCardDetector


class TestYOLODetector:
    @pytest.fixture
    def detector(self):
        return YOLOCardDetector("models/yolo11n.pt")
    
    def test_detector_initialization(self, detector):
        assert detector is not None
        assert detector.model is not None
    
    def test_detect_single_card(self, detector):
        # Load test image
        image = cv2.imread("tests/fixtures/single_card.jpg")
        
        # Detect
        detections = detector.detect(image)
        
        # Verify
        assert len(detections) > 0
        assert 'bbox' in detections[0]
        assert 'confidence' in detections[0]
    
    def test_detect_multiple_cards(self, detector):
        image = cv2.imread("tests/fixtures/multiple_cards.jpg")
        detections = detector.detect(image)
        
        assert len(detections) >= 2
    
    def test_aspect_ratio_filtering(self, detector):
        image = cv2.imread("tests/fixtures/mixed_objects.jpg")
        detections = detector.detect(image)
        
        for det in detections:
            aspect_ratio = det['aspect_ratio']
            assert 0.5 < aspect_ratio < 0.9
```

### Integration Tests
```python
# tests/test_integration.py
import pytest
import cv2
from src.main import PokemonCardPriceOverlay


class TestEndToEnd:
    @pytest.fixture
    def app(self):
        return PokemonCardPriceOverlay("config.yaml")
    
    def test_full_pipeline(self, app):
        # Load test image
        frame = cv2.imread("tests/fixtures/charizard.jpg")
        
        # Process
        output = app.process_frame(frame)
        
        # Verify output
        assert output is not None
        assert output.shape == frame.shape
    
    def test_price_lookup(self, app):
        if app.price_api:
            price_data = app.price_api.get_card_price("base1-4")
            
            assert price_data is not None
            assert 'prices' in price_data
            assert price_data['name'] == 'Charizard'
```

---

## 🚀 DEPLOYMENT

### Docker Setup
```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Download models
RUN python scripts/download_models.py

# Run application
CMD ["python", "src/main.py"]
```

### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    container_name: pokemon-card-overlay
    devices:
      - /dev/video0:/dev/video0  # Camera access
    environment:
      - POKEMON_PRICE_API_KEY=${POKEMON_PRICE_API_KEY}
      - DISPLAY=${DISPLAY}
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - /tmp/.X11-unix:/tmp/.X11-unix
    network_mode: host
```

---

## 📊 PERFORMANCE OPTIMIZATION

### GPU Acceleration
```python
# Enable GPU for YOLO
detector = YOLOCardDetector("models/yolo11n.pt", device='cuda')

# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Multi-threading for API Calls
```python
import concurrent.futures

def fetch_prices_async(card_ids, price_api):
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(price_api.get_card_price, card_id): card_id
            for card_id in card_ids
        }
        
        results = {}
        for future in concurrent.futures.as_completed(futures):
            card_id = futures[future]
            try:
                results[card_id] = future.result()
            except Exception as e:
                logger.error(f"Error fetching {card_id}: {e}")
        
        return results
```

---

## 🎓 LEARNING RESOURCES

### Computer Vision
- [OpenCV Tutorial](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)
- [YOLO11 Documentation](https://docs.ultralytics.com/models/yolo11/)
- [Feature Matching Guide](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)

### Pokemon TCG APIs
- [Pokemon TCG API Docs](https://docs.pokemontcg.io/)
- [PokemonPriceTracker API](https://www.pokemonpricetracker.com/api-docs)

### Python Best Practices
- [Real Python](https://realpython.com/)
- [Python Design Patterns](https://refactoring.guru/design-patterns/python)

---

## 🐛 TROUBLESHOOTING

### Common Issues

**Issue: Low FPS**
- Solution: Use GPU acceleration, reduce image resolution, use YOLO11n (nano) model

**Issue: Poor detection accuracy**
- Solution: Ensure good lighting, hold cards flat, increase confidence threshold

**Issue: No cards identified**
- Solution: Build larger feature database, adjust match threshold, use better camera

**Issue: API rate limit**
- Solution: Increase cache TTL, reduce update frequency, use batch requests

---

## 📝 NEXT STEPS FOR IMPLEMENTATION

### Immediate Actions (Day 1-2)
1. Set up development environment
2. Install all dependencies
3. Download YOLO11 model
4. Test camera capture

### Week 1 Focus
1. Get basic YOLO detection working
2. Test with physical Pokemon cards
3. Implement bounding box visualization
4. Measure baseline performance

### Week 2-3 Focus
1. Build card database (start with 50-100 cards)
2. Implement feature matching
3. Test identification accuracy
4. Optimize matching parameters

### Week 4-5 Focus
1. Integrate price API
2. Implement overlay rendering
3. Add caching and rate limiting
4. Performance optimization

### Week 6+ Future Enhancements
1. Train custom YOLO model on Pokemon cards
2. Add mobile app support
3. Historical price tracking
4. Card condition assessment
5. Collection management features

---

## 🎯 SUCCESS CRITERIA

The project is considered successful when:
- ✓ Detects cards at 30 FPS
- ✓ Identifies cards with >90% accuracy
- ✓ Displays prices within 2 seconds
- ✓ Handles 4+ cards simultaneously
- ✓ Works in various lighting conditions
- ✓ Runs on consumer hardware

---

## 📄 LICENSE & CREDITS

This project uses:
- **YOLO11** by Ultralytics (AGPL-3.0)
- **OpenCV** (Apache 2.0)
- **Pokemon TCG API** (Free for non-commercial use)

Always respect Pokemon/Nintendo intellectual property and use responsibly.

---

## 🤝 CONTRIBUTION GUIDELINES

If extending this project:
1. Follow PEP 8 style guide
2. Add unit tests for new features
3. Update documentation
4. Optimize for performance
5. Consider edge cases

---

**END OF CLAUDE.md**

This document provides everything needed to understand, implement, and extend the Pokemon Card Price Overlay project. Good luck building! 🚀