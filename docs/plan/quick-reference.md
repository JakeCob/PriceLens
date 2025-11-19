# PriceLens Quick Reference Guide

Quick reference for common tasks and commands during development.

---

## Environment Setup

```bash
# Create virtual environment with uv
uv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies (fast!)
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt

# Verify installation
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from ultralytics import YOLO; print('YOLO OK')"
```

---

## Common Commands

### Download Models
```bash
python scripts/download_models.py
```

### Build Card Database
```bash
# Download specific sets (fast, ~100 cards)
python scripts/build_card_database.py --sets base1,jungle,fossil

# Download all sets (slow, ~5GB)
python scripts/build_card_database.py --all
```

### Compute Features
```bash
python scripts/compute_features.py
```

### Run Application
```bash
python src/main.py

# With custom config
python src/main.py --config custom_config.yaml

# Debug mode
python src/main.py --debug

# Disable GPU
python src/main.py --no-gpu

# Use specific camera
python src/main.py --camera 1
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_detection.py -v

# Run single test
pytest tests/test_detection.py::TestYOLODetector::test_detect_single_card -v
```

### Benchmarking
```bash
python scripts/benchmark.py
```

---

## Keyboard Controls (During Runtime)

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `s` | Save screenshot |
| `p` | Pause/unpause |
| `r` | Reset cache |
| `d` | Toggle debug mode |
| `c` | Toggle confidence display |
| `h` | Show help overlay |

---

## Project Structure Quick Map

```
src/
├── main.py                      # Main entry point
├── config.py                    # Configuration management
├── detection/
│   ├── yolo_detector.py        # YOLO card detection
│   └── detector_base.py        # Abstract base class
├── identification/
│   ├── feature_matcher.py      # ORB+BEBLID matching
│   └── identifier_base.py      # Abstract base class
├── api/
│   ├── price_api.py            # Price API client
│   └── mock_price_api.py       # Mock for development
└── overlay/
    ├── renderer.py             # Main overlay renderer
    └── components.py           # Reusable UI components

scripts/
├── download_models.py          # Download YOLO11 model
├── build_card_database.py      # Build card reference DB
├── compute_features.py         # Pre-compute card features
└── benchmark.py                # Performance benchmarking

data/
├── card_database/              # Reference card images
│   ├── index.json             # Master card index
│   └── {set_id}/              # Organized by set
└── features/
    └── card_features.pkl      # Pre-computed features

models/
└── yolo11n.pt                  # YOLO11 nano model
```

---

## Configuration Options

### `config.yaml` Key Settings

```yaml
# Camera settings
camera:
  source: 0              # Camera index or video file path
  width: 1280           # Frame width
  height: 720           # Frame height
  fps: 30               # Target FPS

# Detection settings
detection:
  model_path: "models/yolo11n.pt"
  confidence_threshold: 0.5      # Lower = more detections
  iou_threshold: 0.45           # NMS threshold

# Identification settings
identification:
  min_matches: 10               # Minimum feature matches
  match_ratio_threshold: 0.75   # Lowe's ratio test

# Performance settings
performance:
  use_gpu: true                 # Use GPU if available
  frame_skip: 0                 # Skip every N frames (0 = no skip)
```

### Environment Variables

```bash
# Optional API keys
export POKEMON_PRICE_API_KEY="your-key-here"

# Override config settings
export CAMERA_SOURCE=1
export USE_GPU=false
export DEBUG=true
```

---

## Troubleshooting

### Camera Not Working
```bash
# List available cameras
ls /dev/video*

# Test camera with OpenCV
python -c "import cv2; cap = cv2.VideoCapture(0); print(f'Camera opened: {cap.isOpened()}')"

# WSL: Enable USB camera passthrough
# In Windows PowerShell (Admin):
# usbipd wsl attach --busid <bus-id>
```

### GPU Not Detected
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')"

# Install CUDA toolkit (if needed)
# Follow: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
```

### Model Not Found
```bash
# Download YOLO11 model
python scripts/download_models.py

# Verify model exists
ls -lh models/yolo11n.pt
```

### Card Database Empty
```bash
# Build database with starter sets
python scripts/build_card_database.py --sets base1,jungle

# Verify database
python -c "import json; print(json.load(open('data/card_database/index.json')))"
```

### Low FPS
1. Enable GPU: `use_gpu: true` in config
2. Use smaller model: YOLO11n (already default)
3. Lower resolution: `width: 640, height: 480`
4. Enable frame skipping: `frame_skip: 1`

### Poor Identification Accuracy
1. Improve lighting (avoid glare)
2. Hold cards flat and steady
3. Increase `min_matches` threshold
4. Build larger card database
5. Use better camera (1080p+)

---

## Development Workflow

### 1. Starting New Feature
```bash
# Create feature branch
git checkout -b feature/card-grading

# Make changes
# ...

# Run tests
pytest tests/ -v

# Format code
black src/
flake8 src/

# Commit
git add .
git commit -m "Add card grading feature"
```

### 2. Adding New Card Set
```bash
# Find set ID from Pokemon TCG API
# https://api.pokemontcg.io/v2/sets

# Download set
python scripts/build_card_database.py --sets xy1

# Rebuild features
python scripts/compute_features.py

# Test
python src/main.py
```

### 3. Debugging Detection Issues
```bash
# Run with debug mode
python src/main.py --debug

# Save problematic frames
# Press 's' during runtime

# Analyze with test script
python scripts/test_detection.py --image saved_frame.jpg
```

---

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Detection FPS | >30 | TBD |
| Detection Accuracy | >95% | TBD |
| Identification Accuracy | >90% | TBD |
| Identification Time | <50ms | TBD |
| Overlay Render Time | <5ms | TBD |
| End-to-End Latency | <100ms | TBD |

---

## Useful Links

### Documentation
- [YOLO11 Docs](https://docs.ultralytics.com/models/yolo11/)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Pokemon TCG API](https://docs.pokemontcg.io/)

### Resources
- [Pokemon Card Database](https://www.pokemon.com/us/pokemon-tcg/pokemon-cards/)
- [TCGPlayer Price Guide](https://www.tcgplayer.com/search/pokemon/product)
- [ORB Feature Detector](https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html)

### Community
- [r/PokemonTCG](https://www.reddit.com/r/PokemonTCG/)
- [Ultralytics Community](https://community.ultralytics.com/)

---

## Quick Debugging Snippets

### Test Detection on Single Image
```python
import cv2
from src.detection.yolo_detector import YOLOCardDetector

detector = YOLOCardDetector("models/yolo11n.pt")
image = cv2.imread("test_image.jpg")
detections = detector.detect(image)
print(f"Found {len(detections)} cards")
```

### Test Feature Matching
```python
import cv2
from src.identification.feature_matcher import FeatureMatcher

matcher = FeatureMatcher()
matcher.load_database("data/features/card_features.pkl")

card_image = cv2.imread("card.jpg")
matches = matcher.match_card(card_image, top_k=3)
print(f"Top match: {matches[0]['name'] if matches else 'No match'}")
```

### Test Price API
```python
from src.api.price_api import PokemonTCGAPI

api = PokemonTCGAPI()
card_info = api.get_card_info("base1-4")
print(f"Card: {card_info['name']}")
```

---

*Quick reference for PriceLens development*  
*Last updated: November 19, 2025*
