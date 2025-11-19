# AGENTS.md - AI Assistant Context Guide

**Project:** PriceLens - Pokemon Card Price Overlay System  
**Owner:** JakeCob  
**Last Updated:** November 19, 2025  
**Current Status:** Early Development (~10-15% complete)

---

## 🎯 Project Overview

PriceLens is a real-time computer vision system that detects Pokemon trading cards through a camera feed and overlays live market price information directly onto the video stream.

**Tech Stack:**
- Python 3.10+
- YOLO11 (object detection)
- OpenCV (computer vision)
- ORB + BEBLID (feature matching)
- PyTorch (deep learning)
- FastAPI (optional web version)

**Target Performance:**
- 30+ FPS with GPU
- >90% card identification accuracy
- <100ms end-to-end latency
- Support 4-6 simultaneous cards

---

## 📁 Key Documentation Files

### Must Read First
1. **`README.md`** - User-facing documentation, quick start guide
2. **`CLAUDE.md`** - Comprehensive 1,888-line implementation guide (GOLD MINE!)
3. **`docs/plan/implementation-plan.md`** - 7-phase development roadmap
4. **`docs/plan/quick-reference.md`** - Commands, troubleshooting, snippets
5. **`THIS FILE`** - Context for AI agents

### Configuration
- **`config.yaml`** - Application configuration (camera, detection, API, overlay)
- **`requirements.txt`** - Production dependencies
- **`requirements-dev.txt`** - Development dependencies
- **`setup.py`** - Package setup

---

## 🏗️ Architecture Overview

```
Pipeline Flow:
Camera → Frame Capture → Detection (YOLO11) → Identification (ORB+BEBLID)
    → Price Lookup (API) → Overlay Rendering → Display

Key Components:
├── src/detection/          # Card detection (YOLO11)
├── src/identification/     # Card matching (feature matching)
├── src/api/               # Price data fetching
├── src/overlay/           # UI rendering
└── src/main.py            # Main application loop
```

---

## 📊 Current Project State

### ✅ What's Complete (10-15%)
- Project structure and organization
- Configuration system (YAML + env vars)
- Logging infrastructure
- Base abstract classes (`DetectorBase`, `IdentifierBase`)
- Docker setup (Dockerfile, docker-compose.yml)
- Exceptional documentation (CLAUDE.md)
- Test framework structure
- Implementation plan (Nov 19, 2025)

### ❌ What's Missing (Critical Implementation)
- **No YOLO detector implementation** (`src/detection/yolo_detector.py` doesn't exist)
- **No feature matcher** (`src/identification/feature_matcher.py` doesn't exist)
- **No price API clients** (`src/api/` is empty except `__init__.py`)
- **No overlay renderer** (`src/overlay/` is empty except `__init__.py`)
- **Main app incomplete** (has TODOs, no actual processing loop)
- **Empty card database** (`data/card_database/index.json` is empty)
- **No models downloaded** (need to run `scripts/download_models.py`)

### 🚧 Partially Complete
- `src/main.py` - Entry point exists but placeholder implementation
- `scripts/download_models.py` - Script exists but not executed
- `scripts/benchmark.py` - Has structure but all benchmarks are TODOs
- `tests/test_basic.py` - Basic tests exist but minimal

---

## 🎯 Current Phase: Phase 1 - Environment Setup

**Goal:** Get basic card detection working with webcam

**Active Todo List:**
1. ⏳ Set up Python environment with uv
2. ⬜ Download YOLO11 model
3. ⬜ Implement YOLOCardDetector class
4. ⬜ Test camera capture and detection

**Next Session Should:**
1. Complete environment setup
2. Download YOLO11n model (~6MB)
3. Start implementing `src/detection/yolo_detector.py`
4. Test webcam access (may need WSL USB passthrough)

---

## 🔧 Development Environment

### Tools Installed
- ✅ `uv` (v0.9.10) - Fast Python package installer
- Located at: `~/.local/bin/uv`
- Activation: `source $HOME/.local/bin/env`

### System Info
- OS: Linux (WSL2)
- Shell: bash
- User: root
- Working Directory: `/root/Programming Projects/Personal/PriceLens`
- Editor: Cursor/VS Code

### Known Issues
- MCP server error (Azure.Mcp) - Not critical, doesn't affect project
- WSL camera access may require USB passthrough
- GPU availability unknown (need to test CUDA)

---

## 📝 Coding Conventions

### Python Style
- Follow PEP 8
- Use type hints
- Docstrings for all public methods
- Logger instead of print statements
- Abstract base classes for extensibility

### File Organization
```python
# Standard file structure:
#!/usr/bin/env python3
"""Module docstring"""

import standard_library
import third_party
from local_module import LocalClass

# Constants
CONSTANT_NAME = value

# Classes
class ClassName:
    def __init__(self):
        """Init docstring"""
        pass
    
    def public_method(self) -> ReturnType:
        """Method docstring"""
        pass
    
    def _private_method(self) -> ReturnType:
        """Private method"""
        pass

# Main execution
if __name__ == "__main__":
    main()
```

### Git Workflow
- Branch naming: `feature/name`, `fix/name`, `docs/name`
- Commit messages: Conventional Commits format
  - `feat:` - New feature
  - `fix:` - Bug fix
  - `docs:` - Documentation
  - `refactor:` - Code refactoring
  - `test:` - Tests
  - `chore:` - Maintenance

---

## 🚨 Critical Gotchas & Warnings

### 1. WSL Camera Access
**Issue:** Camera may not work in WSL by default  
**Solution:** Use `usbipd` to attach USB camera from Windows
```bash
# In Windows PowerShell (Admin):
usbipd wsl list
usbipd wsl attach --busid <busid>
```

### 2. GPU Availability
**Issue:** CUDA may not be available in WSL  
**Solution:** Test with `torch.cuda.is_available()`, fall back to CPU
```python
import torch
print(f"CUDA: {torch.cuda.is_available()}")
```

### 3. API Keys Required
**Issue:** Price APIs need authentication  
**Solution:** Use mock API for development first
- Pokemon TCG API (free, no prices)
- PokemonPriceTracker (paid, ~$20/month)

### 4. Card Database Size
**Issue:** Full database is ~5GB  
**Solution:** Start with 10-20 popular cards (~50MB)

### 5. BEBLID Descriptor
**Issue:** Requires `opencv-contrib-python`  
**Solution:** Already in `requirements.txt`, but verify install

---

## 🎓 Domain Knowledge

### Pokemon Card Structure
- Standard size: 2.5" × 3.5" (aspect ratio ~0.71)
- Front: Card image, name, HP, attacks, etc.
- Back: Pokemon logo (red/white)
- Variations: Holographic, reverse holo, 1st edition

### Card Identification Challenge
- Multiple editions (Base Set, Shadowless, 1st Edition)
- Condition variations (Near Mint to Heavily Played)
- Holographic reflections affect feature matching
- Similar artwork across different cards

### Price Data Sources
- TCGPlayer (most authoritative)
- eBay (market prices)
- CardMarket (European market)
- Graded cards: PSA 10, PSA 9, BGS 9.5, etc.

---

## 💡 Implementation Tips for AI Agents

### When Starting a New Session
1. **Read this file first** to understand project state
2. **Check todo list** in context for current priorities
3. **Review `docs/plan/implementation-plan.md`** for current phase
4. **Don't assume code exists** - verify files before editing
5. **Test incrementally** - small changes, frequent testing

### When Writing Code
1. **Follow CLAUDE.md examples** - they're well-designed
2. **Inherit from base classes** - `DetectorBase`, `IdentifierBase`
3. **Use config system** - don't hardcode values
4. **Add logging** - use `logger.info()`, `logger.debug()`, etc.
5. **Handle errors gracefully** - computer vision is unpredictable

### When Implementing Detection
```python
# Good: Filter by aspect ratio
if 0.5 < aspect_ratio < 0.9:
    detections.append(detection)

# Good: Add confidence threshold
if confidence > self.conf_threshold:
    process_detection(det)

# Good: Extract with padding
x1 = max(0, x1 - padding)
```

### When Implementing Feature Matching
```python
# Good: Use Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Good: Verify with homography
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
inlier_ratio = np.sum(mask) / len(matches)
```

### When Testing
1. **Use synthetic data first** - create test images programmatically
2. **Test edge cases** - no cards, rotated cards, overlapping cards
3. **Measure performance** - log FPS, latency, accuracy
4. **Test on real hardware** - webcam behavior varies

---

## 📦 Common Commands Reference

```bash
# Environment setup
uv venv
source venv/bin/activate
uv pip install -r requirements.txt

# Download models
python scripts/download_models.py

# Run application
python src/main.py
python src/main.py --debug
python src/main.py --no-gpu

# Testing
pytest tests/ -v
pytest tests/ -v --cov=src

# Build card database
python scripts/build_card_database.py --sets base1,jungle

# Benchmarking
python scripts/benchmark.py
```

---

## 🔗 External Dependencies & APIs

### Pokemon TCG API
- **URL:** https://api.pokemontcg.io/v2
- **Auth:** None required
- **Rate Limit:** 1000 requests/hour
- **Data:** Card info, images, sets (NO PRICES)
- **Docs:** https://docs.pokemontcg.io/

### YOLO11 (Ultralytics)
- **Docs:** https://docs.ultralytics.com/models/yolo11/
- **Model:** yolo11n.pt (nano, ~6MB)
- **License:** AGPL-3.0
- **GPU Support:** CUDA, MPS, OpenVINO

### OpenCV
- **Version:** 4.8.1.78
- **Contrib:** Required for BEBLID descriptor
- **Docs:** https://docs.opencv.org/4.x/

---

## 🎯 Success Criteria Reminder

### Minimum Viable Product (MVP)
- ✅ Detects Pokemon cards at 30 FPS
- ✅ Identifies cards from database with >85% accuracy
- ✅ Displays card name and set
- ✅ Shows price information (even if mock)
- ✅ Handles 2-3 cards simultaneously
- ✅ Works in good lighting conditions

### Timeline
- **Week 1:** Detection working
- **Week 2-3:** Identification working
- **Week 4:** Price API integrated
- **Week 5:** Full UI
- **Week 6:** Testing & polish

---

## 🤝 Working with the User

### User Preferences
- Prefers step-by-step implementation
- Likes to see code working incrementally
- Appreciates detailed explanations
- Uses WSL2 on Windows
- Has `uv` installed for fast package management
- Works in Cursor/VS Code

### Communication Style
- Be concise but thorough
- Show code examples
- Explain trade-offs
- Ask before making major architectural changes
- Provide commit messages when requested

---

## 📌 Quick Reference for Next Session

### Immediate Actions
1. Create virtual environment: `uv venv`
2. Install dependencies: `uv pip install -r requirements.txt`
3. Download YOLO11: `python scripts/download_models.py`
4. Test GPU: `python -c "import torch; print(torch.cuda.is_available())"`
5. Test camera: `python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"`

### Files to Create First
1. `src/detection/yolo_detector.py` - YOLOCardDetector class
2. `scripts/test_detection.py` - Quick test script
3. `tests/test_detection.py` - Unit tests

### Don't Forget
- Activate venv before running Python
- Check CUDA availability (affects performance)
- WSL camera may need USB passthrough
- Start with mock price API (no keys needed)
- Test with real Pokemon cards if available

---

## 🔄 Session History

### Session 1 (November 19, 2025)
- Analyzed repository structure
- Identified 10-15% completion status
- Installed `uv` package manager
- Created implementation plan (7 phases, 6-8 weeks to MVP)
- Created quick reference guide
- Created this AGENTS.md file
- Troubleshot MCP server issue (not critical)
- Set up Phase 1 todo list

**Status:** Ready to begin Phase 1 implementation  
**Next:** Environment setup and YOLO detector implementation

---

## 📚 Additional Resources

### Learning Materials
- CLAUDE.md sections 1-6 cover theory and design
- OpenCV Python tutorials: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
- YOLO11 examples: https://github.com/ultralytics/ultralytics

### Similar Projects
- Card recognition apps (use ML classification)
- QR code scanning (similar overlay approach)
- Document scanners (similar detection pipeline)

### Academic Papers
- ORB: Oriented FAST and Rotated BRIEF
- BEBLID: Boosted Efficient Binary Local Image Descriptor
- YOLO: You Only Look Once

---

## ⚠️ Important Notes

1. **This project is a LEARNING project** - prioritize understanding over speed
2. **GPU is recommended** but not required for development
3. **Start small** - 10-20 cards before building full database
4. **Test incrementally** - don't write all code before testing
5. **Ask user** before making major architectural changes

---

## 🎨 Project Vision

**End Goal:** Point phone at Pokemon cards → See instant prices overlaid on screen

**Inspiration:** Shazam for Pokemon cards

**Use Cases:**
- Collectors: Quick price checks at conventions
- Traders: Evaluate trade value in real-time
- Sellers: Price inventory efficiently
- Buyers: Verify fair pricing before purchase

---

## 📧 Contact & Support

**Repository:** https://github.com/JakeCob/PriceLens  
**Owner:** JakeCob  
**AI Assistant:** GitHub Copilot / Cursor AI

---

*This file is a living document. Update it at the end of each significant session.*

**Last Updated:** November 19, 2025  
**Next Update Due:** After Phase 1 completion
