# PriceLens - Pokemon Card Price Overlay System

Real-time computer vision system that detects Pokemon trading cards through a camera feed and overlays live market price information directly onto the video stream.

## Features

- 🎯 Real-time card detection using YOLO11
- 🔍 Accurate card identification with feature matching
- 💰 Live market price fetching from multiple sources
- 📊 Price trend visualization
- 🎮 30+ FPS performance on consumer hardware

## Quick Start

### Prerequisites

- Python 3.10+
- Webcam or USB camera (720p minimum)
- 8GB RAM minimum (16GB recommended)
- Optional: NVIDIA GPU for enhanced performance

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PriceLens.git
cd PriceLens
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development
```

4. Download YOLO11 model:
```bash
python scripts/download_models.py
```

5. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

6. Build card database (optional):
```bash
python scripts/build_card_database.py
```

### Running the Desktop Application

```bash
python src/main.py
```

Press `q` to quit, `s` to save screenshot.

### Running the Web Application

The web interface provides a modern browser-based experience:

1. **Start the Python backend** (API server):
```bash
python run_web.py
```
The backend runs at `http://localhost:8080`

2. **Start the Next.js frontend** (in a separate terminal):
```bash
cd web
npm install        # First time only
npm run dev
```
The frontend runs at `http://localhost:3000`

3. Open your browser and go to `http://localhost:3000`

#### Web App Features
- 🌐 Modern responsive design with dark mode
- 📱 Works on desktop and mobile browsers
- 📸 Live camera integration
- 💰 Real-time price overlay
- 📊 Collection tracking and analytics


## Configuration

Edit `config.yaml` to customize:
- Camera settings
- Detection parameters
- API preferences
- Overlay appearance
- Performance options

## Project Structure

```
PriceLens/
├── src/                # Source code
│   ├── detection/      # Card detection modules
│   ├── identification/ # Card identification
│   ├── api/           # Price API clients
│   ├── overlay/       # Rendering engine
│   └── utils/         # Utilities
├── models/            # ML models
├── data/              # Card database
├── tests/             # Test suite
├── docs/              # Documentation
└── scripts/           # Utility scripts
```

## Documentation

For detailed implementation guide, see [CLAUDE.md](CLAUDE.md).

## Performance

- Detection: >95% accuracy for front-facing cards
- Identification: >90% accuracy for cards in database
- Processing: <100ms end-to-end latency
- Frame rate: ≥30 FPS on recommended hardware

## API Support

- PokemonPriceTracker API (primary)
- Pokemon TCG API (fallback)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Follow PEP 8 style guide
5. Submit a pull request

## License

MIT License - See LICENSE file for details

## Acknowledgments

- YOLO11 by Ultralytics
- OpenCV Community
- Pokemon TCG API

## Disclaimer

This project is for educational purposes. Pokemon and all related properties are trademarks of Nintendo/Game Freak/The Pokemon Company.