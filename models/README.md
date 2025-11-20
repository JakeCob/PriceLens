# Models Directory

This directory contains machine learning models for PriceLens.

## Required Models

1. **yolo11n.pt** - YOLO11 nano model for card detection
   - Downloaded automatically via `scripts/download_models.py`
   - Size: ~6MB

2. **pokemon_cards_yolo11.pt** (Future)
   - Fine-tuned model specifically for Pokemon cards
   - To be created through training

## Usage

Run the download script from the project root:
```bash
python scripts/download_models.py
```
