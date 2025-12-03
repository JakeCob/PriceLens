# Pokemon Card-Specific Model Setup Guide

This guide explains how to integrate a pre-trained Pokemon card detection model from Roboflow to replace the generic COCO-trained YOLO model.

## Why Use a Card-Specific Model?

The card-specific model offers several advantages over the generic COCO model:

1. **Higher Accuracy**: Trained specifically on Pokemon cards, not general objects
2. **No False Positives**: Won't detect faces, people, or other objects as cards
3. **Better Performance**: Can use lower confidence thresholds since it's specialized
4. **Cleaner Detection**: No need for face filtering or complex class blocking

## Prerequisites

1. **Roboflow API Key**: Get a free API key from [https://roboflow.com](https://roboflow.com)
2. **Python Environment**: Ensure all dependencies are installed
3. **Internet Connection**: Required for initial model download

## Step 1: Install Roboflow

The Roboflow package has already been added to `requirements.txt`:

```bash
pip install roboflow>=1.1.0
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Step 2: Configure API Key

Add your Roboflow API key to your `.env` file:

```bash
# .env
ROBOFLOW_API_KEY=your_roboflow_api_key_here
```

You can also set it as an environment variable:

```bash
export ROBOFLOW_API_KEY=your_api_key_here
```

## Step 3: Download the Card Model

Run the download script:

```bash
python scripts/download_card_model.py
```

This will:
- Connect to Roboflow using your API key
- Download the Pokemon card detection model
- Save the model weights to `models/pokemon_card_yolo11.pt`

### Optional: Verify Model

To verify the model after downloading:

```bash
python scripts/download_card_model.py --verify
```

### Troubleshooting Download

If the automatic download fails:

1. **Manual Download**: Visit [Roboflow Universe](https://universe.roboflow.com/pokemon-scanner/pokemon-card-detector-cuyon)
2. **Export Model**: Export as YOLOv11 format
3. **Save Weights**: Place the `best.pt` file in `models/pokemon_card_yolo11.pt`

## Step 4: Update Configuration

Edit `config.yaml` to use the card-specific model:

```yaml
detection:
  model_path: "models/pokemon_card_yolo11.pt"  # Changed from yolo11n.pt
  use_card_specific_model: true                # Changed from false
  confidence_threshold: 0.4                    # Can lower threshold with card-specific model
  iou_threshold: 0.45
  max_detections: 6
```

### Configuration Parameters

- **model_path**: Path to the model weights file
  - Default COCO model: `models/yolo11n.pt`
  - Card-specific model: `models/pokemon_card_yolo11.pt`

- **use_card_specific_model**: Boolean flag
  - `false` (default): Uses COCO model with class filtering
  - `true`: Uses card-specific model, disables filtering

- **confidence_threshold**: Detection confidence threshold
  - COCO model: 0.5-0.65 (higher to reduce false positives)
  - Card-specific model: 0.3-0.5 (can be lower due to specialization)

## Step 5: Test the Model

Test the model with sample images:

```bash
python scripts/test_detector_images.py
```

Or test with the web API:

```bash
python src/web/api.py
```

Then open your browser to test the live detection.

## Model Comparison

### COCO-Pretrained Model (`yolo11n.pt`)

**Pros:**
- Pre-trained, no download needed
- Works out of the box
- General object detection capability

**Cons:**
- Detects many object types (80 classes)
- May detect faces, people as cards
- Requires face filtering and class blocking
- Lower accuracy on cards specifically
- Higher confidence threshold needed (0.5-0.65)

**Configuration:**
```yaml
detection:
  model_path: "models/yolo11n.pt"
  use_card_specific_model: false
  confidence_threshold: 0.5
```

### Pokemon Card-Specific Model (`pokemon_card_yolo11.pt`)

**Pros:**
- Trained specifically on Pokemon cards
- Higher accuracy for card detection
- No false positives from faces/people
- Can use lower confidence threshold (0.3-0.5)
- Cleaner, simpler detection pipeline

**Cons:**
- Requires Roboflow API key
- Initial download needed
- May only detect Pokemon cards (not other trading cards)

**Configuration:**
```yaml
detection:
  model_path: "models/pokemon_card_yolo11.pt"
  use_card_specific_model: true
  confidence_threshold: 0.4
```

## Advanced: Training Your Own Model

If you want to train your own card detection model:

### Option 1: Roboflow (Recommended for Beginners)

1. Create account on [Roboflow](https://roboflow.com)
2. Create new project with "Object Detection"
3. Upload images of Pokemon cards (100+ recommended)
4. Annotate bounding boxes around cards
5. Generate dataset with augmentation
6. Train using Roboflow's hosted training
7. Export and download model

### Option 2: Local Training with YOLO

Download a larger dataset:

```bash
# Using the merged Pokemon cards dataset (3093 images)
wget https://universe.roboflow.com/pokemon-cards-merged/download

# Train custom model
yolo detect train \
    data=pokemon-cards-merged/data.yaml \
    model=yolo11n.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    device=0  # Use GPU
```

## Switching Between Models

You can easily switch between models by editing `config.yaml`:

### Use COCO Model
```yaml
detection:
  model_path: "models/yolo11n.pt"
  use_card_specific_model: false
  confidence_threshold: 0.5
```

### Use Card-Specific Model
```yaml
detection:
  model_path: "models/pokemon_card_yolo11.pt"
  use_card_specific_model: true
  confidence_threshold: 0.4
```

No code changes needed - just restart the application!

## Performance Tips

### For Card-Specific Model

1. **Lower Confidence Threshold**: Try 0.3-0.4 for better detection
2. **Disable Face Detection**: Already disabled automatically when `use_card_specific_model=true`
3. **Remove Class Filtering**: Already disabled automatically

### For Both Models

1. **GPU Acceleration**: Ensure CUDA is available for faster inference
2. **Model Size**: Use larger models (yolo11m, yolo11l) for better accuracy
3. **Image Quality**: Use good lighting and clear camera for best results

## Troubleshooting

### Problem: "Model not found" error

**Solution**: Run the download script:
```bash
python scripts/download_card_model.py
```

### Problem: Roboflow API key error

**Solution**: Check your `.env` file has the correct key:
```bash
cat .env | grep ROBOFLOW
```

### Problem: Model detects non-cards

**Solution**:
1. Ensure `use_card_specific_model: true` in config
2. Increase confidence threshold
3. Verify you're using the card-specific model, not COCO model

### Problem: Low detection accuracy

**Solution**:
1. Lower confidence threshold to 0.3-0.4
2. Improve lighting conditions
3. Ensure cards are clearly visible
4. Consider training with more data

## References

- [Roboflow Universe - Pokemon Card Detector](https://universe.roboflow.com/pokemon-scanner/pokemon-card-detector-cuyon)
- [YOLO11 Documentation](https://docs.ultralytics.com/models/yolo11/)
- [Roboflow Documentation](https://docs.roboflow.com/)

## Support

If you encounter issues:

1. Check the logs in the console output
2. Verify all configuration settings
3. Test with the COCO model first to ensure basic detection works
4. Review the model output classes: `python -c "from ultralytics import YOLO; print(YOLO('models/pokemon_card_yolo11.pt').names)"`
