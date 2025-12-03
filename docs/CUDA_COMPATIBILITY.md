# CUDA Compatibility Issue - RTX 40-Series & PyTorch 2.9.1

## Issue Summary

When training YOLO models with PyTorch 2.9.1+cu128 on RTX 40-series GPUs (Ada Lovelace architecture), you may encounter:

```
torch.AcceleratorError: CUDA error: an illegal instruction was encountered
```

## Your System Configuration

- **GPU**: NVIDIA GeForce RTX 4070 Laptop GPU
- **CUDA Driver**: 13.0
- **PyTorch**: 2.9.1+cu128
- **Ultralytics**: 8.3.0

## Root Cause

This is a known compatibility issue between:
- PyTorch 2.9.1
- CUDA 12.8 (compiled into PyTorch)
- RTX 40-series GPUs (Ada Lovelace, Compute Capability 8.9)

The error occurs during:
- Mixed precision (AMP) training
- Certain tensor operations in loss calculation
- Gradient scaling operations

## Current Status

**✓ Partial Training Completed**: Training ran for 13-14 epochs before crashing
**✓ Model Weights Saved**: Best checkpoint saved to `models/pokemon_card_yolo11.pt`
**✓ Inference Works**: Model loads and runs inference successfully
**✗ Full Training**: Cannot complete full 30-epoch training due to CUDA error

The partially trained model (13 epochs) achieved:
- **mAP50**: 0.912
- **mAP50-95**: 0.755
- **Recall**: 0.207-0.967 (varied by epoch)

This is already usable for detection, though more training would improve it.

## Workarounds

### Option 1: Use Partially Trained Model (Current)

The model trained for 13 epochs before crashing. This is sufficient for basic card detection:

```yaml
# config.yaml
detection:
  model_path: "models/pokemon_card_yolo11.pt"
  use_card_specific_model: true
  confidence_threshold: 0.4
```

**Pros**: Already done, works for inference
**Cons**: Not fully trained (could be better with more epochs)

### Option 2: Downgrade PyTorch (Recommended for Full Training)

Downgrade to a more stable PyTorch version:

```bash
pip uninstall torch torchvision
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
```

Then retrain:
```bash
python scripts/train_card_model.py
```

**Pros**: More stable, proven compatibility
**Cons**: Requires reinstalling PyTorch

### Option 3: Google Colab Training

Train on Google Colab's free GPUs to avoid local CUDA issues:

1. Upload `Pokemon-Card-Detector-1/` dataset to Google Drive
2. Use this Colab notebook:

```python
# Install ultralytics
!pip install ultralytics

# Train model
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
results = model.train(
    data='/content/drive/MyDrive/Pokemon-Card-Detector-1/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device=0
)

# Download weights
from google.colab import files
files.download('runs/detect/train/weights/best.pt')
```

**Pros**: Free GPU, no local setup issues
**Cons**: Need to upload/download data

### Option 4: Use COCO Model with Filtering

Continue using the COCO-pretrained model with enhanced filtering:

```yaml
# config.yaml
detection:
  model_path: "models/yolo11n.pt"  # or yolo11m.pt for better accuracy
  use_card_specific_model: false
  confidence_threshold: 0.65
```

**Pros**: No training needed, works reliably
**Cons**: May have occasional false positives (mitigated by filtering)

### Option 5: CPU Training (Slow but Stable)

Train on CPU to avoid CUDA issues:

```bash
export CUDA_VISIBLE_DEVICES=""
python -c "
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
model.train(
    data='Pokemon-Card-Detector-1/data.yaml',
    epochs=50,
    device='cpu',
    batch=4
)
"
```

**Pros**: Stable, no CUDA errors
**Cons**: Very slow (10-20x slower than GPU)

## Testing Your Model

Test inference (this should work even with the CUDA issue):

```python
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('models/pokemon_card_yolo11.pt')

# Test on image
results = model.predict('test_image.jpg', conf=0.4)

# Display
for r in results:
    im = r.plot()
    cv2.imshow('Detection', im)
    cv2.waitKey(0)
```

## Recommended Action

**For now**: Use the partially trained model (`models/pokemon_card_yolo11.pt`) with `use_card_specific_model: true`. It should work well for most cards.

**For production**: Either:
1. Train fully on Google Colab
2. Downgrade PyTorch and train locally
3. Use the COCO model with filtering (proven to work)

## Future Updates

Monitor for:
- PyTorch 2.10+ which may fix RTX 40-series compatibility
- Ultralytics updates addressing this issue
- Updated CUDA drivers

## References

- [PyTorch CUDA Compatibility](https://pytorch.org/get-started/locally/)
- [Ultralytics YOLO Issues](https://github.com/ultralytics/ultralytics/issues)
- [NVIDIA CUDA Error Reference](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html)
