# Training Resume Guide - YOLO11 Pokemon Card Detector

## Quick Start

### 1. Open the Notebook
```bash
cd /root/Programming\ Projects/Personal/PriceLens
jupyter notebook notebooks/train_card_detector.ipynb
```

### 2. Run Cells in Order
- **Cells 1-12**: Setup and configuration (run once)
- **Cell 13**: Checkpoint status checker (run anytime to check progress)
- **Cell 14**: Training cell with auto-resume (main training)

### 3. Start Training
Just click "Run" on Cell 14 - it will:
- Detect if you have a checkpoint
- Resume from last epoch if interrupted
- Save progress every epoch

## How Resume Works

### First Run
```
🚀 Initializing YOLO11n model...
   Starting fresh training from pretrained weights...
📁 Results will be saved to: ../models/training_runs/yolo11n_cleveland_notebook/
```
- Starts from epoch 1
- Uses pretrained YOLO11n.pt as base
- Saves checkpoint after each epoch

### After Interruption
```
✅ Found checkpoint: ../models/training_runs/yolo11n_cleveland_notebook/weights/last.pt
   Resuming training from last saved epoch...
🔄 RESUME MODE: Continuing from previous checkpoint
```
- Automatically detects checkpoint
- Resumes from last completed epoch
- Continues to epoch 100

## Checkpoint Files

YOLO saves two checkpoint files:

| File | Purpose |
|------|---------|
| `best.pt` | Best model so far (highest mAP@50) |
| `last.pt` | Most recent epoch checkpoint |

**Location**: `models/training_runs/yolo11n_cleveland_notebook/weights/`

## Common Scenarios

### Training Interrupted at Epoch 15
1. Check status with Cell 13:
   ```
   📈 Training Progress:
      Completed epochs: 15/100
      Progress: 15%
      Latest mAP@50: 0.8942

   🔄 Next run will resume from epoch 16
   ```

2. Run Cell 14 again - it continues from epoch 16

### Training Stops Early (Bug)
If training exits cleanly but incomplete (like the current issue):
1. Check `results.csv` to see last completed epoch
2. Run Cell 14 again - auto-resumes
3. Repeat until 100 epochs complete

### Start Fresh Training
Delete the checkpoint directory:
```bash
rm -rf models/training_runs/yolo11n_cleveland_notebook
```
Then run Cell 14 - starts from epoch 1

## Monitoring Progress

### In Jupyter Notebook
- Live progress bar shows current epoch
- Loss values update in real-time
- GPU memory usage displayed

### From Terminal (While Training)
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor latest metrics
tail -f models/training_runs/yolo11n_cleveland_notebook/results.csv

# Check progress
python scripts/monitor_training.py models/training_runs/yolo11n_cleveland_notebook
```

### Via Results CSV
```bash
cat models/training_runs/yolo11n_cleveland_notebook/results.csv
```

Columns to watch:
- `epoch`: Current epoch number
- `metrics/mAP50(B)`: Main accuracy metric (higher = better)
- `train/box_loss`: Bounding box loss (lower = better)

## Troubleshooting

### "Training stopped at epoch X"
**Solution**: Run Cell 14 again - it will resume from epoch X+1

### "Checkpoint not found"
**Check**: Make sure you ran cells 1-12 first (setup)

### "Out of memory"
**Solution**:
1. Edit Cell 11 configuration:
   ```python
   CONFIG['batch'] = 8  # Reduce from 16
   ```
2. Restart kernel
3. Re-run cells 1-12, then Cell 14

### "CUDA error"
**Solution**:
```bash
# Terminal - reinstall PyTorch
conda activate pricelens
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
Then restart Jupyter kernel

## Expected Timeline

| Epochs | Time (RTX 4070) | mAP@50 Expected |
|--------|-----------------|-----------------|
| 10 | ~15 min | 0.85+ |
| 25 | ~40 min | 0.90+ |
| 50 | ~1.5 hrs | 0.92+ |
| 100 | ~2-3 hrs | 0.93+ |

## After Training Complete

### 1. Final Model Location
```
models/training_runs/yolo11n_cleveland_notebook/weights/best.pt
```

### 2. Copy to PriceLens Models
```bash
cp models/training_runs/yolo11n_cleveland_notebook/weights/best.pt \
   models/pokemon_card_yolo11.pt
```

### 3. Update config.yaml
```yaml
detection:
  model_path: "models/pokemon_card_yolo11.pt"
  confidence_threshold: 0.5
```

### 4. Test the Model
```bash
python scripts/test_detector_standalone.py
```

## Tips

✅ **Save progress often**: YOLO auto-saves every epoch
✅ **Monitor GPU**: Use `nvidia-smi` to ensure GPU is active
✅ **Check Cell 13**: Shows current progress anytime
✅ **Be patient**: 100 epochs takes 2-3 hours
✅ **Resume works**: If interrupted, just run Cell 14 again

❌ **Don't**: Delete weights folder during training
❌ **Don't**: Change CONFIG during resume (will start fresh)
❌ **Don't**: Interrupt during epoch (wait for epoch to complete)

## Contact

Issues? Check:
- `/root/Programming Projects/Personal/PriceLens/docs/CARD_MODEL_SETUP.md`
- `/root/Programming Projects/Personal/PriceLens/docs/CUDA_COMPATIBILITY.md`
