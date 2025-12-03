# Card Detector Training Plan

Goal: Train a Pokemon card-specific YOLO11 detector to replace the COCO weights and improve detection stability and identification quality.

## Prerequisites
- PyTorch and `ultralytics` installed (GPU preferred; CPU works with lower batch sizes).
- Dataset labeled in YOLO format with a single class: `card`.
- Directory layout suggestion: `datasets/cards/{images,labels}` with `data.yaml` at `datasets/cards/data.yaml`.

## 1) Build a Small Labeled Dataset
- Target: 400–800 images, varied angles/lighting/backgrounds, single class `card`.
- Label with Roboflow/CVAT/makesense; export YOLO format (gets you `data.yaml`, `images/`, `labels/`).
- Split: ~70/20/10 train/val/test.

## 2) Fine-tune YOLO11n on the Dataset
- Example command (adjust paths and batch for your hardware):
  ```bash
  yolo train model=models/yolo11n.pt data=datasets/cards/data.yaml imgsz=736 epochs=50 batch=16 device=0 project=runs/cards name=yolo11n-card
  ```
  - If CPU-only: drop `device=0`, reduce `batch` (e.g., 4), and consider `imgsz=640`.
  - Use `yolo11n` for speed; `yolo11m` only if you need higher accuracy and can afford slower inference.

## 3) Pick Checkpoint and Export
- Take `runs/cards/yolo11n-card/weights/best.pt`.
- Copy to repo: `models/card_yolo11n.pt`.
- Optional for CPU speed: export ONNX (or OpenVINO/INT8 if desired):
  ```bash
  yolo export model=runs/cards/yolo11n-card/weights/best.pt format=onnx imgsz=736
  ```

## 4) Wire into the App
- Set `model_path` for `YOLOCardDetector` to `models/card_yolo11n.pt` (or your exported path) in config.
- In `yolo_detector.py`, enforce the card class:
  - `allowed_class_names = {'card'}` (and clear `blocked_class_ids`).
- Tuning defaults for the card model:
  - `conf_threshold`: start at `0.35–0.4`.
  - `imgsz`: set to `640` or `736` when calling `predict` (add to config if not present).
  - Tracking: consider 2-hit confirmation to lock IDs faster with the cleaner model.

## 5) Validate
- Run live detection and check:
  - Box stability (less flicker).
  - Correct crops feeding the identifier.
- If boxes are still shaky: lower tracking confirmation hits to 2 and increase client polling to ~200–250ms.

## 6) Optional Speed/Quality Tweaks
- If GPU unavailable: use the ONNX export with OpenVINO or INT8 to speed up CPU inference.
- If you expand the dataset later: retrain with more images; keep `yolo11n` for speed unless you need higher recall.
- Maintain a simple class list (`card`) to avoid class-mapping issues.
