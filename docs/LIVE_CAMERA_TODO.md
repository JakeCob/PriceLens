# Live Camera Improvements TODO

## Detection Accuracy Issues (Discovered: 2025-11-23)

### 1. False Positives
**Issue**: YOLO is detecting human faces and random objects as Pokemon cards.

**Potential Fixes**:
- [ ] Fine-tune YOLO model specifically on Pokemon cards dataset
- [ ] Add post-detection filtering (check aspect ratio, size, etc.)
- [ ] Use feature matching confidence threshold to reject non-cards
- [ ] Train custom YOLO model with negative samples (faces, hands, etc.)
- [ ] Add object classification layer after detection

**Priority**: High  
**Difficulty**: Medium-High (requires model fine-tuning)

---

### 2. Low Light & Glare Sensitivity
**Issue**: Cards not detected in dim lighting or with glare/reflections.

**Potential Fixes**:
- [ ] Add auto-exposure/brightness adjustment preprocessing
- [ ] Use adaptive histogram equalization (CLAHE) before detection
- [ ] Add "low light mode" with different YOLO settings
- [ ] Guide users with on-screen feedback ("Move to better lighting")
- [ ] Multi-exposure frame capture and merge

**Priority**: Medium  
**Difficulty**: Low-Medium (mostly preprocessing)

---

### 3. Overlapping Cards Detection
**Issue**: Cannot detect cards when partially obscured or overlapping.

**Potential Fixes**:
- [ ] Enable NMS (Non-Maximum Suppression) tuning for overlaps
- [ ] Use instance segmentation instead of bounding boxes
- [ ] Implement Z-ordering detection (depth estimation)
- [ ] Add "single card mode" recommendation in UI
- [ ] Use multiple detection passes with different IOU thresholds

**Priority**: Low  
**Difficulty**: Medium (requires architectural changes)

---

## Performance Optimizations

### 4. Frame Processing Speed
**Current**: ~500ms interval (2 FPS)
- [ ] Use WebWorkers for frame capture
- [ ] Downscale frames before sending to backend
- [ ] Implement frame skipping when detection is slow
- [ ] Cache detector model in GPU memory
- [ ] Use TensorRT or ONNX for faster inference

---

## UX Improvements

### 5. User Guidance
- [ ] Show "detection quality" indicator
- [ ] Add "card found" haptic feedback (mobile)
- [ ] Display detection confidence on overlay
- [ ] Add "best lighting tips" popup on first use
- [ ] Auto-focus on cards when detected

---

## Phase 3+ Features

### 6. Async Price Fetching (IN PROGRESS)
- [/] Background price queue
- [ ] WebSocket for live price updates
- [ ] Price cache warming on camera start

### 7. Advanced Features (Future)
- [ ] Multi-card batch pricing
- [ ] Price history graph overlay
- [ ] Collection value tracking
- [ ] AR mode with 3D card effects
- [ ] Offline mode with cached prices

---

## Notes
- YOLO11n is the fastest model but least accurate
- Consider YOLO11s or YOLO11m for better accuracy (slower)
- The current model was trained on generic objects, not Pokemon cards
- Custom dataset of ~5000 Pokemon card images would improve accuracy significantly
