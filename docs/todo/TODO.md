# PriceLens Performance Improvement TODO

Based on benchmark results from RTX 4070 (2024-12-24)

## 🔴 Critical - Major Bottlenecks

### Card Identification (~2000ms → target <100ms)
The identification pipeline is the primary bottleneck, taking ~2 seconds per card.

- [ ] **Optimize feature matching algorithm**
  - Current: Sequential matching against 420 cards
  - Consider: FAISS/Annoy for approximate nearest neighbor search
  - Consider: Pre-filter candidates by card type/color histogram
  
- [ ] **Reduce feature database size**
  - Current: Full ORB descriptors for all cards
  - Consider: PCA dimensionality reduction
  - Consider: Quantized descriptors
  
- [ ] **Add early termination**
  - Stop matching when confidence > threshold
  - Skip low-probability candidates

## 🟡 Important - Performance Improvements

### End-to-End Pipeline (150-385ms → target <100ms)
- [ ] **Parallel detection + identification**
  - Run identification on previous frame while detecting current
  - Batch multiple card regions together
  
- [ ] **Cache identified cards by position**
  - Skip re-identification for stationary cards
  - Use tracking IDs to maintain state

### Cache Write Latency (18ms → target <5ms)
- [ ] **Async cache writes**
  - Don't block on disk I/O
  - Use write-behind caching
  
- [ ] **Memory-only cache option**
  - For real-time mode, skip persistence

## 🟢 Nice to Have - Optimizations

### Quality Mode Preprocessing (140ms → target <50ms)
- [ ] Optimize shadow removal algorithm
- [ ] GPU-accelerate image enhancement (CUDA/OpenCV)

### Detection Consistency
- [ ] Reduce cold start time (1.9s → <500ms)
  - Pre-warm model on startup
  - Keep model in GPU memory

## 📊 Benchmark Targets

| Component | Current | Target | Priority |
|-----------|---------|--------|----------|
| Identification | 2000ms | <100ms | 🔴 Critical |
| Pipeline | 150ms | <100ms | 🟡 Important |
| Cache Write | 18ms | <5ms | 🟡 Important |
| Detection | 17ms | <20ms | ✅ Done |
| Cache Read | 0.003ms | <1ms | ✅ Done |

## 📝 Notes

- Detection performance is excellent (59-63 FPS)
- Cache read is sub-microsecond - no improvements needed
- GPU memory usage is low (58MB peak) - room for larger models
