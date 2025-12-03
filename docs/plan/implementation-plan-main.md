# PriceLens Unified Implementation Plan
**Updated:** <!-- date on update -->  
**Goal:** Production-ready card detection + identification + pricing overlay with web client and foundations for mobile/advanced features.

## Phase 0 – Foundation (Done / Maintain)
- Modern packaging (`pyproject.toml`), pre-commit hooks, CI workflow (`.github/workflows/ci.yml`).
- Dependencies include EasyOCR, ChromaDB, FilterPy for enhanced ID/interpolation.
- Action: Keep deps up to date; ensure CI runs lint/tests when added.

## Phase 1 – Detection (Core)
**Objective:** Stable, fast card boxes.
- Current: YOLOCardDetector implemented with tracking/filters; FrameInterpolator present. Using COCO weights (`models/yolo11n.pt`/`yolo11m.pt`) → low accuracy for cards.
- Plan:
  1) Train/fetch card-specific weight (`models/card_yolo11n.pt`) per `card-detector-training-plan.md`.
  2) Wire config to use card weight; set `allowed_class_names={'card'}`, disable blocked COCO classes.
  3) Tune runtime: imgsz 640–736, conf 0.35–0.4, tracking confirmation hits=2, optional face filter off.
  4) Add timing logs for encode→YOLO→track to measure latency.

## Phase 2 – Identification
**Objective:** Correct card ID quickly.
- Current: ORB+FLANN matcher with OCR and ChromaDB fallback; precomputed features exist (`data/features/*.pkl`).
- Plan:
  1) Pre-filter candidates (e.g., vector top-K) before ORB to avoid full DB scan.
  2) Add FLANN index or shard features to reduce per-frame matching cost; thread pool per crop.
  3) Cache last good match per tracked card_id; re-run only if bbox changes.
  4) Guard low-keypoint crops; request better frame instead of noisy matches.

## Phase 3 – Pricing
**Objective:** Reliable, cached prices in live loop.
- Current: PriceService with SmartCache; live detect returns cached price only; price preloader exists.
- Plan:
  1) Ensure live endpoint falls back to async fetch when cache miss; surface price once ready.
  2) Expose mock/real price providers via config; add error handling telemetry.
  3) Persist price history (SQLite) for trend display later.

## Phase 4 – Overlay & Web Client
**Objective:** Responsive UI with correct totals.
- Current: Web canvas overlay with session tracker, audio, multi-currency, running total; static renderer for still images.
- Plan:
  1) Increase client detect cadence (e.g., 200–250ms) and align with 2-hit tracking for faster lock-in.
  2) Add debug overlay for timing/confidence to spot bottlenecks.
  3) Keep total/session consistent with normalized price parsing (done); add clear/reset UX.

## Phase 5 – Architecture & Extensibility
- Current: EventBus and PluginManager exist but unused.
- Plan:
  1) Emit/subscribe for `frame.captured`, `cards.detected`, `cards.identified`, `prices.fetched`, `frame.rendered`.
  2) Add lightweight plugins (e.g., card history, high-value alert) to validate the bus.

## Phase 6 – Data & Storage
- Plan:
  1) Schema for scans/prices in SQLite; minimal models to store card scans and totals.
  2) CLI/API to export sessions; prepare for portfolio features later.

## Phase 7 – UI/UX Enhancements
- Plan:
  1) Web dashboard (FastAPI routes + frontend) for recent cards, totals, and history.
  2) Overlay polish: rarity/set badges, trend arrows, small charts (defer if core perf not stable).

## Phase 8 – Mobile & Products (Deferred)
- Flutter app, product detection/pricing, multi-language DBs, and collection/portfolio analytics are future phases once core detection/ID/pricing are solid.

## Immediate Next Actions (Do now)
1) Train or obtain `models/card_yolo11n.pt`; update config to use it with class allowlist and tuned conf/imgsz.
2) Speed up live loop: client interval ~200–250ms; tracking confirmation=2; optional face-filter toggle off for card model.
3) Identification perf: add candidate prefilter (vector top-K) before ORB; cache last match per tracked card; guard low-keypoint crops.
4) Wire live price fetch fallback when cache miss, so committed cards always get a price later.
