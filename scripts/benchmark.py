#!/usr/bin/env python3
"""
Comprehensive Performance Benchmarking Suite for PriceLens

Measures and tracks performance of:
- Detection (YOLO inference)
- Identification (FeatureMatcher)
- Preprocessing (ImageEnhancer)
- End-to-end pipeline

Usage:
    python scripts/benchmark.py --all                    # Run all benchmarks
    python scripts/benchmark.py --detection             # Detection only
    python scripts/benchmark.py --quick                 # Quick sanity check
    python scripts/benchmark.py --all --output results.json  # Save results
"""

import argparse
import gc
import json
import logging
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging - suppress noisy warnings during benchmarks
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Suppress specific noisy loggers
for noisy_logger in ['src.detection.yolo_detector', 'ultralytics', 'PIL', 'easyocr']:
    logging.getLogger(noisy_logger).setLevel(logging.ERROR)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    iterations: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput: float  # items/second
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass 
class GPUInfo:
    """GPU information and metrics."""
    available: bool = False
    name: str = "N/A"
    memory_total_mb: float = 0
    memory_used_mb: float = 0
    memory_free_mb: float = 0
    cuda_version: str = "N/A"
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SystemInfo:
    """System information for reproducibility."""
    python_version: str = ""
    platform: str = ""
    processor: str = ""
    opencv_version: str = ""
    pytorch_version: str = "N/A"
    gpu: GPUInfo = field(default_factory=GPUInfo)
    
    def to_dict(self) -> dict:
        result = asdict(self)
        result['gpu'] = self.gpu.to_dict()
        return result


class BenchmarkSuite:
    """Comprehensive benchmark suite for PriceLens components."""
    
    def __init__(self, 
                 iterations: int = 100,
                 warmup: int = 10,
                 card_db_path: Optional[str] = None):
        self.iterations = iterations
        self.warmup = warmup
        self.results: Dict[str, BenchmarkResult] = {}
        self.system_info = self._collect_system_info()
        
        # Paths
        self.root = Path(__file__).parent.parent
        self.card_db_path = Path(card_db_path) if card_db_path else self.root / "data" / "card_database"
        
        # Lazy-loaded components
        self._detector = None
        self._matcher = None
        self._enhancer = None
        self._test_images = None
        
    def _collect_system_info(self) -> SystemInfo:
        """Collect system information for benchmark reproducibility."""
        info = SystemInfo(
            python_version=platform.python_version(),
            platform=platform.platform(),
            processor=platform.processor() or "Unknown",
            opencv_version=cv2.__version__,
        )
        
        # Check PyTorch and CUDA
        try:
            import torch
            info.pytorch_version = torch.__version__
            
            if torch.cuda.is_available():
                gpu = GPUInfo(
                    available=True,
                    name=torch.cuda.get_device_name(0),
                    cuda_version=torch.version.cuda or "Unknown",
                )
                
                # Get memory info
                try:
                    gpu.memory_total_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                    gpu.memory_used_mb = torch.cuda.memory_allocated(0) / 1024 / 1024
                    gpu.memory_free_mb = gpu.memory_total_mb - gpu.memory_used_mb
                except Exception:
                    pass
                    
                info.gpu = gpu
        except ImportError:
            pass
            
        return info
    
    def _get_test_images(self) -> Dict[str, np.ndarray]:
        """Load test images from card database."""
        if self._test_images is not None:
            return self._test_images
            
        self._test_images = {}
        
        # Create synthetic test images for baseline
        self._test_images["synthetic_720p"] = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        self._test_images["synthetic_1080p"] = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        self._test_images["synthetic_card"] = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
        
        # Load real card images from database
        if self.card_db_path.exists():
            # Pick a few representative cards
            sample_cards = [
                "base1/Charizard_base1-4.jpg",
                "base1/Pikachu_base1-58.jpg",
                "base1/Blastoise_base1-2.jpg",
            ]
            
            for card_path in sample_cards:
                full_path = self.card_db_path / card_path
                if full_path.exists():
                    img = cv2.imread(str(full_path))
                    if img is not None:
                        name = Path(card_path).stem
                        self._test_images[f"real_{name}"] = img
                        
                        # Create a simulated "camera capture" by resizing and adding noise
                        camera_sim = cv2.resize(img, (400, 560))
                        noise = np.random.normal(0, 10, camera_sim.shape).astype(np.int16)
                        camera_sim = np.clip(camera_sim.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                        self._test_images[f"camera_{name}"] = camera_sim
                        
        return self._test_images
    
    def _time_function(self, 
                       func, 
                       *args, 
                       iterations: Optional[int] = None, 
                       warmup: Optional[int] = None,
                       **kwargs) -> List[float]:
        """Time a function over multiple iterations and return timings in ms."""
        iterations = iterations or self.iterations
        warmup = warmup or self.warmup
        
        # Warmup runs
        for _ in range(warmup):
            func(*args, **kwargs)
        
        # Force garbage collection before timing
        gc.collect()
        
        # Timed runs
        timings = []
        for _ in range(iterations):
            start = time.perf_counter()
            func(*args, **kwargs)
            end = time.perf_counter()
            timings.append((end - start) * 1000)  # Convert to ms
            
        return timings
    
    def _compute_result(self, name: str, timings: List[float]) -> BenchmarkResult:
        """Compute statistics from timing data."""
        sorted_timings = sorted(timings)
        n = len(timings)
        
        return BenchmarkResult(
            name=name,
            iterations=n,
            mean_ms=statistics.mean(timings),
            std_ms=statistics.stdev(timings) if n > 1 else 0,
            min_ms=min(timings),
            max_ms=max(timings),
            p50_ms=sorted_timings[int(n * 0.50)],
            p95_ms=sorted_timings[int(n * 0.95)],
            p99_ms=sorted_timings[int(n * 0.99)],
            throughput=1000.0 / statistics.mean(timings) if statistics.mean(timings) > 0 else 0,
        )
    
    # =========================================================================
    # DETECTION BENCHMARKS
    # =========================================================================
    
    def benchmark_detection(self, quick: bool = False) -> Dict[str, BenchmarkResult]:
        """Benchmark YOLO card detection performance."""
        results = {}
        iterations = 20 if quick else self.iterations
        
        print("\n" + "=" * 60)
        print("DETECTION BENCHMARK (YOLOCardDetector)")
        print("=" * 60)
        
        try:
            from src.detection.yolo_detector import YOLOCardDetector
            from src.config import Config
            
            # Load config
            config_path = self.root / "config.yaml"
            config = Config(str(config_path)) if config_path.exists() else None
            
            # Initialize detector
            model_path = self.root / "models" / "pokemon_card_detector_latest.pt"
            if not model_path.exists():
                model_path = self.root / "models" / "yolo11n.pt"
            
            if not model_path.exists():
                print("  ⚠ No YOLO model found, skipping detection benchmark")
                return results
                
            print(f"  Loading model: {model_path.name}")
            detector = YOLOCardDetector(
                model_path=str(model_path),
                conf_threshold=0.35,
                use_card_specific_model=True,
            )
            
            test_images = self._get_test_images()
            
            # Cold start timing (first inference)
            print("  Measuring cold start...")
            cold_frame = test_images["synthetic_720p"].copy()
            cold_start = time.perf_counter()
            detector.detect(cold_frame)
            cold_end = time.perf_counter()
            cold_time = (cold_end - cold_start) * 1000
            print(f"  Cold start: {cold_time:.2f}ms")
            
            # Benchmark different resolutions
            for res_name in ["synthetic_720p", "synthetic_1080p"]:
                if res_name not in test_images:
                    continue
                    
                frame = test_images[res_name]
                print(f"  Benchmarking {res_name} ({frame.shape[1]}x{frame.shape[0]})...")
                
                timings = self._time_function(
                    detector.detect, 
                    frame,
                    iterations=iterations,
                )
                
                result = self._compute_result(f"detection_{res_name}", timings)
                results[result.name] = result
                self.results[result.name] = result
                
                print(f"    Mean: {result.mean_ms:.2f}ms | P95: {result.p95_ms:.2f}ms | FPS: {result.throughput:.1f}")
            
            # Benchmark with real card images
            for name, img in test_images.items():
                if name.startswith("real_"):
                    # Create a "frame" with the card in it
                    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    h, w = img.shape[:2]
                    scale = min(600 / h, 400 / w)
                    new_h, new_w = int(h * scale), int(w * scale)
                    resized = cv2.resize(img, (new_w, new_h))
                    
                    y_off = (720 - new_h) // 2
                    x_off = (1280 - new_w) // 2
                    frame[y_off:y_off+new_h, x_off:x_off+new_w] = resized
                    
                    print(f"  Benchmarking {name}...")
                    timings = self._time_function(
                        detector.detect,
                        frame,
                        iterations=iterations // 2,  # Fewer iterations for real images
                    )
                    
                    result = self._compute_result(f"detection_{name}", timings)
                    results[result.name] = result
                    self.results[result.name] = result
                    
                    print(f"    Mean: {result.mean_ms:.2f}ms | Detections will vary")
            
            # GPU memory usage (if available)
            self._report_gpu_memory("After detection benchmark")
            
        except Exception as e:
            print(f"  ❌ Detection benchmark failed: {e}")
            logger.exception("Detection benchmark error")
            
        return results
    
    # =========================================================================
    # IDENTIFICATION BENCHMARKS
    # =========================================================================
    
    def benchmark_identification(self, quick: bool = False) -> Dict[str, BenchmarkResult]:
        """Benchmark FeatureMatcher card identification performance."""
        results = {}
        iterations = 20 if quick else self.iterations
        
        print("\n" + "=" * 60)
        print("IDENTIFICATION BENCHMARK (FeatureMatcher)")
        print("=" * 60)
        
        try:
            from src.identification.feature_matcher import FeatureMatcher
            
            # Initialize matcher
            print("  Initializing FeatureMatcher...")
            matcher = FeatureMatcher(
                n_features=500,
                match_threshold=0.75,
                min_matches=10,
                use_ocr=False,  # Disable OCR for speed testing
                use_vector_db=False,
            )
            
            # Load feature database
            features_path = self.root / "data" / "me1_me2_features.pkl"
            if features_path.exists():
                print(f"  Loading feature database: {features_path.name}")
                matcher.load_database(str(features_path))
            else:
                print("  ⚠ No feature database found, using empty database")
            
            test_images = self._get_test_images()
            
            # Benchmark feature extraction
            print("\n  Feature Extraction Benchmark:")
            for name, img in test_images.items():
                if not name.startswith(("real_", "camera_", "synthetic_card")):
                    continue
                    
                print(f"    {name}...")
                timings = self._time_function(
                    matcher.compute_features,
                    img,
                    iterations=iterations,
                )
                
                result = self._compute_result(f"feature_extraction_{name}", timings)
                results[result.name] = result
                self.results[result.name] = result
                
                print(f"      Mean: {result.mean_ms:.2f}ms | P95: {result.p95_ms:.2f}ms")
            
            # Benchmark full identification pipeline
            print("\n  Full Identification Benchmark:")
            for name, img in test_images.items():
                if not name.startswith(("real_", "camera_")):
                    continue
                    
                print(f"    {name}...")
                timings = self._time_function(
                    matcher.identify,
                    img,
                    iterations=iterations // 2,
                )
                
                result = self._compute_result(f"identification_{name}", timings)
                results[result.name] = result
                self.results[result.name] = result
                
                print(f"      Mean: {result.mean_ms:.2f}ms | P95: {result.p95_ms:.2f}ms")
            
        except Exception as e:
            print(f"  ❌ Identification benchmark failed: {e}")
            logger.exception("Identification benchmark error")
            
        return results
    
    # =========================================================================
    # PREPROCESSING BENCHMARKS
    # =========================================================================
    
    def benchmark_preprocessing(self, quick: bool = False) -> Dict[str, BenchmarkResult]:
        """Benchmark ImageEnhancer preprocessing performance."""
        results = {}
        iterations = 50 if quick else self.iterations * 2  # Preprocessing is fast, more iterations
        
        print("\n" + "=" * 60)
        print("PREPROCESSING BENCHMARK (ImageEnhancer)")
        print("=" * 60)
        
        try:
            from src.preprocessing.enhancer import ImageEnhancer
            
            test_images = self._get_test_images()
            
            # Test each mode
            for mode in ["speed", "balanced", "quality"]:
                print(f"\n  Mode: {mode}")
                
                enhancer = ImageEnhancer(config={
                    "enabled": True,
                    "mode": mode,
                })
                
                # Detection pipeline (full frame)
                frame = test_images["synthetic_720p"]
                print(f"    Detection pipeline (720p frame)...")
                timings = self._time_function(
                    enhancer.enhance_for_detection,
                    frame,
                    iterations=iterations,
                )
                
                result = self._compute_result(f"preprocess_detection_{mode}", timings)
                results[result.name] = result
                self.results[result.name] = result
                print(f"      Mean: {result.mean_ms:.2f}ms | P95: {result.p95_ms:.2f}ms")
                
                # Identification pipeline (card crop)
                card = test_images.get("camera_Charizard_base1-4", test_images["synthetic_card"])
                print(f"    Identification pipeline (card crop)...")
                timings = self._time_function(
                    enhancer.enhance_for_identification,
                    card,
                    iterations=iterations,
                )
                
                result = self._compute_result(f"preprocess_identification_{mode}", timings)
                results[result.name] = result
                self.results[result.name] = result
                print(f"      Mean: {result.mean_ms:.2f}ms | P95: {result.p95_ms:.2f}ms")
                
                # OCR pipeline
                print(f"    OCR pipeline (card crop)...")
                timings = self._time_function(
                    enhancer.enhance_for_ocr,
                    card,
                    iterations=iterations,
                )
                
                result = self._compute_result(f"preprocess_ocr_{mode}", timings)
                results[result.name] = result
                self.results[result.name] = result
                print(f"      Mean: {result.mean_ms:.2f}ms | P95: {result.p95_ms:.2f}ms")
                
        except Exception as e:
            print(f"  ❌ Preprocessing benchmark failed: {e}")
            logger.exception("Preprocessing benchmark error")
            
        return results
    
    # =========================================================================
    # END-TO-END PIPELINE BENCHMARKS
    # =========================================================================
    
    def benchmark_pipeline(self, quick: bool = False) -> Dict[str, BenchmarkResult]:
        """Benchmark full detection + identification pipeline."""
        results = {}
        iterations = 10 if quick else 50
        
        print("\n" + "=" * 60)
        print("END-TO-END PIPELINE BENCHMARK")
        print("=" * 60)
        
        try:
            from src.detection.yolo_detector import YOLOCardDetector
            from src.identification.feature_matcher import FeatureMatcher
            from src.preprocessing.enhancer import ImageEnhancer
            
            # Initialize all components
            model_path = self.root / "models" / "pokemon_card_detector_latest.pt"
            if not model_path.exists():
                model_path = self.root / "models" / "yolo11n.pt"
                
            if not model_path.exists():
                print("  ⚠ No YOLO model found, skipping pipeline benchmark")
                return results
            
            print("  Initializing pipeline components...")
            
            enhancer = ImageEnhancer(config={"enabled": True, "mode": "balanced"})
            
            detector = YOLOCardDetector(
                model_path=str(model_path),
                conf_threshold=0.35,
                enhancer=enhancer,
            )
            
            matcher = FeatureMatcher(
                n_features=500,
                use_ocr=False,
                use_vector_db=False,
                enhancer=enhancer,
            )
            
            features_path = self.root / "data" / "me1_me2_features.pkl"
            if features_path.exists():
                matcher.load_database(str(features_path))
            
            test_images = self._get_test_images()
            
            # Create test frames with cards
            def run_pipeline(frame):
                """Full pipeline: detect -> extract -> identify."""
                detections = detector.detect(frame)
                if detections:
                    regions = detector.extract_card_regions(frame, detections)
                    for region in regions:
                        matcher.identify(region)
                return detections
            
            # Test with synthetic frame (baseline)
            print("\n  Synthetic frame (no cards)...")
            frame = test_images["synthetic_720p"]
            timings = self._time_function(run_pipeline, frame, iterations=iterations)
            result = self._compute_result("pipeline_empty_frame", timings)
            results[result.name] = result
            self.results[result.name] = result
            print(f"    Mean: {result.mean_ms:.2f}ms | FPS: {result.throughput:.1f}")
            
            # Test with real card in frame
            for name, card_img in test_images.items():
                if not name.startswith("real_"):
                    continue
                    
                # Create frame with card
                frame = np.zeros((720, 1280, 3), dtype=np.uint8) + 50  # Gray background
                h, w = card_img.shape[:2]
                scale = min(500 / h, 350 / w)
                new_h, new_w = int(h * scale), int(w * scale)
                resized = cv2.resize(card_img, (new_w, new_h))
                
                y_off = (720 - new_h) // 2
                x_off = (1280 - new_w) // 2
                frame[y_off:y_off+new_h, x_off:x_off+new_w] = resized
                
                print(f"  {name}...")
                timings = self._time_function(run_pipeline, frame, iterations=iterations)
                result = self._compute_result(f"pipeline_{name}", timings)
                results[result.name] = result
                self.results[result.name] = result
                print(f"    Mean: {result.mean_ms:.2f}ms | FPS: {result.throughput:.1f}")
            
            # Report GPU memory
            self._report_gpu_memory("After pipeline benchmark")
            
        except Exception as e:
            print(f"  ❌ Pipeline benchmark failed: {e}")
            logger.exception("Pipeline benchmark error")
            
        return results
    
    # =========================================================================
    # GPU PROFILING (NVIDIA)
    # =========================================================================
    
    def profile_gpu_memory(self) -> Dict[str, float]:
        """Profile GPU memory usage during operations."""
        print("\n" + "=" * 60)
        print("GPU MEMORY PROFILING")
        print("=" * 60)
        
        memory_stats = {}
        
        try:
            import torch
            if not torch.cuda.is_available():
                print("  ⚠ CUDA not available")
                return memory_stats
                
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            baseline = torch.cuda.memory_allocated() / 1024 / 1024
            memory_stats["baseline_mb"] = baseline
            print(f"  Baseline memory: {baseline:.2f} MB")
            
            # Load detector and measure memory
            from src.detection.yolo_detector import YOLOCardDetector
            
            model_path = self.root / "models" / "pokemon_card_detector_latest.pt"
            if not model_path.exists():
                model_path = self.root / "models" / "yolo11n.pt"
                
            if model_path.exists():
                detector = YOLOCardDetector(str(model_path))
                after_load = torch.cuda.memory_allocated() / 1024 / 1024
                memory_stats["after_model_load_mb"] = after_load
                print(f"  After model load: {after_load:.2f} MB (+{after_load - baseline:.2f})")
                
                # Run inference
                test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
                detector.detect(test_frame)
                
                after_inference = torch.cuda.memory_allocated() / 1024 / 1024
                peak = torch.cuda.max_memory_allocated() / 1024 / 1024
                
                memory_stats["after_inference_mb"] = after_inference
                memory_stats["peak_mb"] = peak
                
                print(f"  After inference: {after_inference:.2f} MB")
                print(f"  Peak memory: {peak:.2f} MB")
                
        except ImportError:
            print("  ⚠ PyTorch not available for GPU profiling")
        except Exception as e:
            print(f"  ❌ GPU profiling failed: {e}")
            
        return memory_stats
    
    def _report_gpu_memory(self, context: str = ""):
        """Report current GPU memory usage."""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024 / 1024
                peak = torch.cuda.max_memory_allocated() / 1024 / 1024
                print(f"\n  📊 GPU Memory {context}: {allocated:.1f}MB (peak: {peak:.1f}MB)")
        except ImportError:
            pass
    
    # =========================================================================
    # RUN ALL BENCHMARKS
    # =========================================================================
    
    def run_all(self, quick: bool = False) -> Dict:
        """Run all benchmarks and return complete results."""
        start_time = time.time()
        
        print("\n" + "=" * 60)
        print("PRICELENS BENCHMARK SUITE")
        print("=" * 60)
        print(f"  Iterations: {20 if quick else self.iterations}")
        print(f"  Warmup: {5 if quick else self.warmup}")
        print(f"  Mode: {'Quick' if quick else 'Full'}")
        
        # System info
        self._print_system_info()
        
        # Run benchmarks
        self.benchmark_detection(quick)
        self.benchmark_identification(quick)
        self.benchmark_preprocessing(quick)
        self.benchmark_pipeline(quick)
        
        # GPU profiling
        gpu_memory = self.profile_gpu_memory()
        
        elapsed = time.time() - start_time
        
        # Summary
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        
        # Group by category
        categories = {
            "Detection": [],
            "Identification": [],
            "Preprocessing": [],
            "Pipeline": [],
        }
        
        for name, result in self.results.items():
            if name.startswith("detection_"):
                categories["Detection"].append(result)
            elif name.startswith(("identification_", "feature_")):
                categories["Identification"].append(result)
            elif name.startswith("preprocess_"):
                categories["Preprocessing"].append(result)
            elif name.startswith("pipeline_"):
                categories["Pipeline"].append(result)
        
        for category, results_list in categories.items():
            if results_list:
                print(f"\n{category}:")
                for r in results_list:
                    status = "✅" if r.mean_ms < 100 else "⚠️" if r.mean_ms < 200 else "❌"
                    print(f"  {status} {r.name}: {r.mean_ms:.2f}ms (P95: {r.p95_ms:.2f}ms)")
        
        print(f"\n⏱️  Total benchmark time: {elapsed:.1f}s")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_info": self.system_info.to_dict(),
            "config": {
                "iterations": 20 if quick else self.iterations,
                "warmup": 5 if quick else self.warmup,
                "quick_mode": quick,
            },
            "results": {name: result.to_dict() for name, result in self.results.items()},
            "gpu_memory": gpu_memory,
            "elapsed_seconds": elapsed,
        }
    
    def _print_system_info(self):
        """Print system information."""
        print("\n  System Information:")
        print(f"    Python: {self.system_info.python_version}")
        print(f"    Platform: {self.system_info.platform}")
        print(f"    OpenCV: {self.system_info.opencv_version}")
        print(f"    PyTorch: {self.system_info.pytorch_version}")
        
        if self.system_info.gpu.available:
            print(f"    GPU: {self.system_info.gpu.name}")
            print(f"    CUDA: {self.system_info.gpu.cuda_version}")
            print(f"    GPU Memory: {self.system_info.gpu.memory_total_mb:.0f}MB")


def main():
    parser = argparse.ArgumentParser(
        description="PriceLens Performance Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/benchmark.py --all                    Run all benchmarks
  python scripts/benchmark.py --quick                  Quick sanity check
  python scripts/benchmark.py --detection --id         Detection + identification
  python scripts/benchmark.py --all -o results.json    Save results to file
  python scripts/benchmark.py --all --iterations 200   More iterations
        """
    )
    
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--detection", action="store_true", help="Run detection benchmarks")
    parser.add_argument("--identification", "--id", action="store_true", help="Run identification benchmarks")
    parser.add_argument("--preprocessing", "--pre", action="store_true", help="Run preprocessing benchmarks")
    parser.add_argument("--pipeline", action="store_true", help="Run end-to-end pipeline benchmarks")
    parser.add_argument("--gpu", action="store_true", help="Run GPU memory profiling")
    parser.add_argument("--quick", action="store_true", help="Quick mode with fewer iterations")
    parser.add_argument("--iterations", "-n", type=int, default=100, help="Number of iterations (default: 100)")
    parser.add_argument("--warmup", "-w", type=int, default=10, help="Warmup iterations (default: 10)")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file for results")
    
    args = parser.parse_args()
    
    # Default to --all if no specific benchmark selected
    if not any([args.all, args.detection, args.identification, args.preprocessing, args.pipeline, args.gpu]):
        args.all = True
    
    suite = BenchmarkSuite(
        iterations=args.iterations,
        warmup=args.warmup,
    )
    
    if args.all:
        results = suite.run_all(quick=args.quick)
    else:
        results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": suite.system_info.to_dict(),
            "results": {},
        }
        
        suite._print_system_info()
        
        if args.detection:
            suite.benchmark_detection(quick=args.quick)
        if args.identification:
            suite.benchmark_identification(quick=args.quick)
        if args.preprocessing:
            suite.benchmark_preprocessing(quick=args.quick)
        if args.pipeline:
            suite.benchmark_pipeline(quick=args.quick)
        if args.gpu:
            results["gpu_memory"] = suite.profile_gpu_memory()
            
        results["results"] = {name: result.to_dict() for name, result in suite.results.items()}
    
    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📁 Results saved to: {output_path}")
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()