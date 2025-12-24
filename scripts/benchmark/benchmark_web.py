#!/usr/bin/env python3
"""
Web API Performance Benchmarking for PriceLens

Tests the HTTP endpoints for detection and analysis.
Requires the API server to be running.

Usage:
    # Start server first:
    python scripts/run_api.py
    
    # Then run benchmarks:
    python scripts/benchmark_web.py                      # All endpoints
    python scripts/benchmark_web.py --endpoint detect    # Detection only
    python scripts/benchmark_web.py -n 100               # 100 iterations
"""

import argparse
import io
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    import requests
except ImportError:
    print("Error: requests library required. Install with: pip install requests")
    sys.exit(1)

import numpy as np

try:
    import cv2
except ImportError:
    print("Error: OpenCV required. Install with: pip install opencv-python")
    sys.exit(1)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class WebBenchmarkResult:
    """Results from web endpoint benchmark."""
    endpoint: str
    iterations: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    requests_per_second: float
    success_rate: float
    avg_response_size_bytes: int
    
    def to_dict(self) -> dict:
        return asdict(self)


class WebBenchmarkSuite:
    """Benchmark suite for PriceLens web API endpoints."""
    
    def __init__(self, 
                 base_url: str = "http://localhost:7848",
                 iterations: int = 50):
        self.base_url = base_url.rstrip("/")
        self.iterations = iterations
        self.results: Dict[str, WebBenchmarkResult] = {}
        
        # Test images
        self.root = Path(__file__).parent.parent
        self._test_images = None
    
    def _check_server(self) -> bool:
        """Check if the API server is running."""
        try:
            resp = requests.get(f"{self.base_url}/", timeout=2)
            return resp.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def _get_test_images(self) -> Dict[str, bytes]:
        """Get test images as JPEG bytes."""
        if self._test_images is not None:
            return self._test_images
            
        self._test_images = {}
        
        # Create synthetic test images
        for name, shape in [("720p", (720, 1280, 3)), ("480p", (480, 640, 3))]:
            img = np.random.randint(0, 255, shape, dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            self._test_images[f"synthetic_{name}"] = buffer.tobytes()
        
        # Load real card images
        card_paths = [
            self.root / "data" / "card_database" / "base1" / "Charizard_base1-4.jpg",
            self.root / "data" / "card_database" / "base1" / "Pikachu_base1-58.jpg",
        ]
        
        for path in card_paths:
            if path.exists():
                img = cv2.imread(str(path))
                if img is not None:
                    # Create a frame with the card in it
                    frame = np.zeros((720, 1280, 3), dtype=np.uint8) + 50
                    h, w = img.shape[:2]
                    scale = min(500 / h, 350 / w)
                    new_h, new_w = int(h * scale), int(w * scale)
                    resized = cv2.resize(img, (new_w, new_h))
                    
                    y_off = (720 - new_h) // 2
                    x_off = (1280 - new_w) // 2
                    frame[y_off:y_off+new_h, x_off:x_off+new_w] = resized
                    
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    self._test_images[f"frame_{path.stem}"] = buffer.tobytes()
        
        return self._test_images
    
    def _benchmark_endpoint(self, 
                            endpoint: str,
                            image_data: bytes,
                            image_name: str,
                            iterations: Optional[int] = None) -> WebBenchmarkResult:
        """Benchmark a single endpoint with given image data."""
        iterations = iterations or self.iterations
        
        timings = []
        successes = 0
        response_sizes = []
        
        url = f"{self.base_url}/{endpoint}"
        
        for _ in range(iterations):
            files = {"file": (f"{image_name}.jpg", io.BytesIO(image_data), "image/jpeg")}
            
            start = time.perf_counter()
            try:
                resp = requests.post(url, files=files, timeout=30)
                if resp.status_code == 200:
                    successes += 1
                    response_sizes.append(len(resp.content))
            except requests.exceptions.RequestException:
                pass
            end = time.perf_counter()
            
            timings.append((end - start) * 1000)
        
        sorted_timings = sorted(timings)
        n = len(timings)
        
        return WebBenchmarkResult(
            endpoint=endpoint,
            iterations=n,
            mean_ms=statistics.mean(timings),
            std_ms=statistics.stdev(timings) if n > 1 else 0,
            min_ms=min(timings),
            max_ms=max(timings),
            p50_ms=sorted_timings[int(n * 0.50)],
            p95_ms=sorted_timings[int(n * 0.95)],
            p99_ms=sorted_timings[int(n * 0.99)],
            requests_per_second=1000.0 / statistics.mean(timings) if statistics.mean(timings) > 0 else 0,
            success_rate=successes / n if n > 0 else 0,
            avg_response_size_bytes=int(statistics.mean(response_sizes)) if response_sizes else 0,
        )
    
    def benchmark_detect_live(self) -> Dict[str, WebBenchmarkResult]:
        """Benchmark the /detect-live endpoint."""
        results = {}
        
        print("\n" + "=" * 60)
        print("DETECT-LIVE ENDPOINT BENCHMARK")
        print("=" * 60)
        
        test_images = self._get_test_images()
        
        for name, image_data in test_images.items():
            print(f"\n  Testing with {name}...")
            
            result = self._benchmark_endpoint(
                "detect-live",
                image_data,
                name,
                iterations=self.iterations,
            )
            
            result_name = f"detect_live_{name}"
            results[result_name] = result
            self.results[result_name] = result
            
            print(f"    Mean: {result.mean_ms:.1f}ms | P95: {result.p95_ms:.1f}ms")
            print(f"    RPS: {result.requests_per_second:.1f} | Success: {result.success_rate*100:.0f}%")
        
        return results
    
    def benchmark_analyze_image(self) -> Dict[str, WebBenchmarkResult]:
        """Benchmark the /analyze-image endpoint."""
        results = {}
        
        print("\n" + "=" * 60)
        print("ANALYZE-IMAGE ENDPOINT BENCHMARK")
        print("=" * 60)
        
        test_images = self._get_test_images()
        
        # Only test with real card images for analysis
        for name, image_data in test_images.items():
            if not name.startswith("frame_"):
                continue
                
            print(f"\n  Testing with {name}...")
            
            result = self._benchmark_endpoint(
                "analyze-image",
                image_data,
                name,
                iterations=self.iterations // 2,  # Fewer iterations, endpoint is slower
            )
            
            result_name = f"analyze_image_{name}"
            results[result_name] = result  
            self.results[result_name] = result
            
            print(f"    Mean: {result.mean_ms:.1f}ms | P95: {result.p95_ms:.1f}ms")
            print(f"    RPS: {result.requests_per_second:.1f} | Success: {result.success_rate*100:.0f}%")
        
        return results
    
    def run_all(self) -> Dict:
        """Run all web benchmarks."""
        start_time = time.time()
        
        print("\n" + "=" * 60)
        print("PRICELENS WEB API BENCHMARK SUITE")
        print("=" * 60)
        print(f"  Base URL: {self.base_url}")
        print(f"  Iterations: {self.iterations}")
        
        # Check server
        print("\n  Checking server...")
        if not self._check_server():
            print("  ❌ Server not running!")
            print("  Start with: python scripts/run_api.py")
            return {"error": "Server not running"}
        
        print("  ✅ Server is running")
        
        self.benchmark_detect_live()
        self.benchmark_analyze_image()
        
        elapsed = time.time() - start_time
        
        # Summary
        print("\n" + "=" * 60)
        print("WEB API BENCHMARK SUMMARY")
        print("=" * 60)
        
        for name, result in self.results.items():
            status = "✅" if result.mean_ms < 200 else "⚠️" if result.mean_ms < 500 else "❌"
            print(f"  {status} {name}: {result.mean_ms:.1f}ms (P95: {result.p95_ms:.1f}ms)")
        
        print(f"\n⏱️  Total: {elapsed:.1f}s")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "base_url": self.base_url,
                "iterations": self.iterations,
            },
            "results": {name: result.to_dict() for name, result in self.results.items()},
            "elapsed_seconds": elapsed,
        }


def main():
    parser = argparse.ArgumentParser(description="PriceLens Web API Benchmark Suite")
    parser.add_argument("--url", type=str, default="http://localhost:7848",
                        help="API base URL (default: http://localhost:7848)")
    parser.add_argument("--endpoint", type=str, choices=["detect", "analyze", "all"],
                        default="all", help="Endpoint to benchmark")
    parser.add_argument("--iterations", "-n", type=int, default=50, help="Iterations")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file")
    
    args = parser.parse_args()
    
    suite = WebBenchmarkSuite(
        base_url=args.url,
        iterations=args.iterations,
    )
    
    if args.endpoint == "all":
        results = suite.run_all()
    elif args.endpoint == "detect":
        suite._check_server()
        suite.benchmark_detect_live()
        results = {"results": {name: r.to_dict() for name, r in suite.results.items()}}
    else:
        suite._check_server()
        suite.benchmark_analyze_image()
        results = {"results": {name: r.to_dict() for name, r in suite.results.items()}}
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📁 Results saved to: {args.output}")
    
    print("\n✅ Web benchmark complete!")


if __name__ == "__main__":
    main()
