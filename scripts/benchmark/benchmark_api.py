#!/usr/bin/env python3
"""
API Performance Benchmarking for PriceLens

Tests the PriceService, SmartCache, and fallback chain performance.

Usage:
    python scripts/benchmark_api.py                     # Run all API benchmarks
    python scripts/benchmark_api.py --cache-only        # Cache tests only
    python scripts/benchmark_api.py -o api_results.json # Save results
"""

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class APIBenchmarkResult:
    """Results from API benchmark."""
    name: str
    iterations: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    cache_hit_rate: float = 0.0
    success_rate: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)


class APIBenchmarkSuite:
    """Benchmark suite for PriceLens API components."""
    
    def __init__(self, iterations: int = 50):
        self.iterations = iterations
        self.results: Dict[str, APIBenchmarkResult] = {}
        
        # Test card IDs from the database
        self.test_card_ids = [
            "base1-4",   # Charizard
            "base1-2",   # Blastoise  
            "base1-58",  # Pikachu
            "base1-15",  # Venusaur
            "me1-1",     # Dark Alakazam
            "nonexistent-999",  # For error handling tests
        ]
    
    def _time_function(self, func, *args, iterations: int = None, **kwargs) -> tuple:
        """Time a function and return (timings_ms, successes)."""
        iterations = iterations or self.iterations
        timings = []
        successes = 0
        
        for _ in range(iterations):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                if result is not None:
                    successes += 1
            except Exception:
                pass
            end = time.perf_counter()
            timings.append((end - start) * 1000)
            
        return timings, successes
    
    def _compute_result(self, name: str, timings: List[float], 
                        cache_hits: int = 0, successes: int = 0) -> APIBenchmarkResult:
        """Compute statistics from timing data."""
        sorted_timings = sorted(timings)
        n = len(timings)
        
        return APIBenchmarkResult(
            name=name,
            iterations=n,
            mean_ms=statistics.mean(timings),
            std_ms=statistics.stdev(timings) if n > 1 else 0,
            min_ms=min(timings),
            max_ms=max(timings),
            p50_ms=sorted_timings[int(n * 0.50)],
            p95_ms=sorted_timings[int(n * 0.95)],
            cache_hit_rate=cache_hits / n if n > 0 else 0,
            success_rate=successes / n if n > 0 else 0,
        )
    
    def benchmark_smart_cache(self) -> Dict[str, APIBenchmarkResult]:
        """Benchmark SmartCache performance."""
        results = {}
        
        print("\n" + "=" * 60)
        print("SMART CACHE BENCHMARK")
        print("=" * 60)
        
        try:
            from src.api.smart_cache import SmartCache
            from src.api.base import PriceData
            
            cache = SmartCache()
            
            # Create test price data using correct fields
            test_price = PriceData(
                market=100.0,
                low=90.0,
                high=110.0,
                currency="USD",
            )
            
            # Benchmark cache write
            print("\n  Cache Write:")
            def cache_write():
                cache.set(f"test-{time.time()}", test_price)
            
            timings, _ = self._time_function(cache_write, iterations=self.iterations)
            result = self._compute_result("cache_write", timings)
            results[result.name] = result
            self.results[result.name] = result
            print(f"    Mean: {result.mean_ms:.3f}ms | P95: {result.p95_ms:.3f}ms")
            
            # Prime cache for read tests
            for i in range(100):
                cache.set(f"read-test-{i}", test_price)
            
            # Benchmark cache read (hit)
            print("\n  Cache Read (Hit):")
            def cache_read_hit():
                return cache.get("read-test-50")
            
            timings, successes = self._time_function(cache_read_hit, iterations=self.iterations * 2)
            result = self._compute_result("cache_read_hit", timings, successes=successes)
            results[result.name] = result
            self.results[result.name] = result
            print(f"    Mean: {result.mean_ms:.3f}ms | P95: {result.p95_ms:.3f}ms | Hit: {result.success_rate*100:.0f}%")
            
            # Benchmark cache read (miss)
            print("\n  Cache Read (Miss):")
            def cache_read_miss():
                return cache.get("nonexistent-key")
            
            timings, successes = self._time_function(cache_read_miss, iterations=self.iterations)
            result = self._compute_result("cache_read_miss", timings, successes=successes)
            results[result.name] = result
            self.results[result.name] = result
            print(f"    Mean: {result.mean_ms:.3f}ms | P95: {result.p95_ms:.3f}ms")
            
            # Benchmark stale read
            print("\n  Cache Stale Read:")
            def cache_stale_read():
                return cache.get_stale("read-test-25")
            
            timings, successes = self._time_function(cache_stale_read, iterations=self.iterations)
            result = self._compute_result("cache_stale_read", timings, successes=successes)
            results[result.name] = result
            self.results[result.name] = result
            print(f"    Mean: {result.mean_ms:.3f}ms | P95: {result.p95_ms:.3f}ms")
            
        except Exception as e:
            print(f"  ❌ Cache benchmark failed: {e}")
            
        return results
    
    def benchmark_price_service(self, skip_network: bool = True) -> Dict[str, APIBenchmarkResult]:
        """Benchmark PriceService performance."""
        results = {}
        
        print("\n" + "=" * 60)
        print("PRICE SERVICE BENCHMARK")
        print("=" * 60)
        
        if skip_network:
            print("  (Network calls skipped - testing cache/local only)")
        
        try:
            from src.api.service import PriceService
            
            service = PriceService(enable_history=False)
            
            # Warm the cache with some prices first
            print("\n  Warming cache...")
            for card_id in self.test_card_ids[:3]:
                try:
                    service.get_price(card_id, timeout_seconds=2)
                except Exception:
                    pass
            
            # Benchmark cached price fetch
            print("\n  Price Fetch (Cached):")
            cached_card = self.test_card_ids[0]  # Charizard should be cached
            
            def get_cached_price():
                return service.get_price(cached_card)
            
            timings, successes = self._time_function(get_cached_price, iterations=self.iterations)
            result = self._compute_result("price_fetch_cached", timings, successes=successes)
            results[result.name] = result
            self.results[result.name] = result
            print(f"    Mean: {result.mean_ms:.2f}ms | P95: {result.p95_ms:.2f}ms | Success: {result.success_rate*100:.0f}%")
            
            # Benchmark price with trend
            print("\n  Price With Trend:")
            def get_price_trend():
                return service.get_price_with_trend(cached_card)
            
            timings, successes = self._time_function(get_price_trend, iterations=self.iterations)
            result = self._compute_result("price_with_trend", timings, successes=successes)
            results[result.name] = result
            self.results[result.name] = result
            print(f"    Mean: {result.mean_ms:.2f}ms | P95: {result.p95_ms:.2f}ms")
            
            # Benchmark price dict (API response format)
            print("\n  Price Dict (API Response):")
            def get_price_dict():
                return service.get_price_dict(cached_card)
            
            timings, successes = self._time_function(get_price_dict, iterations=self.iterations)
            result = self._compute_result("price_dict", timings, successes=successes)
            results[result.name] = result
            self.results[result.name] = result
            print(f"    Mean: {result.mean_ms:.2f}ms | P95: {result.p95_ms:.2f}ms")
            
            # Get service status
            print("\n  Service Status:")
            status = service.get_status()
            print(f"    Circuit Breaker: {status.get('api_circuit', 'N/A')}")
            print(f"    Failure Cache: {status.get('failure_cache', 'N/A')}")
            print(f"    Scraper Available: {status.get('scraper_available', False)}")
            
        except Exception as e:
            print(f"  ❌ Price service benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            
        return results
    
    def benchmark_custom_source(self) -> Dict[str, APIBenchmarkResult]:
        """Benchmark CustomPriceSource (local JSON) performance."""
        results = {}
        
        print("\n" + "=" * 60)
        print("CUSTOM PRICE SOURCE BENCHMARK")
        print("=" * 60)
        
        try:
            from src.api.custom_source import CustomPriceSource
            
            source = CustomPriceSource()
            
            # Benchmark custom source lookup
            print("\n  Custom Source Lookup:")
            def lookup_custom():
                return source.get_price("base1-4")  # Charizard
            
            timings, successes = self._time_function(lookup_custom, iterations=self.iterations * 2)
            result = self._compute_result("custom_source_lookup", timings, successes=successes)
            results[result.name] = result
            self.results[result.name] = result
            print(f"    Mean: {result.mean_ms:.3f}ms | P95: {result.p95_ms:.3f}ms | Found: {result.success_rate*100:.0f}%")
            
        except Exception as e:
            print(f"  ❌ Custom source benchmark failed: {e}")
            
        return results
    
    def run_all(self, skip_network: bool = True) -> Dict:
        """Run all API benchmarks."""
        start_time = time.time()
        
        print("\n" + "=" * 60)
        print("PRICELENS API BENCHMARK SUITE")
        print("=" * 60)
        
        self.benchmark_smart_cache()
        self.benchmark_custom_source()
        self.benchmark_price_service(skip_network=skip_network)
        
        elapsed = time.time() - start_time
        
        # Summary
        print("\n" + "=" * 60)
        print("API BENCHMARK SUMMARY")
        print("=" * 60)
        
        for name, result in self.results.items():
            status = "✅" if result.mean_ms < 10 else "⚠️" if result.mean_ms < 100 else "❌"
            print(f"  {status} {name}: {result.mean_ms:.2f}ms (P95: {result.p95_ms:.2f}ms)")
        
        print(f"\n⏱️  Total: {elapsed:.1f}s")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "iterations": self.iterations,
                "skip_network": skip_network,
            },
            "results": {name: result.to_dict() for name, result in self.results.items()},
            "elapsed_seconds": elapsed,
        }


def main():
    parser = argparse.ArgumentParser(description="PriceLens API Benchmark Suite")
    parser.add_argument("--cache-only", action="store_true", help="Only benchmark cache")
    parser.add_argument("--with-network", action="store_true", help="Include network calls (slower)")
    parser.add_argument("--iterations", "-n", type=int, default=50, help="Iterations (default: 50)")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file")
    
    args = parser.parse_args()
    
    suite = APIBenchmarkSuite(iterations=args.iterations)
    
    if args.cache_only:
        results = {
            "timestamp": datetime.now().isoformat(),
            "results": {},
        }
        suite.benchmark_smart_cache()
        results["results"] = {name: r.to_dict() for name, r in suite.results.items()}
    else:
        results = suite.run_all(skip_network=not args.with_network)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📁 Results saved to: {args.output}")
    
    print("\n✅ API benchmark complete!")


if __name__ == "__main__":
    main()
