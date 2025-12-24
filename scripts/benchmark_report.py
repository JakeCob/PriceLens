#!/usr/bin/env python3
"""
Benchmark Report Generator for PriceLens

Generates comparison reports from benchmark results and detects regressions.

Usage:
    python scripts/benchmark_report.py results.json                    # Summary report
    python scripts/benchmark_report.py --compare baseline.json new.json # Compare runs
    python scripts/benchmark_report.py results.json -o report.md       # Save markdown
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_results(filepath: str) -> Dict:
    """Load benchmark results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def format_time(ms: float) -> str:
    """Format milliseconds for display."""
    if ms < 1:
        return f"{ms*1000:.1f}µs"
    elif ms < 1000:
        return f"{ms:.2f}ms"
    else:
        return f"{ms/1000:.2f}s"


def get_status_emoji(value: float, thresholds: Tuple[float, float]) -> str:
    """Get status emoji based on thresholds (good, warning)."""
    good, warning = thresholds
    if value <= good:
        return "✅"
    elif value <= warning:
        return "⚠️"
    else:
        return "❌"


def get_change_indicator(old: float, new: float, lower_is_better: bool = True) -> str:
    """Get change indicator with percentage."""
    if old == 0:
        return "N/A"
    
    change_pct = ((new - old) / old) * 100
    
    if abs(change_pct) < 1:
        return "➖ No change"
    
    if lower_is_better:
        if change_pct < -5:
            return f"🚀 -{abs(change_pct):.1f}% faster"
        elif change_pct > 10:
            return f"🔴 +{change_pct:.1f}% slower"
        elif change_pct > 5:
            return f"🟡 +{change_pct:.1f}% slower"
        else:
            return f"➖ {change_pct:+.1f}%"
    else:
        if change_pct > 5:
            return f"🚀 +{change_pct:.1f}% better"
        elif change_pct < -10:
            return f"🔴 {change_pct:.1f}% worse"
        else:
            return f"➖ {change_pct:+.1f}%"


def generate_summary_report(results: Dict) -> str:
    """Generate a summary markdown report from benchmark results."""
    lines = []
    
    # Header
    lines.append("# PriceLens Benchmark Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().isoformat()}")
    
    if "timestamp" in results:
        lines.append(f"**Benchmark Run:** {results['timestamp']}")
    
    lines.append("")
    
    # System Info
    if "system_info" in results:
        info = results["system_info"]
        lines.append("## System Information")
        lines.append("")
        lines.append(f"- **Python:** {info.get('python_version', 'N/A')}")
        lines.append(f"- **Platform:** {info.get('platform', 'N/A')}")
        lines.append(f"- **OpenCV:** {info.get('opencv_version', 'N/A')}")
        lines.append(f"- **PyTorch:** {info.get('pytorch_version', 'N/A')}")
        
        if info.get("gpu", {}).get("available"):
            gpu = info["gpu"]
            lines.append(f"- **GPU:** {gpu.get('name', 'N/A')}")
            lines.append(f"- **CUDA:** {gpu.get('cuda_version', 'N/A')}")
            lines.append(f"- **GPU Memory:** {gpu.get('memory_total_mb', 0):.0f}MB")
        
        lines.append("")
    
    # Results by category
    if "results" in results:
        benchmark_results = results["results"]
        
        # Categorize results
        categories = {
            "Detection": {},
            "Feature Extraction": {},
            "Identification": {},
            "Preprocessing": {},
            "Pipeline": {},
            "Cache": {},
            "API": {},
        }
        
        for name, data in benchmark_results.items():
            if name.startswith("detection_"):
                categories["Detection"][name] = data
            elif name.startswith("feature_"):
                categories["Feature Extraction"][name] = data
            elif name.startswith("identification_"):
                categories["Identification"][name] = data
            elif name.startswith("preprocess_"):
                categories["Preprocessing"][name] = data
            elif name.startswith("pipeline_"):
                categories["Pipeline"][name] = data
            elif name.startswith("cache_"):
                categories["Cache"][name] = data
            else:
                categories["API"][name] = data
        
        # Performance thresholds (good_ms, warning_ms)
        thresholds = {
            "Detection": (50, 100),
            "Feature Extraction": (20, 50),
            "Identification": (100, 200),
            "Preprocessing": (10, 25),
            "Pipeline": (150, 300),
            "Cache": (1, 10),
            "API": (10, 100),
        }
        
        lines.append("## Results")
        lines.append("")
        
        for category, items in categories.items():
            if not items:
                continue
                
            lines.append(f"### {category}")
            lines.append("")
            lines.append("| Benchmark | Mean | P95 | P99 | Throughput | Status |")
            lines.append("|-----------|------|-----|-----|------------|--------|")
            
            threshold = thresholds.get(category, (50, 100))
            
            for name, data in items.items():
                mean = data.get("mean_ms", 0)
                p95 = data.get("p95_ms", 0)
                p99 = data.get("p99_ms", 0)
                throughput = data.get("throughput", 0)
                status = get_status_emoji(mean, threshold)
                
                short_name = name.replace("detection_", "").replace("identification_", "")
                short_name = short_name.replace("preprocess_", "").replace("pipeline_", "")
                short_name = short_name.replace("feature_extraction_", "").replace("cache_", "")
                
                lines.append(f"| {short_name} | {format_time(mean)} | {format_time(p95)} | {format_time(p99)} | {throughput:.1f}/s | {status} |")
            
            lines.append("")
    
    # GPU Memory
    if "gpu_memory" in results and results["gpu_memory"]:
        lines.append("## GPU Memory")
        lines.append("")
        gpu_mem = results["gpu_memory"]
        lines.append(f"- **Baseline:** {gpu_mem.get('baseline_mb', 0):.1f}MB")
        lines.append(f"- **After Model Load:** {gpu_mem.get('after_model_load_mb', 0):.1f}MB")
        lines.append(f"- **After Inference:** {gpu_mem.get('after_inference_mb', 0):.1f}MB")
        lines.append(f"- **Peak:** {gpu_mem.get('peak_mb', 0):.1f}MB")
        lines.append("")
    
    # Summary
    lines.append("## Summary")
    lines.append("")
    
    if "elapsed_seconds" in results:
        lines.append(f"Total benchmark time: **{results['elapsed_seconds']:.1f}s**")
    
    # Count status
    if "results" in results:
        total = len(results["results"])
        fast = sum(1 for d in results["results"].values() if d.get("mean_ms", 0) < 50)
        slow = sum(1 for d in results["results"].values() if d.get("mean_ms", 0) > 100)
        
        lines.append(f"- **Total benchmarks:** {total}")
        lines.append(f"- **Fast (<50ms):** {fast}")
        lines.append(f"- **Slow (>100ms):** {slow}")
    
    lines.append("")
    
    return "\n".join(lines)


def generate_comparison_report(baseline: Dict, current: Dict) -> str:
    """Generate a comparison report between two benchmark runs."""
    lines = []
    
    lines.append("# PriceLens Benchmark Comparison Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().isoformat()}")
    lines.append(f"**Baseline:** {baseline.get('timestamp', 'Unknown')}")
    lines.append(f"**Current:** {current.get('timestamp', 'Unknown')}")
    lines.append("")
    
    # Regressions check
    regressions = []
    improvements = []
    
    baseline_results = baseline.get("results", {})
    current_results = current.get("results", {})
    
    lines.append("## Performance Changes")
    lines.append("")
    lines.append("| Benchmark | Baseline | Current | Change |")
    lines.append("|-----------|----------|---------|--------|")
    
    for name in sorted(set(baseline_results.keys()) | set(current_results.keys())):
        base_data = baseline_results.get(name, {})
        curr_data = current_results.get(name, {})
        
        base_mean = base_data.get("mean_ms", 0)
        curr_mean = curr_data.get("mean_ms", 0)
        
        if base_mean == 0 and curr_mean == 0:
            continue
        
        change = get_change_indicator(base_mean, curr_mean, lower_is_better=True)
        
        # Track regressions (>10% slower)
        if base_mean > 0:
            pct_change = ((curr_mean - base_mean) / base_mean) * 100
            if pct_change > 10:
                regressions.append((name, base_mean, curr_mean, pct_change))
            elif pct_change < -10:
                improvements.append((name, base_mean, curr_mean, pct_change))
        
        base_str = format_time(base_mean) if base_mean else "N/A"
        curr_str = format_time(curr_mean) if curr_mean else "N/A"
        
        lines.append(f"| {name} | {base_str} | {curr_str} | {change} |")
    
    lines.append("")
    
    # Regressions section
    if regressions:
        lines.append("## ⚠️ Regressions Detected")
        lines.append("")
        lines.append("The following benchmarks are significantly slower:")
        lines.append("")
        for name, base, curr, pct in sorted(regressions, key=lambda x: -x[3]):
            lines.append(f"- **{name}**: {format_time(base)} → {format_time(curr)} (+{pct:.1f}%)")
        lines.append("")
    
    # Improvements section
    if improvements:
        lines.append("## 🚀 Improvements")
        lines.append("")
        lines.append("The following benchmarks are significantly faster:")
        lines.append("")
        for name, base, curr, pct in sorted(improvements, key=lambda x: x[3]):
            lines.append(f"- **{name}**: {format_time(base)} → {format_time(curr)} ({pct:.1f}%)")
        lines.append("")
    
    # Overall summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Regressions:** {len(regressions)}")
    lines.append(f"- **Improvements:** {len(improvements)}")
    lines.append(f"- **Stable:** {len(current_results) - len(regressions) - len(improvements)}")
    lines.append("")
    
    if regressions:
        lines.append("> [!WARNING]")
        lines.append("> Performance regressions detected! Review the changes before merging.")
    elif improvements:
        lines.append("> [!NOTE]")
        lines.append("> Performance improvements detected. Good job! 🎉")
    else:
        lines.append("> [!NOTE]")
        lines.append("> Performance is stable. No significant changes detected.")
    
    lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark comparison reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/benchmark_report.py results.json
  python scripts/benchmark_report.py --compare baseline.json current.json
  python scripts/benchmark_report.py results.json -o report.md
        """
    )
    
    parser.add_argument("results", nargs="?", help="Benchmark results JSON file")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "CURRENT"),
                        help="Compare two benchmark runs")
    parser.add_argument("--output", "-o", type=str, help="Output markdown file")
    
    args = parser.parse_args()
    
    if args.compare:
        baseline = load_results(args.compare[0])
        current = load_results(args.compare[1])
        report = generate_comparison_report(baseline, current)
    elif args.results:
        results = load_results(args.results)
        report = generate_summary_report(results)
    else:
        parser.print_help()
        sys.exit(1)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"📁 Report saved to: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
