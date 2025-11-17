#!/usr/bin/env python3
"""
Comprehensive memory overhead analysis comparing DeepEP vs Baseline.
Analyzes ALL categories to identify sources of +50.80 GB overhead.
"""

import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def format_bytes(bytes_val):
    """Format bytes to human readable."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(bytes_val) < 1024.0:
            return f"{bytes_val:7.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:7.2f} PB"


def categorize_allocation(frames):
    """Categorize allocation based on stack trace."""
    frames_str = "\n".join([f"{f['filename']}:{f['line']} {f['name']}" for f in frames])

    # Check for various categories
    if any(
        keyword in frames_str
        for keyword in ["moe.py", "GroupedExperts", "TokenChoiceTopKRouter", "MoE"]
    ):
        return "Expert/MoE"
    elif any(
        keyword in frames_str
        for keyword in ["attention", "Attention", "CausalSelfAttention"]
    ):
        return "Attention"
    elif any(keyword in frames_str for keyword in ["linear", "Linear", "matmul", "mm"]):
        return "Linear"
    elif any(keyword in frames_str for keyword in ["optimizer", "Adam", "AdamW"]):
        return "Optimizer State"
    elif any(
        keyword in frames_str
        for keyword in ["all_to_all", "all_gather", "reduce_scatter", "broadcast"]
    ):
        return "Communication"
    elif any(
        keyword in frames_str
        for keyword in ["fsdp", "FSDP", "FullyShardedDataParallel"]
    ):
        return "FSDP"
    elif any(keyword in frames_str for keyword in ["gradient", "Gradient", "backward"]):
        return "Gradient"
    elif any(keyword in frames_str for keyword in ["embedding", "Embedding"]):
        return "Embedding"
    elif any(keyword in frames_str for keyword in ["norm", "LayerNorm", "RMSNorm"]):
        return "Normalization"
    elif any(
        keyword in frames_str
        for keyword in [
            "deepep",
            "deep_ep",
            "fused_dispatch",
            "fused_combine",
            "PrimusTurbo",
        ]
    ):
        return "DeepEP Overhead"
    else:
        return "Unknown"


def extract_allocation_info(segment):
    """Extract detailed information from a segment."""
    allocations = []

    if "blocks" in segment:
        for block in segment["blocks"]:
            if block["state"] == "active_allocated":
                size = block["size"]
                frames = block.get("frames", [])
                category = categorize_allocation(frames)

                # Get top frame for identification
                top_frame = ""
                if frames:
                    top_frame = f"{frames[0]['filename']}:{frames[0]['line']} {frames[0]['name']}"

                allocations.append(
                    {
                        "size": size,
                        "category": category,
                        "frames": frames,
                        "top_frame": top_frame,
                        "address": block.get("address", 0),
                    }
                )

    return allocations


def analyze_snapshot(pickle_path: Path) -> Tuple[Dict, List]:
    """Analyze a single snapshot file."""
    print(f"Loading snapshot: {pickle_path}")

    try:
        with open(pickle_path, "rb") as f:
            snapshot = pickle.load(f)
    except Exception as e:
        print(f"Error loading {pickle_path}: {e}")
        return {}, []

    # Collect all allocations
    all_allocations = []

    if "segments" in snapshot:
        for segment in snapshot["segments"]:
            allocations = extract_allocation_info(segment)
            all_allocations.extend(allocations)

    # Aggregate by category
    category_stats = defaultdict(
        lambda: {"total_bytes": 0, "count": 0, "allocations": []}
    )

    for alloc in all_allocations:
        cat = alloc["category"]
        category_stats[cat]["total_bytes"] += alloc["size"]
        category_stats[cat]["count"] += 1
        category_stats[cat]["allocations"].append(alloc)

    # Convert to regular dict and sort allocations by size
    category_stats = dict(category_stats)
    for cat in category_stats:
        category_stats[cat]["allocations"].sort(key=lambda x: x["size"], reverse=True)

    return category_stats, all_allocations


def print_category_summary(stats: Dict, title: str):
    """Print summary of all categories."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")

    total_bytes = sum(cat["total_bytes"] for cat in stats.values())

    # Sort by size
    sorted_cats = sorted(stats.items(), key=lambda x: x[1]["total_bytes"], reverse=True)

    print(f"\n{'Category':<25} {'Size':<15} {'Count':<10} {'Percent':<10}")
    print("-" * 80)

    for cat, data in sorted_cats:
        size = data["total_bytes"]
        count = data["count"]
        percent = 100 * size / total_bytes if total_bytes > 0 else 0
        print(f"{cat:<25} {format_bytes(size):<15} {count:<10} {percent:>6.2f}%")

    print("-" * 80)
    print(
        f"{'TOTAL':<25} {format_bytes(total_bytes):<15} {sum(c['count'] for c in stats.values()):<10} 100.00%"
    )


def print_top_allocations(stats: Dict, title: str, top_n: int = 50):
    """Print top N allocations across all categories."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")

    # Collect all allocations
    all_allocs = []
    for cat, data in stats.items():
        for alloc in data["allocations"]:
            all_allocs.append((cat, alloc))

    # Sort by size
    all_allocs.sort(key=lambda x: x[1]["size"], reverse=True)

    print(f"\n{'Rank':<6} {'Size':<15} {'Category':<20} {'Location':<50}")
    print("-" * 100)

    for i, (cat, alloc) in enumerate(all_allocs[:top_n], 1):
        size = format_bytes(alloc["size"])
        location = (
            alloc["top_frame"][:47] + "..."
            if len(alloc["top_frame"]) > 50
            else alloc["top_frame"]
        )
        print(f"{i:<6} {size:<15} {cat:<20} {location:<50}")


def compare_snapshots(deepep_stats: Dict, baseline_stats: Dict):
    """Compare DeepEP vs Baseline and show differences."""
    print(f"\n{'='*80}")
    print(f"MEMORY OVERHEAD COMPARISON: DeepEP vs Baseline")
    print(f"{'='*80}")

    # Get all categories
    all_categories = set(deepep_stats.keys()) | set(baseline_stats.keys())

    deepep_total = sum(cat["total_bytes"] for cat in deepep_stats.values())
    baseline_total = sum(cat["total_bytes"] for cat in baseline_stats.values())
    total_overhead = deepep_total - baseline_total

    print(f"\nTotal Active Memory:")
    print(f"  DeepEP:    {format_bytes(deepep_total)}")
    print(f"  Baseline:  {format_bytes(baseline_total)}")
    print(
        f"  Overhead:  {format_bytes(total_overhead)} ({100*total_overhead/baseline_total:.1f}% increase)"
    )

    print(
        f"\n{'Category':<25} {'DeepEP':<15} {'Baseline':<15} {'Overhead':<15} {'% of Total':<12}"
    )
    print("-" * 100)

    # Sort by overhead magnitude
    category_overheads = []
    for cat in all_categories:
        deepep_size = deepep_stats.get(cat, {}).get("total_bytes", 0)
        baseline_size = baseline_stats.get(cat, {}).get("total_bytes", 0)
        overhead = deepep_size - baseline_size
        overhead_pct = 100 * overhead / total_overhead if total_overhead > 0 else 0
        category_overheads.append(
            (cat, deepep_size, baseline_size, overhead, overhead_pct)
        )

    category_overheads.sort(key=lambda x: abs(x[3]), reverse=True)

    for cat, deepep_size, baseline_size, overhead, overhead_pct in category_overheads:
        overhead_str = f"{'+' if overhead >= 0 else ''}{format_bytes(overhead)}"
        print(
            f"{cat:<25} {format_bytes(deepep_size):<15} {format_bytes(baseline_size):<15} {overhead_str:<15} {overhead_pct:>6.2f}%"
        )


def analyze_deepep_specific_allocations(deepep_stats: Dict):
    """Analyze DeepEP-specific memory allocations in detail."""
    print(f"\n{'='*80}")
    print(f"DEEPEP-SPECIFIC ALLOCATIONS (Detailed)")
    print(f"{'='*80}")

    deepep_cat = deepep_stats.get("DeepEP Overhead", {})

    if not deepep_cat:
        print("\nNo DeepEP-specific allocations found in snapshot.")
        return

    total = deepep_cat["total_bytes"]
    allocs = deepep_cat["allocations"]

    print(f"\nTotal DeepEP Overhead: {format_bytes(total)}")
    print(f"Number of allocations: {len(allocs)}")

    print(f"\nTop 20 DeepEP Allocations:")
    print(f"{'Rank':<6} {'Size':<15} {'Location':<70}")
    print("-" * 100)

    for i, alloc in enumerate(allocs[:20], 1):
        size = format_bytes(alloc["size"])
        location = alloc["top_frame"]
        print(f"{i:<6} {size:<15} {location:<70}")

        # Print stack trace for top 5
        if i <= 5:
            print(f"       Stack trace:")
            for frame in alloc["frames"][:5]:
                print(f"         {frame['filename']}:{frame['line']} {frame['name']}")
            print()


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python deep_analysis_all_overhead.py <deepep_snapshot.pickle> [baseline_snapshot.pickle]"
        )
        sys.exit(1)

    deepep_path = Path(sys.argv[1])
    baseline_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    # Analyze DeepEP snapshot
    print("=" * 80)
    print("ANALYZING DEEPEP SNAPSHOT")
    print("=" * 80)
    deepep_stats, deepep_allocs = analyze_snapshot(deepep_path)
    print_category_summary(deepep_stats, "DeepEP Memory by Category")
    print_top_allocations(deepep_stats, "Top 50 Allocations in DeepEP", top_n=50)
    analyze_deepep_specific_allocations(deepep_stats)

    # Analyze baseline if provided
    if baseline_path and baseline_path.exists():
        print("\n\n")
        print("=" * 80)
        print("ANALYZING BASELINE SNAPSHOT")
        print("=" * 80)
        baseline_stats, baseline_allocs = analyze_snapshot(baseline_path)
        print_category_summary(baseline_stats, "Baseline Memory by Category")
        print_top_allocations(
            baseline_stats, "Top 50 Allocations in Baseline", top_n=50
        )

        # Compare
        print("\n\n")
        compare_snapshots(deepep_stats, baseline_stats)
    else:
        print("\n\nNote: Provide baseline snapshot for comparison analysis.")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
