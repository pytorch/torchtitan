#!/usr/bin/env python3
"""
Comprehensive Memory Profiling Analyzer
Analyzes PyTorch memory snapshots and generates visualizations + detailed reports.

Auto-generates timestamped output directories for each run.
"""

import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def format_size(bytes_size):
    """Format bytes to human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.2f} TB"


def format_size_gb(bytes_size):
    """Format bytes to GB."""
    return bytes_size / (1024**3)


def get_tensor_category(frames):
    """Categorize tensor by its allocation location."""
    if not frames:
        return "Unknown"

    for frame in frames:
        filename = frame.get("filename", "").lower()
        name = frame.get("name", "").lower()

        if "optimizer" in filename or "optimizer" in name:
            return "Optimizer State"
        elif "expert" in filename or "moe" in filename:
            return "Expert/MoE"
        elif "attention" in filename or "attn" in filename:
            return "Attention"
        elif "linear" in filename or "matmul" in filename:
            return "Linear/MatMul"
        elif "embedding" in filename or "emb" in filename:
            return "Embedding"
        elif "norm" in filename:
            return "Normalization"
        elif "activation" in filename:
            return "Activation"
        elif "gradient" in filename or "backward" in filename:
            return "Gradient"
        elif "all_to_all" in filename or "communication" in filename:
            return "Communication"
        elif "fsdp" in filename:
            return "FSDP"

    return "Other"


def load_and_analyze_snapshot(snapshot_path):
    """Load and analyze memory snapshot."""
    with open(snapshot_path, "rb") as f:
        snapshot = pickle.load(f)

    segments = snapshot.get("segments", [])

    total_allocated = 0
    total_reserved = 0
    active_blocks = []
    inactive_blocks = []

    for segment in segments:
        blocks = segment.get("blocks", [])
        for block in blocks:
            size = block["size"]
            state = block["state"]

            if state == "active_allocated":
                total_allocated += size
                active_blocks.append(block)
            elif state == "inactive":
                total_reserved += size
                inactive_blocks.append(block)
            else:
                total_reserved += size

    total_reserved += total_allocated

    # Sort by size
    active_blocks.sort(key=lambda b: b["size"], reverse=True)
    inactive_blocks.sort(key=lambda b: b["size"], reverse=True)

    # Categorize active blocks
    category_totals = defaultdict(int)
    for block in active_blocks:
        category = get_tensor_category(block.get("frames", []))
        category_totals[category] += block["size"]

    return {
        "total_allocated": total_allocated,
        "total_reserved": total_reserved,
        "active_blocks": active_blocks,
        "inactive_blocks": inactive_blocks,
        "category_totals": category_totals,
    }


def generate_visualizations(analysis, output_dir):
    """Generate all visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use("seaborn-v0_8-darkgrid")
    colors = plt.cm.Set3(np.linspace(0, 1, 12))

    # 1. Memory Overview (Pie Chart)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Overall allocation vs cache
    total_allocated_gb = format_size_gb(analysis["total_allocated"])
    cache_gb = format_size_gb(analysis["total_reserved"] - analysis["total_allocated"])

    ax1.pie(
        [total_allocated_gb, cache_gb],
        labels=["Allocated", "Cache/Fragmentation"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#2ecc71", "#e74c3c"],
    )
    ax1.set_title(
        f'Memory Usage Overview\nTotal: {format_size(analysis["total_reserved"])}',
        fontsize=14,
        fontweight="bold",
    )

    # Category breakdown
    categories = list(analysis["category_totals"].keys())
    sizes_gb = [format_size_gb(analysis["category_totals"][cat]) for cat in categories]

    ax2.pie(
        sizes_gb, labels=categories, autopct="%1.1f%%", startangle=90, colors=colors
    )
    ax2.set_title("Active Memory by Category", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "memory_overview.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Top Memory Consumers (Bar Chart)
    fig, ax = plt.subplots(figsize=(14, 8))

    top_n = 30
    top_blocks = analysis["active_blocks"][:top_n]
    sizes_mb = [block["size"] / (1024**2) for block in top_blocks]
    categories = [get_tensor_category(block.get("frames", [])) for block in top_blocks]

    # Color by category
    category_colors = {
        cat: colors[i % len(colors)] for i, cat in enumerate(set(categories))
    }
    bar_colors = [category_colors[cat] for cat in categories]

    bars = ax.barh(range(len(sizes_mb)), sizes_mb, color=bar_colors)
    ax.set_yticks(range(len(sizes_mb)))
    ax.set_yticklabels([f"#{i+1}" for i in range(len(sizes_mb))], fontsize=8)
    ax.set_xlabel("Size (MB)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Allocation Rank", fontsize=12, fontweight="bold")
    ax.set_title(f"Top {top_n} Memory Allocations", fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=category_colors[cat], label=cat)
        for cat in sorted(set(categories))
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "top_consumers.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Category Breakdown (Bar Chart)
    fig, ax = plt.subplots(figsize=(12, 7))

    categories = sorted(
        analysis["category_totals"].keys(),
        key=lambda k: analysis["category_totals"][k],
        reverse=True,
    )
    sizes_gb = [format_size_gb(analysis["category_totals"][cat]) for cat in categories]

    bars = ax.bar(categories, sizes_gb, color=colors[: len(categories)])
    ax.set_ylabel("Memory Usage (GB)", fontsize=12, fontweight="bold")
    ax.set_title("Memory Usage by Category", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, size in zip(bars, sizes_gb):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{size:.1f} GB",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "category_breakdown.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Allocation Size Distribution (Histogram)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Active blocks distribution
    active_sizes_mb = [
        block["size"] / (1024**2) for block in analysis["active_blocks"]
    ]
    ax1.hist(active_sizes_mb, bins=50, color="#3498db", edgecolor="black", alpha=0.7)
    ax1.set_xlabel("Allocation Size (MB)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax1.set_title("Active Allocation Size Distribution", fontsize=14, fontweight="bold")
    ax1.set_yscale("log")
    ax1.grid(axis="y", alpha=0.3)

    # Inactive blocks (cache) distribution
    if analysis["inactive_blocks"]:
        inactive_sizes_mb = [
            block["size"] / (1024**2) for block in analysis["inactive_blocks"]
        ]
        ax2.hist(
            inactive_sizes_mb, bins=30, color="#e74c3c", edgecolor="black", alpha=0.7
        )
        ax2.set_xlabel("Block Size (MB)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Count", fontsize=12, fontweight="bold")
        ax2.set_title(
            "Cached/Fragmented Block Size Distribution", fontsize=14, fontweight="bold"
        )
        ax2.grid(axis="y", alpha=0.3)
    else:
        ax2.text(
            0.5,
            0.5,
            "No cached blocks\n(Excellent!)",
            ha="center",
            va="center",
            fontsize=16,
            transform=ax2.transAxes,
        )
        ax2.set_title("Cached/Fragmented Blocks", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "size_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 5. Memory Efficiency Summary
    fig, ax = plt.subplots(figsize=(10, 6))

    allocated_gb = format_size_gb(analysis["total_allocated"])
    cache_gb = format_size_gb(analysis["total_reserved"] - analysis["total_allocated"])
    total_gb = format_size_gb(analysis["total_reserved"])
    efficiency = (allocated_gb / total_gb) * 100

    metrics = ["Total\nReserved", "Actually\nAllocated", "Cache/\nWasted"]
    values = [total_gb, allocated_gb, cache_gb]
    bar_colors_list = ["#95a5a6", "#2ecc71", "#e74c3c"]

    bars = ax.bar(
        metrics, values, color=bar_colors_list, edgecolor="black", linewidth=2
    )
    ax.set_ylabel("Memory (GB)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Memory Efficiency: {efficiency:.1f}%", fontsize=14, fontweight="bold"
    )

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.1f} GB",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # Add efficiency indicator
    ax.axhline(y=allocated_gb, color="green", linestyle="--", linewidth=2, alpha=0.5)
    ax.text(
        2.5,
        allocated_gb + 2,
        f"Efficiency: {efficiency:.1f}%",
        fontsize=11,
        fontweight="bold",
        color="green",
    )

    ax.set_ylim(0, max(values) * 1.2)
    plt.tight_layout()
    plt.savefig(output_dir / "memory_efficiency.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n✅ Generated 5 visualization files in: {output_dir}/")
    print(f"   1. memory_overview.png       - Overall memory usage breakdown")
    print(f"   2. top_consumers.png         - Top 30 memory allocations")
    print(f"   3. category_breakdown.png    - Memory by category")
    print(f"   4. size_distribution.png     - Allocation size histograms")
    print(f"   5. memory_efficiency.png     - Efficiency metrics")


def generate_text_report(analysis, output_file):
    """Generate detailed text report."""
    output_file = Path(output_file)

    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("MEMORY PROFILING ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Summary
        allocated = analysis["total_allocated"]
        reserved = analysis["total_reserved"]
        cache = reserved - allocated
        efficiency = (allocated / reserved) * 100

        f.write(f"SUMMARY\n")
        f.write(f"-" * 80 + "\n")
        f.write(f"Total Reserved:      {format_size(reserved):>15s}  (100.0%)\n")
        f.write(
            f"Total Allocated:     {format_size(allocated):>15s}  ({efficiency:.1f}%)\n"
        )
        f.write(
            f"Cache/Wasted:        {format_size(cache):>15s}  ({cache/reserved*100:.1f}%)\n"
        )
        f.write(f"Memory Efficiency:   {efficiency:.1f}%\n")
        f.write(f"\n")

        # Category breakdown
        f.write(f"MEMORY BY CATEGORY\n")
        f.write(f"-" * 80 + "\n")
        for cat, size in sorted(
            analysis["category_totals"].items(), key=lambda x: x[1], reverse=True
        ):
            pct = size / allocated * 100
            f.write(f"  {cat:30s}: {format_size(size):>15s}  ({pct:.1f}%)\n")
        f.write(f"\n")

        # Top allocations
        f.write(f"TOP 50 ALLOCATIONS\n")
        f.write(f"-" * 80 + "\n")
        f.write(f"{'Rank':<6} {'Size':<15} {'Category':<25}\n")
        f.write(f"-" * 80 + "\n")

        for i, block in enumerate(analysis["active_blocks"][:50], 1):
            size = block["size"]
            category = get_tensor_category(block.get("frames", []))
            f.write(f"{i:<6} {format_size(size):<15} {category:<25}\n")

        f.write(f"\n")

        # Fragmentation analysis
        f.write(f"FRAGMENTATION ANALYSIS\n")
        f.write(f"-" * 80 + "\n")

        num_active = len(analysis["active_blocks"])
        num_inactive = len(analysis["inactive_blocks"])
        inactive_total = sum(b["size"] for b in analysis["inactive_blocks"])

        f.write(f"Active Blocks:       {num_active:,}\n")
        f.write(f"Inactive Blocks:     {num_inactive:,}\n")
        f.write(f"Fragmentation:       {cache/reserved*100:.2f}%\n")

        if num_inactive > 0:
            large_inactive = [
                b for b in analysis["inactive_blocks"] if b["size"] > 100 * 1024 * 1024
            ]
            f.write(f"Large Unused (>100MB): {len(large_inactive)}\n")
            f.write(
                f"  Total wasted: {format_size(sum(b['size'] for b in large_inactive))}\n"
            )

        f.write(f"\n")

        # Recommendations
        f.write(f"OPTIMIZATION RECOMMENDATIONS\n")
        f.write(f"-" * 80 + "\n")

        if efficiency >= 98:
            f.write(f"✅ Excellent memory efficiency ({efficiency:.1f}%)\n")
            f.write(f"   No significant optimizations needed.\n")
        elif efficiency >= 95:
            f.write(f"✓ Good memory efficiency ({efficiency:.1f}%)\n")
            f.write(f"   Minor optimizations possible.\n")
        else:
            f.write(f"⚠️  Memory efficiency could be improved ({efficiency:.1f}%)\n")
            f.write(f"   - Cache overhead: {format_size(cache)}\n")
            f.write(f"   - Consider more aggressive allocator settings\n")

        if cache / reserved > 0.1:
            f.write(f"\n⚠️  High cache overhead (>10%)\n")
            f.write(f"   - Try: PYTORCH_ALLOC_CONF with lower max_split_size_mb\n")

        f.write(f"\n" + "=" * 80 + "\n")

    print(f"✅ Generated detailed report: {output_file}")


def create_output_directory(base_dir, custom_name=None):
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if custom_name:
        # Sanitize custom name
        custom_name = custom_name.replace(" ", "_").replace("/", "_")
        dir_name = f"{timestamp}_{custom_name}"
    else:
        dir_name = timestamp

    output_dir = Path(base_dir) / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create/update "latest" symlink
    latest_link = Path(base_dir) / "latest"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    try:
        latest_link.symlink_to(dir_name, target_is_directory=True)
    except OSError:
        # Windows or permission issues - skip symlink
        pass

    return output_dir


def extract_run_name_from_snapshot(snapshot_path):
    """Extract a meaningful run name from snapshot path."""
    path = Path(snapshot_path)

    # Try to extract batch size or other identifiers from path
    parts = path.parts
    for part in reversed(parts):
        if "lbs" in part.lower():
            return part
        if "batch" in part.lower():
            return part
        if "iteration" in part.lower():
            return part

    # Default to iteration number and rank
    if "iteration" in str(path):
        iteration = [p for p in parts if "iteration" in p]
        if iteration:
            iter_num = iteration[0].split("_")[-1]
            rank = path.stem.split("_")[0] if "rank" in path.stem else "rank0"
            return f"iter{iter_num}_{rank}"

    return "analysis"


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze.py <snapshot.pickle> [output_base_dir] [run_name]")
        print()
        print("Arguments:")
        print("  snapshot.pickle    - Path to memory snapshot file")
        print(
            "  output_base_dir    - Base directory for results (default: scripts/memory_analysis/results)"
        )
        print(
            "  run_name          - Custom name for this run (optional, auto-detected from path)"
        )
        print()
        print("Examples:")
        print("  # Auto-generate timestamped directory")
        print(
            "  python analyze.py outputs/memory_snapshot/iteration_1/rank0_memory_snapshot.pickle"
        )
        print()
        print("  # Specify base directory")
        print("  python analyze.py snapshot.pickle scripts/memory_analysis/results")
        print()
        print("  # Custom run name")
        print(
            "  python analyze.py snapshot.pickle scripts/memory_analysis/results lbs8_selective_ac"
        )
        sys.exit(1)

    snapshot_path = Path(sys.argv[1])

    # Determine base directory
    if len(sys.argv) > 2:
        base_dir = Path(sys.argv[2])
    else:
        # Default to scripts/memory_analysis/results
        base_dir = Path(__file__).parent / "results"

    # Determine run name
    if len(sys.argv) > 3:
        run_name = sys.argv[3]
    else:
        run_name = extract_run_name_from_snapshot(snapshot_path)

    if not snapshot_path.exists():
        print(f"Error: Snapshot file not found: {snapshot_path}")
        sys.exit(1)

    # Create timestamped output directory
    output_dir = create_output_directory(base_dir, run_name)

    print(f"Loading snapshot: {snapshot_path}")
    analysis = load_and_analyze_snapshot(snapshot_path)

    print(f"\nAnalysis complete!")
    print(f"  Total Reserved:  {format_size(analysis['total_reserved'])}")
    print(f"  Total Allocated: {format_size(analysis['total_allocated'])}")
    print(
        f"  Cache/Wasted:    {format_size(analysis['total_reserved'] - analysis['total_allocated'])}"
    )
    print(
        f"  Efficiency:      {(analysis['total_allocated'] / analysis['total_reserved'] * 100):.1f}%"
    )

    # Generate outputs
    print(f"\nGenerating visualizations...")
    generate_visualizations(analysis, output_dir)

    print(f"\nGenerating text report...")
    generate_text_report(analysis, output_dir / "analysis_report.txt")

    print(f"\n{'='*80}")
    print(f"✅ Analysis complete! All outputs saved to: {output_dir}/")

    # Show relative path from current directory
    try:
        rel_path = output_dir.relative_to(Path.cwd())
        print(f"   Relative path: ./{rel_path}")
    except ValueError:
        pass

    # Show "latest" link location
    latest_link = base_dir / "latest"
    if latest_link.exists():
        print(f"   Latest results: {base_dir}/latest -> {output_dir.name}")

    print(f"{'='*80}")


if __name__ == "__main__":
    main()
