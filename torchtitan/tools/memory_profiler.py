"""
Detailed memory profiler for distributed training.
Instruments key allocation points to track exactly where memory goes.
"""

import json
from collections import defaultdict
from typing import Dict

import torch
from torchtitan.tools.logging import logger


class DetailedMemoryProfiler:
    """Track memory allocations at key points in training."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.checkpoints = {}
        self.allocations = defaultdict(list)
        self.device = torch.cuda.current_device()

    def checkpoint(self, name: str):
        """Record memory state at a checkpoint."""
        if not self.enabled:
            return

        stats = torch.cuda.memory_stats(self.device)

        self.checkpoints[name] = {
            "active": stats.get("active_bytes.all.current", 0),
            "allocated": stats.get("allocated_bytes.all.current", 0),
            "reserved": stats.get("reserved_bytes.all.current", 0),
            "peak_active": stats.get("active_bytes.all.peak", 0),
            "num_allocs": stats.get("num_alloc.all.current", 0),
        }

    def compute_delta(self, name: str, prev_checkpoint: str) -> Dict:
        """Compute memory delta between two checkpoints."""
        if (
            not self.enabled
            or name not in self.checkpoints
            or prev_checkpoint not in self.checkpoints
        ):
            return {}

        current = self.checkpoints[name]
        previous = self.checkpoints[prev_checkpoint]

        return {
            "active_delta": current["active"] - previous["active"],
            "allocated_delta": current["allocated"] - previous["allocated"],
            "reserved_delta": current["reserved"] - previous["reserved"],
        }

    def log_checkpoint(self, name: str, prev_checkpoint: str = None):
        """Log checkpoint with optional delta."""
        if not self.enabled or name not in self.checkpoints:
            return

        stats = self.checkpoints[name]
        active_gb = stats["active"] / (1024**3)
        reserved_gb = stats["reserved"] / (1024**3)

        msg = f"[MemProfile] {name}: Active={active_gb:.2f} GB, Reserved={reserved_gb:.2f} GB"

        if prev_checkpoint:
            delta = self.compute_delta(name, prev_checkpoint)
            if delta:
                delta_gb = delta["active_delta"] / (1024**3)
                msg += f", Delta={delta_gb:+.2f} GB"

        logger.info(msg)

    def get_breakdown(self) -> Dict:
        """Compute memory breakdown from checkpoints."""
        if not self.enabled or not self.checkpoints:
            return {}

        # Define key memory components based on checkpoints
        breakdown = {}

        checkpoint_names = list(self.checkpoints.keys())
        for i in range(len(checkpoint_names) - 1):
            curr_name = checkpoint_names[i + 1]
            prev_name = checkpoint_names[i]

            delta = self.compute_delta(curr_name, prev_name)
            if delta and delta["active_delta"] > 0:
                breakdown[f"{prev_name}_to_{curr_name}"] = delta["active_delta"]

        return breakdown

    def save_report(self, filepath: str):
        """Save detailed memory report to JSON."""
        if not self.enabled:
            return

        report = {
            "checkpoints": self.checkpoints,
            "breakdown": self.get_breakdown(),
        }

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Memory report saved to {filepath}")

    def print_summary(self):
        """Print summary table of memory usage."""
        if not self.enabled or not self.checkpoints:
            return

        logger.info("=" * 80)
        logger.info("DETAILED MEMORY PROFILE SUMMARY")
        logger.info("=" * 80)

        # Print checkpoints
        logger.info(f"\n{'Checkpoint':<40} {'Active (GB)':>15} {'Reserved (GB)':>15}")
        logger.info("-" * 72)

        for name, stats in self.checkpoints.items():
            active_gb = stats["active"] / (1024**3)
            reserved_gb = stats["reserved"] / (1024**3)
            logger.info(f"{name:<40} {active_gb:>14.2f} {reserved_gb:>14.2f}")

        # Print breakdown
        breakdown = self.get_breakdown()
        if breakdown:
            logger.info(f"\n{'Component':<40} {'Memory (GB)':>15} {'Percentage':>12}")
            logger.info("-" * 72)

            total = sum(breakdown.values())
            for name, size in sorted(
                breakdown.items(), key=lambda x: x[1], reverse=True
            ):
                size_gb = size / (1024**3)
                pct = (size / total * 100) if total > 0 else 0
                logger.info(f"{name:<40} {size_gb:>14.2f} {pct:>11.1f}%")

        logger.info("=" * 80)


# Global profiler instance
_profiler = None


def get_memory_profiler() -> DetailedMemoryProfiler:
    """Get or create global memory profiler."""
    global _profiler
    if _profiler is None:
        _profiler = DetailedMemoryProfiler(enabled=True)
    return _profiler


def checkpoint(name: str):
    """Record memory checkpoint."""
    get_memory_profiler().checkpoint(name)


def log_checkpoint(name: str, prev_checkpoint: str = None):
    """Log memory checkpoint."""
    get_memory_profiler().log_checkpoint(name, prev_checkpoint)


def print_summary():
    """Print memory profile summary."""
    get_memory_profiler().print_summary()


def save_report(filepath: str):
    """Save memory report."""
    get_memory_profiler().save_report(filepath)
