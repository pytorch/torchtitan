"""Detailed memory tracking throughout training step"""
import logging
from typing import Dict, List

import torch

logger = logging.getLogger(__name__)


class DetailedMemoryTracker:
    """Track memory at every phase of training with cache clearing"""

    def __init__(self, enabled: bool = True, clear_cache: bool = True):
        self.enabled = enabled
        self.clear_cache_between_steps = clear_cache
        self.measurements: List[Dict] = []
        self.device = torch.cuda.current_device()

        if self.enabled:
            logger.info(f"DetailedMemoryTracker enabled (clear_cache={clear_cache})")

    def measure(self, phase: str, step: int):
        """Capture memory state at a specific phase"""
        if not self.enabled:
            return

        stats = torch.cuda.memory_stats(self.device)

        measurement = {
            "step": step,
            "phase": phase,
            "reserved": torch.cuda.memory_reserved(self.device),
            "allocated": torch.cuda.memory_allocated(self.device),
            "active": stats.get("active_bytes.all.current", 0),
            "peak_active": stats.get("active_bytes.all.peak", 0),
            "num_allocs": stats.get("num_alloc_retries.all.current", 0),
        }

        self.measurements.append(measurement)

        # Calculate fragmentation
        fragmentation = measurement["reserved"] - measurement["allocated"]
        frag_pct = (
            (fragmentation / measurement["reserved"] * 100)
            if measurement["reserved"] > 0
            else 0
        )

        logger.info(
            f"[MemTrack] Step {step} | {phase:20s} | "
            f"Reserved: {measurement['reserved']/1e9:6.2f} GB | "
            f"Allocated: {measurement['allocated']/1e6:7.2f} MB | "
            f"Active: {measurement['active']/1e6:7.2f} MB | "
            f"Frag: {frag_pct:5.1f}%"
        )

    def clear_cache_and_measure(self, phase: str, step: int):
        """Clear cache and measure to see minimum memory"""
        if not self.enabled:
            return

        # Measure before clearing
        self.measure(f"{phase}_before_clear", step)

        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Measure after clearing
        self.measure(f"{phase}_after_clear", step)

    def step_complete(self, step: int):
        """Called after each training step"""
        if not self.enabled:
            return

        if self.clear_cache_between_steps:
            self.clear_cache_and_measure("step_end", step)

    def get_summary(self) -> str:
        """Get summary of all measurements"""
        if not self.measurements:
            return "No measurements recorded"

        summary = ["", "=" * 100, "DETAILED MEMORY TRACKING SUMMARY", "=" * 100, ""]

        # Group by step
        steps = {}
        for m in self.measurements:
            step = m["step"]
            if step not in steps:
                steps[step] = []
            steps[step].append(m)

        for step, measures in sorted(steps.items()):
            summary.append(f"\nStep {step}:")
            summary.append(
                f"{'Phase':<30} {'Reserved':>12} {'Allocated':>12} {'Active':>12} {'Frag%':>8}"
            )
            summary.append("-" * 80)

            for m in measures:
                frag_pct = (
                    ((m["reserved"] - m["allocated"]) / m["reserved"] * 100)
                    if m["reserved"] > 0
                    else 0
                )
                summary.append(
                    f"{m['phase']:<30} "
                    f"{m['reserved']/1e9:10.2f} GB "
                    f"{m['allocated']/1e6:10.2f} MB "
                    f"{m['active']/1e6:10.2f} MB "
                    f"{frag_pct:7.1f}%"
                )

        # Peak measurements
        summary.append("\n" + "=" * 100)
        summary.append("PEAK MEASUREMENTS ACROSS ALL STEPS:")
        summary.append("=" * 100)

        peak_reserved = max(m["reserved"] for m in self.measurements)
        peak_allocated = max(m["allocated"] for m in self.measurements)
        peak_active = max(m["active"] for m in self.measurements)

        peak_reserved_phase = [
            m for m in self.measurements if m["reserved"] == peak_reserved
        ][0]
        peak_allocated_phase = [
            m for m in self.measurements if m["allocated"] == peak_allocated
        ][0]
        peak_active_phase = [
            m for m in self.measurements if m["active"] == peak_active
        ][0]

        summary.append(
            f"Peak Reserved:   {peak_reserved/1e9:7.2f} GB at Step {peak_reserved_phase['step']} ({peak_reserved_phase['phase']})"
        )
        step = peak_allocated_phase["step"]
        phase = peak_allocated_phase["phase"]
        summary.append(
            f"Peak Allocated:  {peak_allocated/1e6:7.2f} MB at Step {step} ({phase})"
        )
        summary.append(
            f"Peak Active:     {peak_active/1e6:7.2f} MB at Step {peak_active_phase['step']} ({peak_active_phase['phase']})"
        )

        # Minimum after cache clear
        cleared_measures = [m for m in self.measurements if "after_clear" in m["phase"]]
        if cleared_measures:
            min_reserved_cleared = min(m["reserved"] for m in cleared_measures)
            min_measure = [
                m for m in cleared_measures if m["reserved"] == min_reserved_cleared
            ][0]
            summary.append(
                f"\nMinimum Reserved (after cache clear): {min_reserved_cleared/1e9:7.2f} GB at Step {min_measure['step']}"
            )
            summary.append(f"  Active at minimum: {min_measure['active']/1e6:7.2f} MB")

        summary.append("=" * 100)
        return "\n".join(summary)
