#!/usr/bin/env python3
"""
Deep Memory Profiler for Kimi K2 1T Model
Tracks memory allocation at each layer/operation to identify where OOM occurs.
"""

import json
import sys
from collections import defaultdict
from typing import Dict, List

import torch


class DeepMemoryProfiler:
    def __init__(self, output_file: str = "memory_profile.json"):
        self.output_file = output_file
        self.memory_events: List[Dict] = []
        self.hooks = []
        self.current_step = 0
        self.current_phase = "init"

    def _get_memory_stats(self) -> Dict:
        """Get current GPU memory statistics."""
        if not torch.cuda.is_available():
            return {}

        stats = torch.cuda.memory_stats()
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "active_gb": stats.get("active_bytes.all.current", 0) / 1e9,
            "inactive_gb": stats.get("inactive_split_bytes.all.current", 0) / 1e9,
            "num_alloc_retries": stats.get("num_alloc_retries", 0),
            "num_ooms": stats.get("num_ooms", 0),
        }

    def log_memory(self, event_name: str, extra_info: Dict = None):
        """Log memory at a specific event."""
        mem_stats = self._get_memory_stats()
        event = {
            "step": self.current_step,
            "phase": self.current_phase,
            "event": event_name,
            "memory": mem_stats,
        }
        if extra_info:
            event["extra"] = extra_info
        self.memory_events.append(event)

        # Print for real-time monitoring
        print(
            f"[MemProf] Step {self.current_step} | {self.current_phase} | {event_name} | "
            f"Alloc: {mem_stats.get('allocated_gb', 0):.2f} GB | "
            f"Reserved: {mem_stats.get('reserved_gb', 0):.2f} GB"
        )

    def _make_forward_hook(self, layer_name: str):
        """Create a forward hook for a layer."""

        def hook(module, input, output):
            input_shapes = []
            for inp in input:
                if isinstance(inp, torch.Tensor):
                    input_shapes.append(list(inp.shape))

            output_shapes = []
            if isinstance(output, torch.Tensor):
                output_shapes.append(list(output.shape))
            elif isinstance(output, (tuple, list)):
                for out in output:
                    if isinstance(out, torch.Tensor):
                        output_shapes.append(list(out.shape))

            self.log_memory(
                f"forward:{layer_name}",
                {
                    "input_shapes": input_shapes,
                    "output_shapes": output_shapes,
                },
            )

        return hook

    def _make_backward_hook(self, layer_name: str):
        """Create a backward hook for a layer."""

        def hook(module, grad_input, grad_output):
            self.log_memory(f"backward:{layer_name}")

        return hook

    def attach_hooks(self, model: torch.nn.Module, layers_to_track: List[str] = None):
        """Attach memory tracking hooks to model layers."""
        if layers_to_track is None:
            # Default: track key layers in DeepSeek/MoE model
            layers_to_track = [
                "embed_tokens",
                "layers.0",  # First transformer layer
                "layers.30",  # Middle layer
                "layers.60",  # Last layer (if exists)
                "moe",  # MoE layers
                "experts",  # Expert modules
                "norm",
                "lm_head",
            ]

        for name, module in model.named_modules():
            should_track = any(track_name in name for track_name in layers_to_track)
            if should_track:
                # Forward hook
                handle = module.register_forward_hook(self._make_forward_hook(name))
                self.hooks.append(handle)
                # Backward hook
                handle = module.register_full_backward_hook(
                    self._make_backward_hook(name)
                )
                self.hooks.append(handle)
                print(f"[MemProf] Attached hooks to: {name}")

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def set_step(self, step: int):
        self.current_step = step

    def set_phase(self, phase: str):
        self.current_phase = phase

    def save_profile(self):
        """Save memory profile to JSON file."""
        with open(self.output_file, "w") as f:
            json.dump(self.memory_events, f, indent=2)
        print(f"[MemProf] Saved profile to {self.output_file}")

    def print_summary(self):
        """Print memory profile summary."""
        print("\n" + "=" * 80)
        print("MEMORY PROFILE SUMMARY")
        print("=" * 80)

        # Group by event name and find max memory
        event_max_mem = defaultdict(float)
        event_counts = defaultdict(int)

        for event in self.memory_events:
            name = event["event"]
            mem = event["memory"].get("allocated_gb", 0)
            event_max_mem[name] = max(event_max_mem[name], mem)
            event_counts[name] += 1

        # Sort by max memory
        sorted_events = sorted(event_max_mem.items(), key=lambda x: x[1], reverse=True)

        print(f"\n{'Event':<60} {'Max Alloc (GB)':<15} {'Count':<10}")
        print("-" * 85)
        for event_name, max_mem in sorted_events[:30]:
            print(f"{event_name:<60} {max_mem:<15.2f} {event_counts[event_name]:<10}")

        # Find peak memory point
        if self.memory_events:
            peak_event = max(
                self.memory_events, key=lambda x: x["memory"].get("reserved_gb", 0)
            )
            print(f"\n{'='*80}")
            print(
                f"PEAK MEMORY: {peak_event['memory'].get('reserved_gb', 0):.2f} GB reserved"
            )
            print(
                f"  At: Step {peak_event['step']} | Phase: {peak_event['phase']} | Event: {peak_event['event']}"
            )
            if "extra" in peak_event:
                print(f"  Extra: {peak_event['extra']}")


def analyze_memory_difference(profile_2k: str, profile_4k: str):
    """Compare memory profiles between 2k and 4k to find differences."""
    with open(profile_2k) as f:
        events_2k = json.load(f)
    with open(profile_4k) as f:
        events_4k = json.load(f)

    print("\n" + "=" * 80)
    print("MEMORY COMPARISON: 2k vs 4k context")
    print("=" * 80)

    # Build event maps
    def build_event_map(events):
        event_map = {}
        for e in events:
            key = (e["step"], e["phase"], e["event"])
            event_map[key] = e["memory"]
        return event_map

    map_2k = build_event_map(events_2k)
    map_4k = build_event_map(events_4k)

    # Find common events and compare
    common_keys = set(map_2k.keys()) & set(map_4k.keys())

    differences = []
    for key in common_keys:
        mem_2k = map_2k[key].get("allocated_gb", 0)
        mem_4k = map_4k[key].get("allocated_gb", 0)
        diff = mem_4k - mem_2k
        if abs(diff) > 0.1:  # Only show significant differences
            differences.append((key, mem_2k, mem_4k, diff))

    # Sort by difference
    differences.sort(key=lambda x: x[3], reverse=True)

    print(f"\n{'Event':<50} {'2k (GB)':<10} {'4k (GB)':<10} {'Diff (GB)':<10}")
    print("-" * 80)
    for key, mem_2k, mem_4k, diff in differences[:20]:
        step, phase, event = key
        event_short = event[:45] if len(event) > 45 else event
        print(f"{event_short:<50} {mem_2k:<10.2f} {mem_4k:<10.2f} {diff:<+10.2f}")

    # Summary
    total_2k = (
        max(e["memory"].get("reserved_gb", 0) for e in events_2k) if events_2k else 0
    )
    total_4k = (
        max(e["memory"].get("reserved_gb", 0) for e in events_4k) if events_4k else 0
    )

    print(f"\n{'='*80}")
    print("Peak Reserved Memory:")
    print(f"  2k context: {total_2k:.2f} GB")
    print(f"  4k context: {total_4k:.2f} GB")
    print(f"  Difference: {total_4k - total_2k:+.2f} GB")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        # Compare mode
        analyze_memory_difference(sys.argv[1], sys.argv[2])
    else:
        print(
            "Usage: python deep_memory_profiler.py <profile_2k.json> <profile_4k.json>"
        )
