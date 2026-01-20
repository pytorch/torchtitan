"""
Lightweight NaN/Inf tracker for debugging training issues.

This module provides hooks to track tensor statistics (min, max, mean, nan_count, inf_count)
at each layer without saving tensors, making it suitable for large model debugging.

Usage:
    from torchtitan.utils.nan_tracker import NaNTracker

    tracker = NaNTracker()
    tracker.register_hooks(model)

    # In training loop:
    loss = model(inputs)
    loss.backward()

    # Check for NaN
    if tracker.has_nan():
        tracker.print_nan_report()

    tracker.step()  # Reset for next step
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class TensorStats:
    """Statistics for a single tensor."""

    name: str
    shape: Tuple[int, ...]
    dtype: str
    min_val: float
    max_val: float
    mean_val: float
    std_val: float
    nan_count: int
    inf_count: int
    total_elements: int

    @property
    def has_nan(self) -> bool:
        return self.nan_count > 0

    @property
    def has_inf(self) -> bool:
        return self.inf_count > 0

    def __str__(self) -> str:
        status = ""
        if self.has_nan:
            status += f" [NaN: {self.nan_count}/{self.total_elements}]"
        if self.has_inf:
            status += f" [Inf: {self.inf_count}/{self.total_elements}]"
        return (
            f"{self.name}: shape={self.shape}, dtype={self.dtype}, "
            f"min={self.min_val:.4g}, max={self.max_val:.4g}, "
            f"mean={self.mean_val:.4g}, std={self.std_val:.4g}{status}"
        )


@dataclass
class LayerStats:
    """Statistics for a layer's inputs and outputs."""

    layer_name: str
    layer_type: str
    input_stats: List[TensorStats] = field(default_factory=list)
    output_stats: List[TensorStats] = field(default_factory=list)
    grad_input_stats: List[TensorStats] = field(default_factory=list)
    grad_output_stats: List[TensorStats] = field(default_factory=list)

    @property
    def has_nan(self) -> bool:
        for stats_list in [
            self.input_stats,
            self.output_stats,
            self.grad_input_stats,
            self.grad_output_stats,
        ]:
            for stats in stats_list:
                if stats.has_nan:
                    return True
        return False

    @property
    def has_inf(self) -> bool:
        for stats_list in [
            self.input_stats,
            self.output_stats,
            self.grad_input_stats,
            self.grad_output_stats,
        ]:
            for stats in stats_list:
                if stats.has_inf:
                    return True
        return False


def compute_tensor_stats(tensor: torch.Tensor, name: str) -> Optional[TensorStats]:
    """Compute statistics for a tensor without storing it."""
    if tensor is None:
        return None

    if not isinstance(tensor, torch.Tensor):
        return None

    # Handle DTensor by getting local tensor
    if hasattr(tensor, "_local_tensor"):
        tensor = tensor._local_tensor

    # Flatten for stats computation
    flat = tensor.detach().float().flatten()

    # Count NaN and Inf
    nan_mask = torch.isnan(flat)
    inf_mask = torch.isinf(flat)
    nan_count = nan_mask.sum().item()
    inf_count = inf_mask.sum().item()

    # Compute stats on valid values only
    valid_mask = ~(nan_mask | inf_mask)
    valid_vals = flat[valid_mask]

    if valid_vals.numel() > 0:
        min_val = valid_vals.min().item()
        max_val = valid_vals.max().item()
        mean_val = valid_vals.mean().item()
        std_val = valid_vals.std().item() if valid_vals.numel() > 1 else 0.0
    else:
        min_val = float("nan")
        max_val = float("nan")
        mean_val = float("nan")
        std_val = float("nan")

    return TensorStats(
        name=name,
        shape=tuple(tensor.shape),
        dtype=str(tensor.dtype),
        min_val=min_val,
        max_val=max_val,
        mean_val=mean_val,
        std_val=std_val,
        nan_count=int(nan_count),
        inf_count=int(inf_count),
        total_elements=tensor.numel(),
    )


class NaNTracker:
    """
    Lightweight tracker for NaN/Inf in model activations and gradients.

    Registers forward and backward hooks on model layers to compute statistics
    without storing tensors.

    Args:
        track_forward: Track forward pass activations
        track_backward: Track backward pass gradients
        layer_types: Only track these layer types (default: all)
        include_patterns: Only track layers matching these patterns
        exclude_patterns: Exclude layers matching these patterns
        log_every_layer: Print stats for every layer (verbose)
        rank: Current process rank for distributed training
    """

    def __init__(
        self,
        track_forward: bool = True,
        track_backward: bool = True,
        layer_types: Optional[Tuple[type, ...]] = None,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        log_every_layer: bool = False,
        rank: int = 0,
    ):
        self.track_forward = track_forward
        self.track_backward = track_backward
        self.layer_types = layer_types
        self.include_patterns = include_patterns or []
        self.exclude_patterns = exclude_patterns or []
        self.log_every_layer = log_every_layer
        self.rank = rank

        self.step_num = 0
        self.layer_stats: Dict[str, LayerStats] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._first_nan_layer: Optional[str] = None
        self._first_nan_phase: Optional[str] = None
        self._forward_order: List[str] = []

    def _should_track(self, name: str, module: nn.Module) -> bool:
        """Check if this layer should be tracked."""
        # Check layer type filter
        if self.layer_types is not None:
            if not isinstance(module, self.layer_types):
                return False

        # Check include patterns
        if self.include_patterns:
            if not any(p in name for p in self.include_patterns):
                return False

        # Check exclude patterns
        if self.exclude_patterns:
            if any(p in name for p in self.exclude_patterns):
                return False

        return True

    def _create_forward_hook(self, layer_name: str, layer_type: str):
        """Create a forward hook for a layer."""

        def hook(module, inputs, outputs):
            if layer_name not in self.layer_stats:
                self.layer_stats[layer_name] = LayerStats(
                    layer_name=layer_name,
                    layer_type=layer_type,
                )
                self._forward_order.append(layer_name)

            stats = self.layer_stats[layer_name]

            # Process inputs
            if isinstance(inputs, tuple):
                for i, inp in enumerate(inputs):
                    if isinstance(inp, torch.Tensor):
                        tensor_stats = compute_tensor_stats(inp, f"input_{i}")
                        if tensor_stats:
                            stats.input_stats.append(tensor_stats)
                            if tensor_stats.has_nan and self._first_nan_layer is None:
                                self._first_nan_layer = layer_name
                                self._first_nan_phase = f"forward_input_{i}"

            # Process outputs
            if isinstance(outputs, torch.Tensor):
                tensor_stats = compute_tensor_stats(outputs, "output")
                if tensor_stats:
                    stats.output_stats.append(tensor_stats)
                    if tensor_stats.has_nan and self._first_nan_layer is None:
                        self._first_nan_layer = layer_name
                        self._first_nan_phase = "forward_output"
            elif isinstance(outputs, tuple):
                for i, out in enumerate(outputs):
                    if isinstance(out, torch.Tensor):
                        tensor_stats = compute_tensor_stats(out, f"output_{i}")
                        if tensor_stats:
                            stats.output_stats.append(tensor_stats)
                            if tensor_stats.has_nan and self._first_nan_layer is None:
                                self._first_nan_layer = layer_name
                                self._first_nan_phase = f"forward_output_{i}"

            if self.log_every_layer and self.rank == 0:
                self._print_layer_stats(layer_name, "forward")

        return hook

    def _create_backward_hook(self, layer_name: str, layer_type: str):
        """Create a backward hook for a layer."""

        def hook(module, grad_input, grad_output):
            if layer_name not in self.layer_stats:
                self.layer_stats[layer_name] = LayerStats(
                    layer_name=layer_name,
                    layer_type=layer_type,
                )

            stats = self.layer_stats[layer_name]

            # Process grad_output (gradient w.r.t. layer output)
            if isinstance(grad_output, tuple):
                for i, grad in enumerate(grad_output):
                    if isinstance(grad, torch.Tensor):
                        tensor_stats = compute_tensor_stats(grad, f"grad_output_{i}")
                        if tensor_stats:
                            stats.grad_output_stats.append(tensor_stats)
                            if tensor_stats.has_nan and self._first_nan_layer is None:
                                self._first_nan_layer = layer_name
                                self._first_nan_phase = f"backward_grad_output_{i}"

            # Process grad_input (gradient w.r.t. layer input)
            if isinstance(grad_input, tuple):
                for i, grad in enumerate(grad_input):
                    if isinstance(grad, torch.Tensor):
                        tensor_stats = compute_tensor_stats(grad, f"grad_input_{i}")
                        if tensor_stats:
                            stats.grad_input_stats.append(tensor_stats)
                            if tensor_stats.has_nan and self._first_nan_layer is None:
                                self._first_nan_layer = layer_name
                                self._first_nan_phase = f"backward_grad_input_{i}"

            if self.log_every_layer and self.rank == 0:
                self._print_layer_stats(layer_name, "backward")

        return hook

    def register_hooks(self, model: nn.Module) -> None:
        """Register forward and backward hooks on model layers."""
        for name, module in model.named_modules():
            if not self._should_track(name, module):
                continue

            # Skip container modules
            if isinstance(module, (nn.ModuleList, nn.ModuleDict, nn.Sequential)):
                continue

            layer_type = type(module).__name__

            if self.track_forward:
                hook = module.register_forward_hook(
                    self._create_forward_hook(name, layer_type)
                )
                self.hooks.append(hook)

            if self.track_backward:
                hook = module.register_full_backward_hook(
                    self._create_backward_hook(name, layer_type)
                )
                self.hooks.append(hook)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def step(self) -> None:
        """Reset statistics for next step."""
        self.step_num += 1
        self.layer_stats.clear()
        self._first_nan_layer = None
        self._first_nan_phase = None
        self._forward_order.clear()

    def has_nan(self) -> bool:
        """Check if any NaN was detected this step."""
        return self._first_nan_layer is not None

    def has_any_nan_or_inf(self) -> bool:
        """Check if any NaN or Inf was detected this step."""
        for stats in self.layer_stats.values():
            if stats.has_nan or stats.has_inf:
                return True
        return False

    def get_first_nan_location(self) -> Optional[Tuple[str, str]]:
        """Get the first layer and phase where NaN appeared."""
        if self._first_nan_layer:
            return (self._first_nan_layer, self._first_nan_phase)
        return None

    def _print_layer_stats(self, layer_name: str, phase: str) -> None:
        """Print statistics for a single layer."""
        if layer_name not in self.layer_stats:
            return

        stats = self.layer_stats[layer_name]
        print(f"[Step {self.step_num}][{phase}] {layer_name} ({stats.layer_type}):")

        if phase == "forward":
            for s in stats.input_stats:
                print(f"  IN:  {s}")
            for s in stats.output_stats:
                print(f"  OUT: {s}")
        else:
            for s in stats.grad_output_stats:
                print(f"  GRAD_OUT: {s}")
            for s in stats.grad_input_stats:
                print(f"  GRAD_IN:  {s}")

    def print_nan_report(self) -> None:
        """Print a detailed report of where NaN/Inf occurred."""
        if self.rank != 0:
            return

        print(f"\n{'='*80}")
        print(f"NaN/Inf REPORT - Step {self.step_num}")
        print(f"{'='*80}")

        if self._first_nan_layer:
            print(
                f"\n** FIRST NaN detected at: {self._first_nan_layer} ({self._first_nan_phase}) **\n"
            )

        # Print layers in forward order
        nan_layers = []
        for layer_name in self._forward_order:
            if layer_name in self.layer_stats:
                stats = self.layer_stats[layer_name]
                if stats.has_nan or stats.has_inf:
                    nan_layers.append(layer_name)

        if nan_layers:
            print(f"Layers with NaN/Inf ({len(nan_layers)} total):")
            for layer_name in nan_layers:
                stats = self.layer_stats[layer_name]
                print(f"\n  {layer_name} ({stats.layer_type}):")
                for s in stats.input_stats:
                    if s.has_nan or s.has_inf:
                        print(f"    [FWD IN]  {s}")
                for s in stats.output_stats:
                    if s.has_nan or s.has_inf:
                        print(f"    [FWD OUT] {s}")
                for s in stats.grad_output_stats:
                    if s.has_nan or s.has_inf:
                        print(f"    [BWD GRAD_OUT] {s}")
                for s in stats.grad_input_stats:
                    if s.has_nan or s.has_inf:
                        print(f"    [BWD GRAD_IN]  {s}")
        else:
            print("No NaN/Inf detected in tracked layers.")

        print(f"\n{'='*80}\n")

    def print_summary(self) -> None:
        """Print a summary of all layer statistics."""
        if self.rank != 0:
            return

        print(f"\n{'='*80}")
        print(f"LAYER STATISTICS SUMMARY - Step {self.step_num}")
        print(f"{'='*80}")
        print(f"Total layers tracked: {len(self.layer_stats)}")

        nan_count = sum(1 for s in self.layer_stats.values() if s.has_nan)
        inf_count = sum(1 for s in self.layer_stats.values() if s.has_inf)
        print(f"Layers with NaN: {nan_count}")
        print(f"Layers with Inf: {inf_count}")

        if self._first_nan_layer:
            print(f"\nFirst NaN at: {self._first_nan_layer} ({self._first_nan_phase})")

        print(f"{'='*80}\n")

    def get_stats_dict(self) -> Dict[str, Any]:
        """Get statistics as a dictionary for logging."""
        result = {
            "step": self.step_num,
            "has_nan": self.has_nan(),
            "first_nan_layer": self._first_nan_layer,
            "first_nan_phase": self._first_nan_phase,
            "layers_with_nan": [],
            "layers_with_inf": [],
        }

        for name, stats in self.layer_stats.items():
            if stats.has_nan:
                result["layers_with_nan"].append(name)
            if stats.has_inf:
                result["layers_with_inf"].append(name)

        return result


def create_nan_tracker_for_deepseek(
    model: nn.Module,
    rank: int = 0,
    verbose: bool = False,
) -> NaNTracker:
    """
    Create a NaN tracker optimized for DeepSeek models.

    Tracks key layers that are most likely to produce NaN:
    - Attention layers (FlexAttention, MLA)
    - MoE layers (router, experts)
    - Normalization layers
    - Output projection
    """
    tracker = NaNTracker(
        track_forward=True,
        track_backward=True,
        include_patterns=[
            "attention",
            "moe",
            "router",
            "expert",
            "norm",
            "output",
            "feed_forward",
            "tok_embeddings",
        ],
        exclude_patterns=[
            "freqs_cis",
        ],
        log_every_layer=verbose,
        rank=rank,
    )

    tracker.register_hooks(model)
    return tracker
