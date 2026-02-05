# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding debug utilities for capturing and visualizing DTensor sharding info."""

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

import torch
import torch.nn as nn
from torch import distributed as dist
from torch.distributed.tensor import DTensor
from torch.utils._pytree import tree_flatten
from torch.utils.hooks import RemovableHandle

from torchtitan.tools.logging import logger

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

__all__ = ["ShardingDebugContext", "ShardingEntry"]

# Width for ASCII diagram separators
_LINE_WIDTH = 80


@dataclass
class ShardingEntry:
    """Structured data for a single sharding info entry.

    Attributes:
        call_order: Sequential order in which this entry was recorded.
        phase: One of "forward", "backward", or "recompute".
        module_name: Fully qualified name of the module (e.g., "model.layers.0.attn").
        tensor_type: Type of tensor (e.g., "input0", "weight", "output0", "grad_input0").
        placements: Tuple of placement strings (e.g., ("Shard(0)",) or ("Replicate",)).
        mesh: Formatted device mesh string, or None for local (non-distributed) tensors.
        shape: Tensor shape as a tuple of integers.
    """

    call_order: int
    phase: str
    module_name: str
    tensor_type: str
    placements: tuple[str, ...]
    mesh: str | None
    shape: tuple[int, ...]


class ShardingDebugContext:
    """Context manager that logs DTensor sharding info for all module operations.

    Registers forward and backward hooks on all modules to capture sharding
    information for inputs, outputs, parameters, and gradients during
    one forward/backward pass.
    """

    def __init__(
        self,
        model_parts: list[nn.Module],
        dump_folder: str,
        collapse_layers: bool = True,
    ):
        self.model_parts = model_parts
        self.collapse_layers = collapse_layers
        self.hooks: list[RemovableHandle] = []

        # Once root module's forward completes, subsequent forward hook calls are
        # from AC.
        self.initial_forward_complete: dict[int, bool] = {}

        self.num_modules_traced = 0
        self.forward_entries = 0
        self.recompute_entries = 0
        self.backward_entries = 0
        self.placements_seen: set[str] = set()
        self.call_counter = 0
        self.entries: list[ShardingEntry] = []
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.output_dir = os.path.join(dump_folder, "sharding_info")

    def __enter__(self) -> "ShardingDebugContext":
        os.makedirs(self.output_dir, exist_ok=True)
        self._register_hooks()
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        self._remove_hooks()

        ascii_content = self._generate_ascii_diagram()
        txt_path = os.path.join(self.output_dir, f"rank{self.rank}_sharding_info.txt")
        with open(txt_path, "w") as f:
            f.write(ascii_content)

        html_content = self._generate_html()
        html_path = os.path.join(self.output_dir, f"rank{self.rank}_sharding_info.html")
        with open(html_path, "w") as f:
            f.write(html_content)

        logger.info(f"Sharding info written to {self.output_dir}")

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on all modules."""
        if not self.model_parts:
            return

        for idx, model in enumerate(self.model_parts):
            self.initial_forward_complete[idx] = False

            for fqn, module in model.named_modules():
                self.num_modules_traced += 1
                module._sharding_debug_fqn = fqn  # type: ignore[attr-defined]
                module._sharding_debug_idx = idx  # type: ignore[attr-defined]
                part_prefix = f"part{idx}." if len(self.model_parts) > 1 else ""

                forward_hook = self._create_forward_hook(fqn, idx, part_prefix)
                handle = module.register_forward_hook(forward_hook)
                self.hooks.append(handle)

                backward_hook = self._create_backward_hook(fqn, part_prefix)
                handle = module.register_full_backward_hook(backward_hook)
                self.hooks.append(handle)

    def _remove_hooks(self) -> None:
        """Remove all registered hooks and clean up module attributes."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()

        for model in self.model_parts:
            for module in model.modules():
                for attr in ("_sharding_debug_fqn", "_sharding_debug_idx"):
                    if hasattr(module, attr):
                        delattr(module, attr)

    @staticmethod
    def _clean_fqn(fqn: str) -> str:
        """Remove _checkpoint_wrapped_module from FQN."""
        parts = fqn.split(".")
        cleaned = [p for p in parts if p != "_checkpoint_wrapped_module"]
        return ".".join(cleaned)

    def _create_forward_hook(self, fqn: str, idx: int, part_prefix: str) -> Callable:
        """Create a forward hook that captures input/output sharding info."""
        clean_fqn = self._clean_fqn(fqn)
        is_root = fqn == ""

        def hook(module, inputs, outputs):
            is_recompute = self.initial_forward_complete.get(idx, False)
            phase = "recompute" if is_recompute else "forward"
            module_name = part_prefix + ("model." + clean_fqn if clean_fqn else "model")

            flat_inputs, _ = tree_flatten(inputs)
            for i, tensor in enumerate(flat_inputs):
                if isinstance(tensor, torch.Tensor):
                    self._add_entry(phase, module_name, f"input{i}", tensor)

            for param_name, param in module.named_parameters(recurse=False):
                if isinstance(param, torch.Tensor):
                    self._add_entry(phase, module_name, param_name, param)

            flat_outputs, _ = tree_flatten(outputs)
            for i, tensor in enumerate(flat_outputs):
                if isinstance(tensor, torch.Tensor):
                    self._add_entry(phase, module_name, f"output{i}", tensor)

            if is_root and not self.initial_forward_complete.get(idx, False):
                self.initial_forward_complete[idx] = True

        return hook

    def _create_backward_hook(self, fqn: str, part_prefix: str) -> Callable:
        """Create a backward hook that captures gradient sharding info."""
        clean_fqn = self._clean_fqn(fqn)

        def hook(module, grad_input, grad_output):
            module_name = part_prefix + ("model." + clean_fqn if clean_fqn else "model")

            flat_grad_output, _ = tree_flatten(grad_output)
            for i, tensor in enumerate(flat_grad_output):
                if isinstance(tensor, torch.Tensor):
                    self._add_entry("backward", module_name, f"grad_output{i}", tensor)

            for param_name, param in module.named_parameters(recurse=False):
                if param.grad is not None and isinstance(param.grad, torch.Tensor):
                    self._add_entry(
                        "backward", module_name, f"{param_name}.grad", param.grad
                    )

            flat_grad_input, _ = tree_flatten(grad_input)
            for i, tensor in enumerate(flat_grad_input):
                if isinstance(tensor, torch.Tensor):
                    self._add_entry("backward", module_name, f"grad_input{i}", tensor)

        return hook

    def _add_entry(
        self,
        phase: str,
        module_name: str,
        tensor_type: str,
        tensor: torch.Tensor,
    ) -> None:
        """Record a sharding entry for a tensor."""
        self.call_counter += 1

        if isinstance(tensor, DTensor):
            placements = tuple(str(p) for p in tensor.placements)
            mesh = self._format_mesh(tensor.device_mesh)
        else:
            placements = ("Local",)
            mesh = None

        entry = ShardingEntry(
            call_order=self.call_counter,
            phase=phase,
            module_name=module_name,
            tensor_type=tensor_type,
            placements=placements,
            mesh=mesh,
            shape=tuple(tensor.shape),
        )
        self.entries.append(entry)

        if phase == "forward":
            self.forward_entries += 1
        elif phase == "recompute":
            self.recompute_entries += 1
        elif phase == "backward":
            self.backward_entries += 1

        for p in placements:
            self.placements_seen.add(p)

    # Below methods are output generation related methods (ASCII and HTML)
    # All are generated by Claude and can skip if you just want
    # to know how to track the sharding.

    @staticmethod
    def _extract_layer_pattern(module_name: str) -> tuple[str, int | None]:
        """Extract layer pattern and index from module name.

        Returns:
            A tuple of (pattern_or_name, index). If the module name matches a layer
            pattern (e.g., 'model.layers.5.attn'), returns the pattern with {idx}
            placeholder and the layer index: ('model.layers.{idx}.attn', 5).
            If no layer pattern is found, returns the original module name and None:
            ('model.norm', None).
        """
        match = re.match(
            r"^((?:model\.)?)?(layers?|blocks?)\.(\d+)(?:\.(.*))?$", module_name
        )
        if not match:
            return module_name, None
        model_prefix, layer_prefix, idx, suffix = match.groups()
        model_prefix = model_prefix or ""
        base = f"{model_prefix}{layer_prefix}.{{idx}}"
        return (f"{base}.{suffix}", int(idx)) if suffix else (base, int(idx))

    def _get_module_signature(
        self,
        module_name: str,
        entries_by_module: dict[str, list[ShardingEntry]],
    ) -> tuple[tuple[str, str, str | None, tuple[int, ...]], ...]:
        """Get signature for comparing modules: (tensor_type, placement, mesh, shape).

        Deduplicates by tensor_type to handle activation checkpointing where
        hooks may fire multiple times for the same tensor.

        Args:
            module_name: The module name to get the signature for.
            entries_by_module: Pre-built index mapping module names to their entries
                for efficient lookup (avoids O(n²) iteration over all entries).
        """
        seen: dict[str, tuple[str, str, str | None, tuple[int, ...]]] = {}
        for e in entries_by_module.get(module_name, []):
            if e.tensor_type not in seen:
                seen[e.tensor_type] = (
                    e.tensor_type,
                    self._format_placement(e.placements),
                    e.mesh,
                    e.shape,
                )
        return tuple(sorted(seen.values(), key=lambda x: x[0]))

    @staticmethod
    def _finalize_group(
        pattern: str, group: list[tuple[int, str]]
    ) -> tuple[str, list[str]]:
        """Convert a group of (idx, module_name) to (display_name, module_names)."""
        if len(group) == 1:
            return group[0][1], [group[0][1]]
        indices = [i for i, _ in group]
        display = pattern.replace("{idx}", f"[{min(indices)}-{max(indices)}]")
        return display, [n for _, n in group]

    def _collapse_repeated_modules(
        self, module_names: list[str], phase: str
    ) -> list[tuple[str, list[str], bool]]:
        """Collapse layers with identical signatures into ranges like [0-31]."""
        if not self.collapse_layers:
            return [(name, [name], False) for name in module_names]

        # Build index of entries by module name for efficient signature lookup (O(n) vs O(n²))
        entries_by_module: dict[str, list[ShardingEntry]] = defaultdict(list)
        for e in self.entries:
            if e.phase == phase:
                entries_by_module[e.module_name].append(e)

        # Group modules by pattern
        pattern_to_modules: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for module_name in module_names:
            pattern, idx = self._extract_layer_pattern(module_name)
            if idx is not None:
                pattern_to_modules[pattern].append((idx, module_name))

        # Collapse consecutive modules with identical signatures
        pattern_to_collapsed: dict[str, list[tuple[str, list[str]]]] = {}
        for pattern, idx_modules in pattern_to_modules.items():
            idx_modules.sort(key=lambda x: x[0])
            groups: list[tuple[str, list[str]]] = []
            current_group: list[tuple[int, str]] = []
            current_sig: tuple | None = None

            for idx, module_name in idx_modules:
                sig = self._get_module_signature(module_name, entries_by_module)
                if current_sig is None or sig == current_sig:
                    current_group.append((idx, module_name))
                    current_sig = sig
                else:
                    groups.append(self._finalize_group(pattern, current_group))
                    current_group = [(idx, module_name)]
                    current_sig = sig

            if current_group:
                groups.append(self._finalize_group(pattern, current_group))
            pattern_to_collapsed[pattern] = groups

        # Build result preserving original order
        result: list[tuple[str, list[str], bool]] = []
        seen_patterns: set[str] = set()
        output_modules: set[str] = set()

        for module_name in module_names:
            if module_name in output_modules:
                continue
            pattern, idx = self._extract_layer_pattern(module_name)
            if idx is None:
                result.append((module_name, [module_name], False))
                output_modules.add(module_name)
                continue
            if pattern not in pattern_to_collapsed:
                continue
            for display_name, group_modules in pattern_to_collapsed[pattern]:
                if module_name not in group_modules:
                    continue
                group_key = f"{pattern}:{display_name}"
                if group_key not in seen_patterns:
                    seen_patterns.add(group_key)
                    result.append((display_name, group_modules, len(group_modules) > 1))
                    output_modules.update(group_modules)
                break

        return result

    @staticmethod
    def _format_mesh(device_mesh: "DeviceMesh | None") -> str | None:
        """Format device mesh as '(dp=2, tp=4)'.

        Args:
            device_mesh: The device mesh to format.

        Returns:
            Formatted string like '(tp=8)' or '(dp=2, tp=4)', or None if no mesh.
        """
        if device_mesh is None:
            return None
        try:
            names = device_mesh.mesh_dim_names
            shape = device_mesh.shape
            if not names:
                return f"(shape={shape})"
            return "(" + ", ".join(f"{n}={s}" for n, s in zip(names, shape)) + ")"
        except AttributeError:
            # Fall back if mesh doesn't have expected attributes
            return str(device_mesh)

    @staticmethod
    def _format_placement(placements: tuple[str, ...]) -> str:
        """Format placements tuple as a single string."""
        return placements[0] if len(placements) == 1 else ", ".join(placements)

    @staticmethod
    def _format_shape(shape: tuple[int, ...]) -> str:
        """Format shape tuple as a string like '(1, 8192, 4096)'."""
        return f"({', '.join(str(s) for s in shape)})"

    @staticmethod
    def _tensor_sort_key(t: str) -> tuple[int, str]:
        """Sort tensors: inputs/grad_outputs first, params middle, outputs last."""
        if t.startswith(("input", "grad_output")):
            return (0, t)
        if t.startswith(("output", "grad_input")):
            return (2, t)
        return (1, t)

    def _compute_global_max_lengths(self) -> tuple[int, int, int, int]:
        """Compute max lengths for tensor_type, placement, mesh, shape globally."""
        max_type = max_place = max_mesh = max_shape = 0
        for e in self.entries:
            max_type = max(max_type, len(e.tensor_type))
            max_place = max(max_place, len(self._format_placement(e.placements)))
            mesh_str = "None" if e.mesh is None else e.mesh
            max_mesh = max(max_mesh, len(mesh_str))
            max_shape = max(max_shape, len(self._format_shape(e.shape)))
        return max_type, max_place, max_mesh, max_shape

    def _render_module_tensors(
        self,
        lines: list[str],
        tensors: dict[str, list[ShardingEntry]],
        max_lens: tuple[int, int, int, int],
        indent: str = "  ",
    ) -> None:
        """Render tensor entries as ASCII tree lines."""
        sorted_tensors = sorted(tensors.keys(), key=self._tensor_sort_key)
        if not sorted_tensors:
            return

        max_type_len, max_place_len, max_mesh_len, max_shape_len = max_lens

        for idx, tensor_type in enumerate(sorted_tensors):
            entry = tensors[tensor_type][0]
            placement = self._format_placement(entry.placements)
            mesh = "None" if entry.mesh is None else entry.mesh
            shape = self._format_shape(entry.shape)

            branch = "└─" if idx == len(sorted_tensors) - 1 else "├─"
            line = (
                f"{indent}{branch} {tensor_type:<{max_type_len}}: "
                f"{placement:<{max_place_len}}  mesh={mesh:<{max_mesh_len}}  "
                f"shape={shape}"
            )
            lines.append(line)

    def _build_module_tree(
        self, phase: str
    ) -> dict[str, dict[str, list[ShardingEntry]]]:
        """Build dict: module_name -> tensor_type -> entries."""
        tree: dict[str, dict[str, list[ShardingEntry]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for entry in self.entries:
            if entry.phase == phase:
                tree[entry.module_name][entry.tensor_type].append(entry)
        return tree

    def _generate_ascii_diagram(self) -> str:
        """Generate ASCII text representation of sharding info."""
        lines: list[str] = []
        max_lens = self._compute_global_max_lengths()

        lines.append(f"Sharding Info - Rank {self.rank}")
        lines.append("=" * _LINE_WIDTH)
        lines.append("")

        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Total modules traced: {self.num_modules_traced}")
        lines.append(f"Forward entries:      {self.forward_entries}")
        lines.append(f"Recompute entries:    {self.recompute_entries}")
        lines.append(f"Backward entries:     {self.backward_entries}")
        lines.append(f"Unique placements:    {', '.join(sorted(self.placements_seen))}")
        lines.append("")

        lines.append("LEGEND")
        lines.append("-" * 40)
        lines.append("  R         = Replicated across all devices")
        lines.append("  S(n)      = Sharded along dimension n")
        lines.append("  P(op)     = Partial with reduction operation (e.g., P(sum))")
        lines.append("  Local     = Local tensor (not distributed)")
        lines.append(
            "  [recompute] = Module recomputed during backward (activation checkpointing)"
        )
        lines.append("  [0-N]     = Layers 0 through N with identical sharding pattern")
        lines.append("")

        forward_entries = [e for e in self.entries if e.phase == "forward"]
        if forward_entries:
            lines.append("=" * _LINE_WIDTH)
            lines.append("FORWARD PASS")
            lines.append("=" * _LINE_WIDTH)
            lines.append("")

            module_tree = self._build_module_tree("forward")

            def get_first_call_order(module_name: str) -> int:
                return min(
                    (
                        e.call_order
                        for t in module_tree[module_name].values()
                        for e in t
                    ),
                    default=0,
                )

            sorted_modules = sorted(module_tree.keys(), key=get_first_call_order)
            collapsed = self._collapse_repeated_modules(sorted_modules, "forward")

            for display_name, original_names, _ in collapsed:
                lines.append(display_name)
                self._render_module_tensors(
                    lines, module_tree[original_names[0]], max_lens
                )
                lines.append("")

        backward_entries = [
            e for e in self.entries if e.phase in ("backward", "recompute")
        ]
        if backward_entries:
            lines.append("=" * _LINE_WIDTH)
            lines.append("BACKWARD PASS")
            lines.append("=" * _LINE_WIDTH)
            lines.append("")

            trees = {
                "recompute": self._build_module_tree("recompute"),
                "backward": self._build_module_tree("backward"),
            }

            # Build (module_name, phase) list preserving interleaved execution order
            module_phase_order: list[tuple[str, str]] = []
            seen: set[tuple[str, str]] = set()
            for entry in sorted(backward_entries, key=lambda e: e.call_order):
                key = (entry.module_name, entry.phase)
                if key not in seen:
                    seen.add(key)
                    module_phase_order.append(key)

            # Build collapsed maps for each phase
            def build_collapsed_map(
                phase: str,
            ) -> dict[str, tuple[str, list[str], bool]]:
                names = [m for m, p in module_phase_order if p == phase]
                collapsed = self._collapse_repeated_modules(names, phase)
                result: dict[str, tuple[str, list[str], bool]] = {}
                for display, orig_names, is_col in collapsed:
                    for name in orig_names:
                        result[name] = (display, orig_names, is_col)
                return result

            collapsed_maps = {
                "recompute": build_collapsed_map("recompute"),
                "backward": build_collapsed_map("backward"),
            }
            output_groups: dict[str, set[str]] = {"recompute": set(), "backward": set()}

            for module_name, phase in module_phase_order:
                if module_name not in collapsed_maps[phase]:
                    continue
                display_name, original_names, _ = collapsed_maps[phase][module_name]
                if display_name in output_groups[phase]:
                    continue
                output_groups[phase].add(display_name)
                prefix = "[recompute] " if phase == "recompute" else ""
                lines.append(f"{prefix}{display_name}")
                self._render_module_tensors(
                    lines, trees[phase][original_names[0]], max_lens
                )
                lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _load_html_template() -> str:
        """Load the HTML template from the package directory."""
        import pathlib

        template_path = pathlib.Path(__file__).parent / "sharding_info_template.html"
        return template_path.read_text()

    def _get_module_tensors(
        self, module_name: str, phase: str
    ) -> list[dict[str, str | None]]:
        """Get deduplicated, sorted tensor info for a module.

        Args:
            module_name: The module name to get tensors for.
            phase: The phase ("forward", "backward", or "recompute").

        Returns:
            List of tensor info dicts with keys: tensorType, placement, mesh, shape.
        """
        seen: dict[str, dict[str, str | None]] = {}
        for e in self.entries:
            if e.module_name == module_name and e.phase == phase:
                if e.tensor_type not in seen:
                    seen[e.tensor_type] = {
                        "tensorType": e.tensor_type,
                        "placement": self._format_placement(e.placements),
                        "mesh": e.mesh,
                        "shape": self._format_shape(e.shape),
                    }
        # Sort: inputs/grad_outputs first, params middle, outputs/grad_inputs last
        return sorted(
            seen.values(), key=lambda t: self._tensor_sort_key(str(t["tensorType"]))
        )

    def _build_html_module_views(
        self,
    ) -> dict[str, dict[str, list[dict]]]:
        """Pre-compute collapsed and uncollapsed module views for HTML.

        Returns a dict with structure:
        {
            "forward": {"collapsed": [...], "uncollapsed": [...]},
            "backward": {"collapsed": [...], "uncollapsed": [...]}
        }

        Each module entry contains:
        - displayName: The name to show
        - originalNames: List of original module names (for search)
        - isRecompute: Whether this is a recompute module
        - tensors: List of tensor info dicts
        """
        result: dict[str, dict[str, list[dict]]] = {
            "forward": {"collapsed": [], "uncollapsed": []},
            "backward": {"collapsed": [], "uncollapsed": []},
        }

        # Build module trees: phase -> module_name -> tensor_type -> entries
        trees: dict[str, dict[str, dict[str, list[ShardingEntry]]]] = {
            "forward": self._build_module_tree("forward"),
            "recompute": self._build_module_tree("recompute"),
            "backward": self._build_module_tree("backward"),
        }

        # Helper to get first call order for sorting
        def get_first_call_order(
            module_name: str, tree: dict[str, dict[str, list[ShardingEntry]]]
        ) -> int:
            return min(
                (e.call_order for t in tree[module_name].values() for e in t),
                default=0,
            )

        # Process forward pass
        forward_tree = trees["forward"]
        if forward_tree:
            sorted_modules = sorted(
                forward_tree.keys(),
                key=lambda m: get_first_call_order(m, forward_tree),
            )

            # Uncollapsed view
            for module_name in sorted_modules:
                result["forward"]["uncollapsed"].append(
                    {
                        "displayName": module_name,
                        "originalNames": [module_name],
                        "isRecompute": False,
                        "tensors": self._get_module_tensors(module_name, "forward"),
                    }
                )

            # Collapsed view
            collapsed = self._collapse_repeated_modules(sorted_modules, "forward")
            for display_name, original_names, _ in collapsed:
                result["forward"]["collapsed"].append(
                    {
                        "displayName": display_name,
                        "originalNames": original_names,
                        "isRecompute": False,
                        "tensors": self._get_module_tensors(
                            original_names[0], "forward"
                        ),
                    }
                )

        # Process backward pass (interleaved recompute + backward)
        backward_entries = [
            e for e in self.entries if e.phase in ("backward", "recompute")
        ]
        if backward_entries:
            # Build (module_name, phase) list preserving execution order
            module_phase_order: list[tuple[str, str]] = []
            seen: set[tuple[str, str]] = set()
            for entry in sorted(backward_entries, key=lambda e: e.call_order):
                key = (entry.module_name, entry.phase)
                if key not in seen:
                    seen.add(key)
                    module_phase_order.append(key)

            # Uncollapsed view - preserve execution order
            for module_name, phase in module_phase_order:
                result["backward"]["uncollapsed"].append(
                    {
                        "displayName": module_name,
                        "originalNames": [module_name],
                        "isRecompute": phase == "recompute",
                        "tensors": self._get_module_tensors(module_name, phase),
                    }
                )

            # Collapsed view - collapse each phase separately, maintain order
            recompute_names = [m for m, p in module_phase_order if p == "recompute"]
            backward_names = [m for m, p in module_phase_order if p == "backward"]

            collapsed_recompute = self._collapse_repeated_modules(
                recompute_names, "recompute"
            )
            collapsed_backward = self._collapse_repeated_modules(
                backward_names, "backward"
            )

            # Build lookup maps
            recompute_map: dict[str, tuple[str, list[str]]] = {}
            for display, orig_names, _ in collapsed_recompute:
                for name in orig_names:
                    recompute_map[name] = (display, orig_names)

            backward_map: dict[str, tuple[str, list[str]]] = {}
            for display, orig_names, _ in collapsed_backward:
                for name in orig_names:
                    backward_map[name] = (display, orig_names)

            # Output in execution order, deduplicating groups
            output_groups: dict[str, set[str]] = {"recompute": set(), "backward": set()}
            for module_name, phase in module_phase_order:
                lookup = recompute_map if phase == "recompute" else backward_map
                if module_name not in lookup:
                    continue
                display_name, orig_names = lookup[module_name]
                if display_name in output_groups[phase]:
                    continue
                output_groups[phase].add(display_name)
                result["backward"]["collapsed"].append(
                    {
                        "displayName": display_name,
                        "originalNames": orig_names,
                        "isRecompute": phase == "recompute",
                        "tensors": self._get_module_tensors(orig_names[0], phase),
                    }
                )

        return result

    def _generate_html(self) -> str:
        """Generate interactive HTML visualization of sharding info."""
        from string import Template

        # Pre-compute both collapsed and uncollapsed views
        # This moves all collapsing logic to Python, simplifying the JavaScript
        module_views = self._build_html_module_views()
        module_views_json = json.dumps(module_views, indent=None)

        template = Template(self._load_html_template())
        return template.substitute(
            RANK=self.rank,
            NUM_MODULES_TRACED=self.num_modules_traced,
            FORWARD_ENTRIES=self.forward_entries,
            RECOMPUTE_ENTRIES=self.recompute_entries,
            BACKWARD_ENTRIES=self.backward_entries,
            MODULE_VIEWS_JSON=module_views_json,
            COLLAPSE_LAYERS_CHECKED="checked" if self.collapse_layers else "",
        )
