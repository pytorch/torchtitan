# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Resolve model-declared activation rematerialization save regions."""

from __future__ import annotations

import re

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Any

import torch.nn as nn

from torchtitan.protocols.module import Module


def _class_defining_attribute(module_type: type[Any], attribute_name: str) -> type[Any]:
    """Return the nearest class that defines ``attribute_name`` itself."""
    return next(base for base in module_type.__mro__ if attribute_name in base.__dict__)


def _available_remat_save_region_names(module: nn.Module) -> tuple[str, ...]:
    module_type = type(module)
    # Read from the instance so configurations can expose only active call sites.
    region_names = getattr(module, "AVAILABLE_REMAT_SAVE_REGIONS", ())
    if not region_names:
        return ()

    region_owner = _class_defining_attribute(
        module_type, "AVAILABLE_REMAT_SAVE_REGIONS"
    )
    forward_owner = _class_defining_attribute(module_type, "forward")

    # Region names describe call sites in a particular forward implementation.
    # Do not inherit them through a subclass that replaces that implementation.
    if forward_owner is not region_owner and issubclass(forward_owner, region_owner):
        return ()
    return tuple(region_names)


def available_remat_save_regions(root: nn.Module) -> list[str]:
    """Return remat save-region names offered by ``root`` and descendants."""
    names = []
    for module_fqn, module in root.named_modules():
        for region_name in _available_remat_save_region_names(module):
            names.append(f"{module_fqn}.{region_name}" if module_fqn else region_name)
    return names


@dataclass(frozen=True, slots=True)
class ResolvedRematRegion:
    """One model-declared region resolved for a checkpoint block."""

    module_fqn: str
    module_type: str
    local_name: str
    qualified_name: str
    matched_patterns: tuple[str, ...]
    module: Module = field(repr=False, compare=False)

    @property
    def is_saved(self) -> bool:
        return bool(self.matched_patterns)


@dataclass(frozen=True, slots=True)
class ResolvedRematBlockPolicy:
    """Effective remat policy for one checkpointed transformer block."""

    checkpoint_fqn: str
    checkpoint_type: str
    regions: tuple[ResolvedRematRegion, ...]

    def signature(self) -> tuple[Any, ...]:
        """Return the structure and decisions used for report grouping."""
        return (
            self.checkpoint_type,
            tuple(
                (
                    region.module_fqn,
                    region.module_type,
                    region.local_name,
                    region.is_saved,
                )
                for region in self.regions
            ),
        )


@dataclass(frozen=True, slots=True)
class ResolvedRematPolicy:
    """Resolved save-region selectors and their per-block effective policy."""

    save_patterns: tuple[str, ...]
    blocks: tuple[ResolvedRematBlockPolicy, ...]

    @property
    def regions(self) -> tuple[ResolvedRematRegion, ...]:
        return tuple(region for block in self.blocks for region in block.regions)

    @property
    def available_save_regions(self) -> list[str]:
        return list(dict.fromkeys(region.qualified_name for region in self.regions))

    @property
    def selected_save_regions(self) -> list[str]:
        return list(
            dict.fromkeys(
                region.qualified_name for region in self.regions if region.is_saved
            )
        )

    def matches_by_pattern(self) -> dict[str, list[str]]:
        matches = {pattern: [] for pattern in self.save_patterns}
        for region in self.regions:
            for pattern in region.matched_patterns:
                matches[pattern].append(region.qualified_name)
        return {
            pattern: list(dict.fromkeys(pattern_matches))
            for pattern, pattern_matches in matches.items()
        }

    def validate(self) -> None:
        """Raise for missing model regions or selectors with no global match."""
        if not self.available_save_regions:
            raise ValueError(
                "RematAC requires model-provided AVAILABLE_REMAT_SAVE_REGIONS, "
                "but this model does not provide any."
            )

        unmatched_patterns = [
            pattern
            for pattern, matches in self.matches_by_pattern().items()
            if not matches
        ]
        if unmatched_patterns:
            raise ValueError(
                "RematAC save_regions patterns did not match any checkpointed "
                f"model region: {unmatched_patterns}. Available save regions: "
                f"{self.available_save_regions}."
            )

    def install(self) -> None:
        """Install qualified names and save decisions on participating modules."""
        module_policies: dict[int, tuple[Module, dict[str, str], dict[str, bool]]] = {}
        for region in self.regions:
            module_key = id(region.module)
            if module_key not in module_policies:
                module_policies[module_key] = (region.module, {}, {})
            _, region_names, save_decisions = module_policies[module_key]
            region_names[region.local_name] = region.qualified_name
            save_decisions[region.local_name] = region.is_saved

        for module, region_names, save_decisions in module_policies.values():
            Module.configure_remat_regions(module, region_names, save_decisions)

    def format(self) -> str:
        """Render selector expansion and policies grouped by transformer layer."""
        lines = ["RematAC save-region selector expansion:"]
        matches_by_pattern = self.matches_by_pattern()
        if not matches_by_pattern:
            lines.append("  none")
        for pattern, matches in matches_by_pattern.items():
            lines.append(f"  {pattern!r} -> {', '.join(matches)}")

        lines.extend(("", "Effective RematAC transformer-layer policies:"))
        grouped_blocks: dict[
            tuple[Any, ...], list[ResolvedRematBlockPolicy]
        ] = defaultdict(list)
        for block in self.blocks:
            grouped_blocks[block.signature()].append(block)

        for blocks in grouped_blocks.values():
            checkpoint_fqns = _compact_checkpoint_fqns(
                [block.checkpoint_fqn for block in blocks]
            )
            lines.append(f"  {checkpoint_fqns} [{blocks[0].checkpoint_type}]")
            for region in blocks[0].regions:
                action = "SAVE" if region.is_saved else "RECOMPUTE"
                lines.append(
                    f"    {region.qualified_name} [{region.module_type}] -> {action}"
                )
        return "\n".join(lines)


def find_remat_checkpoint_blocks(root: nn.Module) -> list[tuple[str, nn.Module]]:
    """Find every child of a ``layers`` container in ``root`` and descendants."""
    blocks = []
    for root_fqn, module in root.named_modules():
        layers = module._modules.get("layers")
        if layers is None:
            continue
        layers_fqn = f"{root_fqn}.layers" if root_fqn else "layers"
        blocks.extend(
            (f"{layers_fqn}.{layer_id}", block)
            for layer_id, block in layers.named_children()
        )
    return blocks


def resolve_remat_save_policy(
    checkpoint_blocks: Iterable[tuple[str, nn.Module]], save_patterns: list[str]
) -> ResolvedRematPolicy:
    """Resolve selectors against every available region in checkpoint blocks."""
    block_policies = []
    for checkpoint_fqn, checkpoint_block in checkpoint_blocks:
        regions = []
        for module_fqn, module in checkpoint_block.named_modules():
            region_names = _available_remat_save_region_names(module)
            if region_names:
                assert isinstance(module, Module), (
                    "AVAILABLE_REMAT_SAVE_REGIONS must be declared on a "
                    "torchtitan.protocols.module.Module"
                )
            for local_name in region_names:
                qualified_name = (
                    f"{module_fqn}.{local_name}" if module_fqn else local_name
                )
                regions.append(
                    ResolvedRematRegion(
                        module_fqn=module_fqn,
                        module_type=type(module).__name__,
                        local_name=local_name,
                        qualified_name=qualified_name,
                        matched_patterns=tuple(
                            pattern
                            for pattern in save_patterns
                            if fnmatch(qualified_name, pattern)
                        ),
                        module=module,
                    )
                )
        block_policies.append(
            ResolvedRematBlockPolicy(
                checkpoint_fqn=checkpoint_fqn,
                checkpoint_type=type(checkpoint_block).__name__,
                regions=tuple(regions),
            )
        )
    return ResolvedRematPolicy(tuple(save_patterns), tuple(block_policies))


def configure_remat_save_regions(
    root: nn.Module, save_patterns: list[str]
) -> tuple[list[str], list[str]]:
    """Install region names and save decisions on one checkpoint block."""
    policy = resolve_remat_save_policy((("", root),), save_patterns)
    policy.install()
    return policy.selected_save_regions, policy.available_save_regions


_NUMERIC_FQN = re.compile(r"^(.*?)(\d+)$")


def _compact_checkpoint_fqns(checkpoint_fqns: list[str]) -> str:
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    ungrouped: list[str] = []
    for checkpoint_fqn in checkpoint_fqns:
        match = _NUMERIC_FQN.match(checkpoint_fqn)
        if match is None:
            ungrouped.append(checkpoint_fqn)
        else:
            grouped_indices[match.group(1)].append(int(match.group(2)))

    compacted = list(ungrouped)
    for prefix, indices in grouped_indices.items():
        indices.sort()
        ranges = []
        range_start = range_end = indices[0]
        for index in indices[1:]:
            if index == range_end + 1:
                range_end = index
                continue
            ranges.append((range_start, range_end))
            range_start = range_end = index
        ranges.append((range_start, range_end))
        for range_start, range_end in ranges:
            compacted.append(
                f"{prefix}{range_start}"
                if range_start == range_end
                else f"{prefix}{range_start}-{range_end}"
            )
    return f"[{', '.join(compacted)}]"
