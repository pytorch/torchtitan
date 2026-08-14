# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model-declared activation rematerialization regions.

Model code declares candidate regions with ``REMAT_REGIONS`` and wraps their
call sites with :func:`maybe_remat_region`. A remat activation-checkpointing
policy selects candidates by their FQN relative to a transformer block.

The helpers are no-ops unless a policy enables a region. ``torch_remat`` is an
optional dependency and is imported only when an enabled region is used.
"""

from __future__ import annotations

from collections.abc import Callable
from fnmatch import fnmatch
from importlib.metadata import PackageNotFoundError, version
from typing import Any, ParamSpec, TYPE_CHECKING, TypeVar

import torch.nn as nn

if TYPE_CHECKING:
    import torch


_ENABLED_ATTR = "_remat_saved_regions"
_P = ParamSpec("_P")
_R = TypeVar("_R")


def require_torch_remat() -> Any:
    """Import the torch_remat 0.2 API with an actionable error if unavailable."""
    try:
        import torch_remat
    except ImportError as error:
        raise ImportError(
            "Remat activation checkpointing requires torch_remat>=0.2.0. "
            "Until 0.2.0 is published, install it with --no-deps from "
            "https://github.com/meta-pytorch/remat.git@main."
        ) from error

    required_api = ("checkpoint", "recompute_needs_tensor", "region")
    missing_api = [name for name in required_api if not hasattr(torch_remat, name)]
    if missing_api:
        try:
            installed_version = version("torch_remat")
        except PackageNotFoundError:
            installed_version = "unknown"
        raise ImportError(
            "Remat activation checkpointing requires torch_remat>=0.2.0, but "
            f"version {installed_version} is missing: {', '.join(missing_api)}."
        )
    return torch_remat


def _declared_region_names(module: nn.Module) -> tuple[str, ...]:
    module_type = type(module)
    regions = getattr(module_type, "REMAT_REGIONS", ())
    if not regions:
        return ()

    method_resolution_order = module_type.__mro__
    region_owner_index = next(
        index
        for index, base in enumerate(method_resolution_order)
        if "REMAT_REGIONS" in base.__dict__
    )
    forward_owner_index = next(
        index
        for index, base in enumerate(method_resolution_order)
        if "forward" in base.__dict__
    )
    if forward_owner_index < region_owner_index:
        return ()
    return tuple(regions)


def maybe_remat_region(
    fn: Callable[_P, _R], name: str, *, owner: nn.Module
) -> Callable[_P, _R]:
    """Return ``fn`` as a retained remat region when selected, else unchanged."""
    enabled_regions = getattr(owner, _ENABLED_ATTR, None)
    if not enabled_regions:
        return fn
    assert name in _declared_region_names(
        owner
    ), f"{type(owner).__name__} used undeclared remat region {name!r}"
    region_fqn = enabled_regions.get(name)
    if region_fqn is None:
        return fn
    return require_torch_remat().region(fn, region_fqn, recompute=False)


def maybe_recompute_needs(owner: nn.Module, *tensors: torch.Tensor) -> None:
    """Persist selected-region outputs that a bare operation will consume."""
    if not getattr(owner, _ENABLED_ATTR, None):
        return
    require_torch_remat().recompute_needs_tensor(*tensors)


def declared_regions(root: nn.Module) -> list[str]:
    """Return region FQNs offered by ``root`` and its descendants."""
    names = []
    for module_fqn, module in root.named_modules():
        for region_name in _declared_region_names(module):
            names.append(f"{module_fqn}.{region_name}" if module_fqn else region_name)
    return names


def apply_region_selection(
    root: nn.Module, save_patterns: list[str]
) -> tuple[list[str], list[str]]:
    """Enable declared regions matching FQN glob patterns under ``root``."""
    selected = []
    available = []
    for module_fqn, module in root.named_modules():
        regions = _declared_region_names(module)
        if not regions:
            continue
        enabled_regions = {}
        for region_name in regions:
            region_fqn = f"{module_fqn}.{region_name}" if module_fqn else region_name
            available.append(region_fqn)
            if any(fnmatch(region_fqn, pattern) for pattern in save_patterns):
                enabled_regions[region_name] = region_fqn
                selected.append(region_fqn)
        setattr(module, _ENABLED_ATTR, enabled_regions)
    return selected, available
