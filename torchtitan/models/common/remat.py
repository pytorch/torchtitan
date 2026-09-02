# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model-declared activation rematerialization regions.

Model code exposes candidate save regions with ``AVAILABLE_REMAT_SAVE_REGIONS``
and wraps their call sites with :func:`maybe_remat_save_region`. A remat
activation-checkpointing policy selects candidates by their qualified name
relative to a transformer block.

The helpers are no-ops unless a policy enables a region. ``torch_remat`` is an
optional dependency and is imported only when an enabled region is used.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fnmatch import fnmatch
from importlib.metadata import PackageNotFoundError, version
from typing import Any, ParamSpec, TYPE_CHECKING, TypeVar

import torch.nn as nn

if TYPE_CHECKING:
    import torch


_REMAT_SAVE_REGION_SELECTION_ATTR = "_remat_save_region_selection"
_P = ParamSpec("_P")
_R = TypeVar("_R")


@dataclass(frozen=True, slots=True)
class RematSaveRegionSelection:
    """Effective remat save-region selection installed on one module.

    For example, a feed-forward module selected to save ``w1`` stores
    ``{"w1": "feed_forward.w1"}``, mapping the local call-site name to the
    name qualified relative to its transformer block.
    """

    qualified_name_by_local_name: dict[str, str]
    """Map local region names to names qualified relative to the block."""


def require_torch_remat() -> Any:
    """Import the torch_remat 0.2 API with an actionable error if unavailable."""
    try:
        import torch_remat
    except ImportError as error:
        raise ImportError(
            "Remat activation checkpointing requires torch_remat>=0.2.0. "
            "Install the optional dependency with "
            "`pip install 'torch_remat>=0.2.0'`."
        ) from error

    required_api = (
        "checkpoint",
        "is_recomputing",
        "recompute_needs_tensor",
        "region",
    )
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


def _class_defining_attribute(module_type: type[Any], attribute_name: str) -> type[Any]:
    """Return the nearest class that defines ``attribute_name`` itself."""
    return next(base for base in module_type.__mro__ if attribute_name in base.__dict__)


def _available_remat_save_region_names(module: nn.Module) -> tuple[str, ...]:
    module_type = type(module)
    available_save_region_names = getattr(
        module_type, "AVAILABLE_REMAT_SAVE_REGIONS", ()
    )
    if not available_save_region_names:
        return ()

    save_region_owner = _class_defining_attribute(
        module_type, "AVAILABLE_REMAT_SAVE_REGIONS"
    )
    forward_owner = _class_defining_attribute(module_type, "forward")

    # Available save regions describe call sites in a particular forward method.
    # If a subclass replaces that forward without redeclaring the regions, the
    # inherited names may no longer exist in the active implementation.
    if forward_owner is not save_region_owner and issubclass(
        forward_owner, save_region_owner
    ):
        return ()
    return tuple(available_save_region_names)


def maybe_remat_save_region(
    fn: Callable[_P, _R], name: str, *, owner: nn.Module
) -> Callable[_P, _R]:
    """Return ``fn`` as a retained remat region when selected, else unchanged."""
    selection = remat_save_region_selection(owner)
    breakpoint()
    if selection is None:
        return fn
    assert name in _available_remat_save_region_names(
        owner
    ), f"{type(owner).__name__} used unavailable remat region {name!r}"
    qualified_save_region_name = selection.qualified_name_by_local_name.get(name)
    if qualified_save_region_name is None:
        return fn
    return require_torch_remat().region(fn, qualified_save_region_name, recompute=False)


def maybe_remat_recompute_needs(owner: nn.Module, *tensors: torch.Tensor) -> None:
    """Persist selected-region outputs that a bare operation will consume."""
    selection = remat_save_region_selection(owner)
    if selection is None or not selection.qualified_name_by_local_name:
        return
    require_torch_remat().recompute_needs_tensor(*tensors)


def is_remat_recomputing(owner: nn.Module) -> bool:
    """Return whether execution is inside a torch_remat replay."""
    if remat_save_region_selection(owner) is None:
        return False
    return bool(require_torch_remat().is_recomputing())


def remat_save_region_selection(
    module: nn.Module,
) -> RematSaveRegionSelection | None:
    """Return the effective save-region selection installed on ``module``."""
    return getattr(module, _REMAT_SAVE_REGION_SELECTION_ATTR, None)


def available_remat_save_regions(root: nn.Module) -> list[str]:
    """Return remat save-region names offered by ``root`` and descendants."""
    names = []
    for module_fqn, module in root.named_modules():
        for region_name in _available_remat_save_region_names(module):
            qualified_save_region_name = (
                f"{module_fqn}.{region_name}" if module_fqn else region_name
            )
            names.append(qualified_save_region_name)
    return names


def configure_remat_save_regions(
    root: nn.Module, save_patterns: list[str]
) -> tuple[list[str], list[str]]:
    """Enable available save regions matching glob patterns under ``root``."""
    selected_save_regions = []
    available_save_regions = []
    for module_fqn, module in root.named_modules():
        available_save_region_names = _available_remat_save_region_names(module)
        if not available_save_region_names:
            continue
        selected_save_region_names = {}
        for region_name in available_save_region_names:
            qualified_save_region_name = (
                f"{module_fqn}.{region_name}" if module_fqn else region_name
            )
            available_save_regions.append(qualified_save_region_name)
            if any(
                fnmatch(qualified_save_region_name, pattern)
                for pattern in save_patterns
            ):
                selected_save_region_names[region_name] = qualified_save_region_name
                selected_save_regions.append(qualified_save_region_name)
        selection = RematSaveRegionSelection(
            qualified_name_by_local_name=selected_save_region_names
        )
        setattr(module, _REMAT_SAVE_REGION_SELECTION_ATTR, selection)
    return selected_save_regions, available_save_regions
