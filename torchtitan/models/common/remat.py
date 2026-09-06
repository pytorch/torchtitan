# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Discover and configure model-declared remat save regions."""

from __future__ import annotations

from fnmatch import fnmatch
from typing import Any

import torch.nn as nn

from torchtitan.protocols.module import Module


def _class_defining_attribute(module_type: type[Any], attribute_name: str) -> type[Any]:
    """Return the nearest class that defines an attribute itself."""
    return next(base for base in module_type.__mro__ if attribute_name in base.__dict__)


def _available_remat_save_region_names(module: nn.Module) -> tuple[str, ...]:
    """Return the local remat regions implemented by the active ``forward``.

    ``AVAILABLE_REMAT_SAVE_REGIONS`` describes call sites in the ``forward``
    implementation associated with that declaration. A subclass may inherit
    the declaration when it also inherits that ``forward``. If it overrides
    ``forward``, the inherited names are ignored because those call sites may
    no longer exist. The subclass must redeclare the attribute to describe its
    replacement implementation.

    The value is read from the module instance so a class-level declaration can
    still be narrowed according to the module's configuration.
    """
    module_type = type(module)
    region_names = getattr(module, "AVAILABLE_REMAT_SAVE_REGIONS", ())
    if not region_names:
        return ()

    region_owner = _class_defining_attribute(
        module_type, "AVAILABLE_REMAT_SAVE_REGIONS"
    )
    if module_type.forward is not region_owner.forward:
        return ()
    return tuple(region_names)


def available_remat_save_regions(root: nn.Module) -> list[str]:
    """Return qualified remat save-region names from ``root`` and descendants."""
    region_names = []
    for module_fqn, module in root.named_modules():
        for local_name in _available_remat_save_region_names(module):
            region_names.append(
                f"{module_fqn}.{local_name}" if module_fqn else local_name
            )
    return region_names


def configure_remat_save_regions(
    root: nn.Module, save_patterns: list[str]
) -> tuple[list[str], list[str]]:
    """Install qualified names and save decisions on region-bearing modules.

    Region names are qualified relative to ``root``. Each shell-style pattern
    is matched against those qualified names. Regions without a match remain
    available but are configured for recomputation.

    Returns:
        The selected region names followed by all available region names.
    """
    selected_region_names = []
    available_region_names = []
    for module_fqn, module in root.named_modules():
        local_region_names = _available_remat_save_region_names(module)
        if not local_region_names:
            continue
        assert isinstance(module, Module), (
            "AVAILABLE_REMAT_SAVE_REGIONS must be declared on a "
            "torchtitan.protocols.module.Module"
        )

        qualified_names = {}
        save_decisions = {}
        for local_name in local_region_names:
            qualified_name = f"{module_fqn}.{local_name}" if module_fqn else local_name
            is_saved = any(
                fnmatch(qualified_name, pattern) for pattern in save_patterns
            )
            available_region_names.append(qualified_name)
            qualified_names[local_name] = qualified_name
            save_decisions[local_name] = is_saved
            if is_saved:
                selected_region_names.append(qualified_name)

        module.configure_remat_regions(qualified_names, save_decisions)

    return selected_region_names, available_region_names
