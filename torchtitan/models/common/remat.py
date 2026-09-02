# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model-declared activation rematerialization regions.

Model code exposes candidate save regions with ``AVAILABLE_REMAT_SAVE_REGIONS``
and calls ``torch_remat.region`` directly. Regions whose recomputation can
change program behavior are also declared in ``REQUIRED_REMAT_SAVE_REGIONS``.
A remat activation-checkpointing policy installs the qualified region names and
save decisions on each module instance.
"""

# TODO: Complete Qwen 3.5 GatedDeltaNet integration. Its input projections need
# a deliberate grouping policy, and inner_gated_delta_net and out_proj still
# need remat save-region boundaries.
# TODO: GPT-OSS remat integration is intentionally tabled. Its custom attention
# and GptOssGroupedExperts override region-bearing shared implementations and
# need explicit remat save-region boundaries.
# TODO: Remaining Kimi K3 remat integration is intentionally tabled. MLA, KDA,
# KimiLatentMoE, and block residual projections still need explicit boundary
# audits.
# TODO: Audit whether TorchAOTokenDispatcher is used by supported training
# configurations before adding remat regions. Its padded permutation metadata
# and unpadding path need backend-specific dispatch/combine validation.

from __future__ import annotations

from fnmatch import fnmatch
from typing import Any

import torch.nn as nn


def _class_defining_attribute(module_type: type[Any], attribute_name: str) -> type[Any]:
    """Return the nearest class that defines ``attribute_name`` itself."""
    return next(base for base in module_type.__mro__ if attribute_name in base.__dict__)


def _declared_remat_save_region_names(
    module: nn.Module, attribute_name: str
) -> tuple[str, ...]:
    module_type = type(module)
    # Read from the instance so a model can expose only the call sites enabled
    # by its configuration while the defining class still owns the contract.
    save_region_names = getattr(module, attribute_name, ())
    if not save_region_names:
        return ()

    save_region_owner = _class_defining_attribute(module_type, attribute_name)
    forward_owner = _class_defining_attribute(module_type, "forward")

    # Save regions describe call sites in a particular forward method. If a
    # subclass replaces that forward without redeclaring the regions, the
    # inherited names may no longer exist in the active implementation.
    if forward_owner is not save_region_owner and issubclass(
        forward_owner, save_region_owner
    ):
        return ()
    return tuple(save_region_names)


def _available_remat_save_region_names(module: nn.Module) -> tuple[str, ...]:
    return _declared_remat_save_region_names(module, "AVAILABLE_REMAT_SAVE_REGIONS")


def _required_remat_save_region_names(module: nn.Module) -> tuple[str, ...]:
    required_save_region_names = _declared_remat_save_region_names(
        module, "REQUIRED_REMAT_SAVE_REGIONS"
    )
    available_save_region_names = _available_remat_save_region_names(module)
    assert set(required_save_region_names).issubset(available_save_region_names), (
        f"{type(module).__name__}.REQUIRED_REMAT_SAVE_REGIONS must be a subset "
        "of AVAILABLE_REMAT_SAVE_REGIONS"
    )
    return required_save_region_names


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


def required_remat_save_regions(root: nn.Module) -> list[str]:
    """Return save regions that ``root`` and descendants require retaining."""
    names = []
    for module_fqn, module in root.named_modules():
        for region_name in _required_remat_save_region_names(module):
            qualified_save_region_name = (
                f"{module_fqn}.{region_name}" if module_fqn else region_name
            )
            names.append(qualified_save_region_name)
    return names


def configure_remat_save_regions(
    root: nn.Module, save_patterns: list[str]
) -> tuple[list[str], list[str]]:
    """Install region names and save decisions on every region-bearing module."""
    selected_save_regions = []
    available_save_regions = []
    for module_fqn, module in root.named_modules():
        available_save_region_names = _available_remat_save_region_names(module)
        required_save_region_names = set(_required_remat_save_region_names(module))
        if not available_save_region_names:
            continue
        remat_region_names = {}
        is_remat_save_region = {}
        for region_name in available_save_region_names:
            qualified_save_region_name = (
                f"{module_fqn}.{region_name}" if module_fqn else region_name
            )
            available_save_regions.append(qualified_save_region_name)
            is_required = region_name in required_save_region_names
            is_saved = is_required or any(
                fnmatch(qualified_save_region_name, pattern)
                for pattern in save_patterns
            )
            remat_region_names[region_name] = qualified_save_region_name
            is_remat_save_region[region_name] = is_saved
            if is_saved:
                selected_save_regions.append(qualified_save_region_name)
        module.remat_region_names = remat_region_names
        module.is_remat_save_region = is_remat_save_region
    return selected_save_regions, available_save_regions
