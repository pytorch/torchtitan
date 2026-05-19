# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Module Override Registry.

Allows swapping model components (RoPE, MoE, attention, norm, etc.) with
alternative implementations — without modifying config_registry functions.

Override authors write a Python module that registers a factory via the
``@register`` decorator. Users activate overrides by listing those modules
in ``--override.modules``.

See ``torchtitan/registry/DESIGN.md`` for the full design document.
"""

from __future__ import annotations

import importlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from torchtitan.tools.logging import logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from torchtitan.config import Configurable


@dataclass(kw_only=True, slots=True)
class OverrideConfig:
    modules: list[str] = field(default_factory=list)
    """
    Python module paths to import for module override registration.

    Each module is imported at startup, triggering any ``@register`` decorators
    it contains. All registered overrides are applied to the model config tree
    before model construction.

    Example: ``["torchtitan.registry.triton_rope", "vendor_x.overrides"]``

    See ``torchtitan/registry/DESIGN.md`` for details.
    """


@dataclass
class ModuleOverride:
    """Metadata for a registered module override."""

    name: str
    target_cls: type[Configurable.Config]
    factory: Callable[[Configurable.Config], Configurable.Config | None]
    description: str


_REGISTRY: dict[str, ModuleOverride] = {}


def register(
    name: str,
    *,
    target: type,
    description: str = "",
) -> Callable:
    """Decorator to register a module override factory.

    Args:
        name: Unique identifier for this override (e.g. ``"triton_rope"``).
        target: The ``Configurable.Config`` class to replace in the config tree.
        description: Human-readable description for logging.

    The decorated function must accept the original Config and return a
    replacement Config, or ``None`` to skip that particular instance.

    Example::

        @register("triton_rope", target=RoPE.Config,
                  description="Triton-based rotary embedding")
        def triton_rope_override(cfg: RoPE.Config) -> TritonRoPE.Config:
            return TritonRoPE.Config(dim=cfg.dim, ...)
    """

    def decorator(fn: Callable) -> Callable:
        if name in _REGISTRY:
            raise ValueError(
                f"Module override '{name}' is already registered. "
                f"Each override name must be unique across all imported modules."
            )
        _REGISTRY[name] = ModuleOverride(
            name=name,
            target_cls=target,
            factory=fn,
            description=description,
        )
        return fn

    return decorator


def _check_conflicts() -> None:
    """Raise if two overrides target the same Config class."""
    by_target: dict[type, list[str]] = defaultdict(list)
    for override in _REGISTRY.values():
        by_target[override.target_cls].append(override.name)

    conflicts = {target: names for target, names in by_target.items() if len(names) > 1}
    if conflicts:
        lines: list[str] = []
        for target, names in conflicts.items():
            lines.append("  " + target.__qualname__ + ": " + ", ".join(sorted(names)))
        raise ValueError(
            "Conflicting module overrides target the same Config class. "
            "Import fewer override modules to resolve:\n" + "\n".join(lines)
        )


def apply_overrides(
    model_config: Configurable.Config,
    override_config: OverrideConfig,
) -> list[str]:
    """Import override modules, check for conflicts, apply to config tree.

    Returns a list of human-readable log lines describing each replacement.
    """
    for mod_path in override_config.modules:
        try:
            importlib.import_module(mod_path)
        except ImportError as e:
            raise ImportError(
                f"Failed to import override module '{mod_path}': {e}"
            ) from e

    _check_conflicts()

    replacements: list[str] = []
    for override in _REGISTRY.values():
        for fqn, cfg, parent, attr in model_config.traverse(override.target_cls):
            new_cfg = override.factory(cfg)
            if new_cfg is None:
                continue
            if isinstance(parent, list) and isinstance(attr, int):
                parent[attr] = new_cfg
            elif isinstance(attr, str):
                setattr(parent, attr, new_cfg)
            replacements.append(
                f"[Override] {override.name}: {fqn} "
                f"{type(cfg).__qualname__} -> {type(new_cfg).__qualname__}"
            )

    if replacements:
        for line in replacements:
            logger.info(line)
        logger.info(f"Applied {len(replacements)} module override(s)")

    return replacements


def clear_registry() -> None:
    """Remove all registered overrides. Intended for testing."""
    _REGISTRY.clear()
