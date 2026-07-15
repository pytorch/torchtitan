# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Configurable override mechanism.

Swaps any ``Configurable`` (model components, optimizer, loss, dataloader, …)
for an alternative implementation — without modifying config_registry functions
or any other in-repo code.

An override author writes a Python module that registers a factory via the
``@override`` decorator, then the user activates it by listing that module in
``--override.imports``. Overrides are applied to the config tree(s) after config
construction and before any ``build()``.

Per-instance targeting is first-class: ``@override(..., fqns=[...])`` selects
*which* matched nodes to replace by FQN glob, and conflict detection is per-node
— two overrides only collide when they claim the same node, not merely the same
Config class. This supports e.g. "fused MoE on these layers only" and A/B-ing a
kernel across layers.

By default, a target also matches subclasses of that Config class. Override
authors can pass ``exact=True`` when the replacement is valid only for the
target's concrete contract.

See ``torchtitan/overrides/README.md`` for the full design document.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field, fields, is_dataclass
from fnmatch import fnmatch
from typing import cast, TYPE_CHECKING, TypeVar

from torchtitan.tools.logging import logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from torchtitan.config.configurable import Configurable

# Precise return type for ``derive``: the constructed ``target_config`` class.
_ConfigT = TypeVar("_ConfigT", bound="Configurable.Config")


@dataclass(kw_only=True, slots=True)
class OverrideConfig:
    imports: list[str] = field(default_factory=list)
    """
    Python module paths to import for override registration.

    Each module is imported once at startup, triggering any ``@override``
    decorators it defines. Only overrides registered by these modules (or their
    submodules) are applied — not the entire global registry.

    Example: ``["torchtitan.overrides.fused_swiglu", "vendor_x.kernels"]``

    See ``torchtitan/overrides/README.md`` for details.
    """


@dataclass
class Override:
    """A single registered override.

    ``factory`` constructs the replacement config from the matched config.
    ``fqns`` selects which matched nodes the override claims (see
    :func:`override`). ``exact`` restricts the match to exactly ``target_cls``
    rather than subclasses.
    """

    name: str
    target_cls: type[Configurable.Config]
    factory: Callable[[Configurable.Config], Configurable.Config]
    fqns: list[str] | None
    description: str
    origin_module: str
    exact: bool = False
    predicate: Callable[[Configurable.Config], bool] | None = None

    def matches(self, fqn: str, cfg: Configurable.Config) -> bool:
        """Whether this override claims the node at ``fqn``.

        ``fqns`` is ``None`` (every instance of ``target_cls``) or a list of FQN
        globs (``fnmatch``; ``*`` also crosses ``.``); a node matches if any glob
        matches its FQN.
        """
        if self.predicate is not None and not self.predicate(cfg):
            return False
        if self.fqns is None:
            return True
        return any(fnmatch(fqn, pattern) for pattern in self.fqns)


_REGISTRY: dict[str, Override] = {}


def override(
    name: str,
    *,
    target: type[Configurable.Config],
    fqns: list[str] | None = None,
    predicate: Callable[[Configurable.Config], bool] | None = None,
    description: str = "",
    exact: bool = False,
) -> Callable:
    """Decorator to register an override factory.

    Args:
        name: Unique identifier for this override (e.g. ``"triton_rope"``).
        target: The ``Configurable.Config`` class to replace. Instances of this
            class (and its subclasses) are candidates; ``fqns`` narrows them.
        fqns: Per-instance selector. ``None`` (default) claims every instance; a
            list of FQN globs (``fnmatch`` style, where ``*`` also crosses
            ``.``) claims instances whose FQN matches any glob. The FQN is the
            bare module FQN, e.g. ``"layers.0.feed_forward"``.
        predicate: Per-config selector for filtering matches by config content.
        description: Human-readable description, surfaced in the override log.
        exact: If ``True``, claim only configs whose concrete type is exactly
            ``target``. The default ``False`` preserves subclass matching, useful
            when a replacement is valid for the full target contract.

    The decorated function takes the matched config and returns its replacement.

    Example — fuse MoE only on the later layers::

        @override("vendor_moe", target=MoE.Config,
                  fqns=["layers.1[0-9].moe"],
                  description="Vendor fused MoE")
        def _vendor_moe(cfg: MoE.Config) -> VendorMoE.Config:
            return VendorMoE.Config(num_experts=cfg.num_experts, ...)
    """
    # Lazy import avoids a config-package import cycle; safe because override
    # modules are imported well after the config package is loaded.
    from torchtitan.config.configurable import Configurable

    if not (isinstance(target, type) and issubclass(target, Configurable.Config)):
        raise TypeError(
            f"override('{name}', target=...) must be a Configurable.Config "
            f"subclass, got {target!r}. Targets like ModelSpec or a plain class "
            f"are not overridable; pick the component's `.Config`."
        )

    def decorator(fn: Callable) -> Callable:
        if name in _REGISTRY:
            raise ValueError(
                f"Override '{name}' is already registered (by module "
                f"'{_REGISTRY[name].origin_module}'). Each override name must be "
                f"unique across all imported modules."
            )
        _REGISTRY[name] = Override(
            name=name,
            target_cls=target,
            factory=fn,
            fqns=fqns,
            description=description,
            origin_module=fn.__module__,
            exact=exact,
            predicate=predicate,
        )
        return fn

    return decorator


def derive(
    source: Configurable.Config,
    target_cls: type[_ConfigT],
    **deltas: object,
) -> _ConfigT:
    """Build ``target_cls`` by copying fields shared with ``source``, then deltas.

    A robust alternative to hand-copying each field in an override factory:

    - fields on both ``source`` and ``target_cls`` → copied by name from ``source``
    - fields only on ``target_cls`` → from ``deltas``, else the field's default
    - fields only on ``source`` → dropped

    The point is version-robustness: when the target ``Config`` later gains a
    field (typically inherited, since replacements usually subclass the target),
    ``derive`` copies it automatically instead of silently reverting to the
    default. The factory states only the deltas it genuinely changes.

    Copying is shallow (like ``dataclasses.replace``): nested configs are shared
    by reference, which is fine because the ``source`` node is being replaced and
    detached from the tree.

    Args:
        source: The matched source config (a dataclass instance).
        target_cls: The replacement ``Configurable.Config`` *class*.
        deltas: Field values to set/override on the replacement. A key not
            present on ``target_cls`` raises ``ValueError``.

    Returns:
        A ``target_cls`` instance.

    Example::

        @override("triton_rope", target=RoPE.Config)
        def triton_rope(cfg: RoPE.Config) -> TritonRoPE.Config:
            return derive(cfg, TritonRoPE.Config, block_size=128)
    """
    if not (isinstance(target_cls, type) and is_dataclass(target_cls)):
        raise TypeError(
            f"derive() target_cls must be a dataclass type, got {target_cls!r}"
        )

    target_field_names = {f.name for f in fields(target_cls) if f.init}
    unknown = set(deltas) - target_field_names
    if unknown:
        raise ValueError(
            f"derive() got deltas for fields not on {target_cls.__qualname__}: "
            f"{sorted(unknown)}"
        )

    source_field_names = (
        {f.name for f in fields(source)} if is_dataclass(source) else set()
    )
    kwargs = dict(deltas)
    for name in target_field_names:
        # Delta wins; otherwise copy a shared field; otherwise leave the
        # target's own default to apply (do not pass the key).
        if name not in kwargs and name in source_field_names:
            kwargs[name] = getattr(source, name)
    # ``is_dataclass`` narrowed ``target_cls`` to a generic dataclass type,
    # erasing ``_ConfigT``; cast back so the precise return type is preserved.
    return cast(_ConfigT, target_cls(**kwargs))


def _resolve_active(imports: list[str]) -> list[Override]:
    """Return overrides registered by the listed import modules (or submodules).

    Provenance is the module where the factory is defined (``fn.__module__``),
    matched against each listed import by exact name or submodule prefix. This
    keeps application strictly limited to the user's ``override.imports`` even
    when other code paths have registered overrides into the global table.
    """
    prefixes = tuple(imports)
    active = []
    for ov in _REGISTRY.values():
        origin = ov.origin_module
        if any(origin == p or origin.startswith(p + ".") for p in prefixes):
            active.append(ov)
    return active


@dataclass
class _Claim:
    """One (override, matched node) pair, collected before any mutation."""

    ov: Override
    fqn: str
    cfg: Configurable.Config
    parent: object | None
    attr: str | int | None


def _collect_claims(
    active: list[Override], config_root: Configurable.Config
) -> list[_Claim]:
    """Traverse the original tree and gather every node each override claims.

    Collecting before mutating makes application order-independent: replacements
    are never re-traversed, so one override cannot affect another's matches.
    """
    claims: list[_Claim] = []
    for ov in active:
        for fqn, cfg, parent, attr in config_root.traverse(ov.target_cls):
            if ov.exact and type(cfg) is not ov.target_cls:
                continue
            if ov.matches(fqn, cfg):
                claims.append(_Claim(ov=ov, fqn=fqn, cfg=cfg, parent=parent, attr=attr))
    return claims


def _check_node_conflicts(claims: list[_Claim]) -> None:
    """Raise if two *different* overrides claim the same node or nested nodes.

    Per-node, not per-class: two overrides of the same Config class on disjoint
    FQNs do not conflict. Conflicts are (a) the exact same node claimed twice, or
    (b) one override claiming an ancestor of another override's node — both make
    the outcome order-dependent.
    """
    # (a) exact same node, identified by container + key.
    by_node: dict[tuple[int, str | int | None], str] = {}
    for c in claims:
        key = (id(c.parent), c.attr)
        owner = by_node.get(key)
        if owner is not None and owner != c.ov.name:
            raise ValueError(
                f"Overrides '{owner}' and '{c.ov.name}' both claim node "
                f"'{c.fqn}'. Narrow their `fqns` selectors so they target "
                f"disjoint nodes."
            )
        by_node[key] = c.ov.name

    # (b) one node is an ancestor of another (nested replacement).
    for a in claims:
        for b in claims:
            if a.ov.name == b.ov.name:
                continue
            if b.fqn.startswith(a.fqn + "."):
                raise ValueError(
                    f"Override '{a.ov.name}' claims '{a.fqn}', an ancestor of "
                    f"'{b.fqn}' claimed by '{b.ov.name}'. Nested overrides are "
                    f"order-dependent; narrow their `fqns` selectors."
                )


def apply_overrides(
    override_config: OverrideConfig,
    config_root: Configurable.Config,
) -> list[str]:
    """Import override modules and apply active overrides to the config tree.

    Args:
        override_config: The override settings (which modules to import).
        config_root: The config tree to traverse and mutate in place. The
            trainer passes the top-level ``Trainer.Config``; the model config
            nested under ``ModelSpec`` is reached via ``ModelSpec.traverse``.

    Returns a list of human-readable log lines describing each replacement.
    """
    for mod_path in override_config.imports:
        try:
            importlib.import_module(mod_path)
        except ImportError as e:
            raise ImportError(
                f"Failed to import override module '{mod_path}': {e}"
            ) from e

    active = _resolve_active(override_config.imports)
    claims = _collect_claims(active, config_root)
    _check_node_conflicts(claims)

    replacements: list[str] = []
    for c in claims:
        new_cfg = c.ov.factory(c.cfg)
        if isinstance(c.parent, list) and isinstance(c.attr, int):
            c.parent[c.attr] = new_cfg
        elif isinstance(c.parent, dict):
            c.parent[c.attr] = new_cfg
        elif isinstance(c.attr, str):
            setattr(c.parent, c.attr, new_cfg)
        else:
            raise ValueError(
                f"Override '{c.ov.name}' claims root config '{c.fqn}', "
                "which cannot be replaced in place."
            )
        replacements.append(
            f"[Override] {c.ov.name}: {c.fqn} "
            f"{type(c.cfg).__qualname__} -> {type(new_cfg).__qualname__}"
        )

    if replacements:
        for line in replacements:
            logger.info(line)
        logger.info(f"Applied {len(replacements)} override(s)")
    elif override_config.imports:
        # The user asked for overrides but nothing matched — likely a wrong
        # `target` class, a `fqns` glob that matches no node, or a module that
        # registered no override. Surface it rather than silently no-op.
        logger.warning(
            f"override.imports={override_config.imports} produced no "
            f"replacements. Check that the imported modules register overrides "
            f"whose `target` and `fqns` match nodes in the config tree."
        )

    return replacements


def clear_overrides() -> None:
    """Remove all registered overrides. Intended for testing."""
    _REGISTRY.clear()
