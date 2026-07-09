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
import json
from dataclasses import dataclass, field, fields, is_dataclass
from fnmatch import fnmatch
from typing import Any, cast, TYPE_CHECKING, TypeVar

from torchtitan.tools.logging import logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from torchtitan.config.configurable import Configurable

# Precise return type for ``derive``: the constructed ``target_config`` class.
_ConfigT = TypeVar("_ConfigT", bound="Configurable.Config")

# One ``override.imports`` entry: a bare module path, or a ``(module_path,
# kwargs)`` tuple whose ``kwargs`` are forwarded to that module's override
# factories. Kept as a runtime object (not just an annotation) so the CLI parser
# can register a tyro rule keyed on ``list[OverrideImport]``.
OverrideImport = str | tuple[str, dict[str, Any]]


@dataclass(kw_only=True, slots=True)
class OverrideConfig:
    imports: list[OverrideImport] = field(default_factory=list)
    """
    Modules to import for override registration, each optionally carrying kwargs.

    An entry is either:

    - a module path string, e.g. ``"torchtitan.overrides.fused_swiglu"``; or
    - a ``(module_path, kwargs)`` tuple, e.g.
      ``("my_pkg.triton_rope", {"block_size": 256})``. The ``kwargs`` are passed
      to every override factory registered by that module when it is applied, so
      two config trees can share one override module yet configure it
      differently (e.g. the RL trainer and generator activate one HybridEP
      dispatch module with opposite capacity factors).

    Each module is imported once at startup, triggering any ``@override``
    decorators it defines. Only overrides registered by these modules (or their
    submodules) are applied -- not the entire global registry.

    On the CLI, ``--override.imports`` takes space- or comma-separated module
    paths; attach kwargs to a module with ``module=<json-object>`` (see
    :func:`parse_cli_imports`), e.g.
    ``--override.imports 'my_pkg.triton_rope={"block_size": 256}'``.

    See ``torchtitan/overrides/README.md`` for details.
    """


def parse_cli_imports(tokens: list[str]) -> list[OverrideImport]:
    """Parse ``--override.imports`` CLI tokens into ``imports`` entries.

    Each token is one of:

    - ``module=<json-object>`` -- a module path whose kwargs follow the name
      directly (one token), e.g. ``my_pkg.triton_rope={"block_size": 256}``; or
    - a plain module path, or a comma-separated group of them (``a.b,c.d``),
      for entries without kwargs.

    So ``["fused_swiglu", 'my_pkg.triton_rope={"block_size": 256}']`` parses to
    ``["fused_swiglu", ("my_pkg.triton_rope", {"block_size": 256})]``. A token
    with kwargs must be a single shell token, so quote it.
    """
    entries: list[OverrideImport] = []
    for token in tokens:
        module, sep, raw_kwargs = token.partition("=")
        if sep:
            # module=<json>: kwargs attached directly to this module. A module
            # path never contains '=', so the first '=' is always the separator
            # and any '=' inside the JSON is preserved.
            kwargs = json.loads(raw_kwargs)
            if not isinstance(kwargs, dict):
                raise ValueError(
                    f"override.imports kwargs for '{module}' must be a JSON "
                    f"object, got {raw_kwargs!r}."
                )
            entries.append((module, kwargs))
        else:
            entries.extend(part for part in token.split(",") if part)
    return entries


def format_cli_imports(entries: list[OverrideImport]) -> list[str]:
    """Serialize ``imports`` entries back to CLI tokens (inverse of the parse)."""
    return [
        entry if isinstance(entry, str) else f"{entry[0]}={json.dumps(entry[1])}"
        for entry in entries
    ]


@dataclass
class Override:
    """A single registered override.

    ``factory`` constructs the replacement config from the matched config, plus
    any keyword arguments supplied by the ``override.imports`` entry that
    activated this override. ``fqns`` selects which matched nodes the override
    claims (see :func:`override`). ``exact`` restricts the match to exactly
    ``target_cls`` rather than subclasses.
    """

    name: str
    target_cls: type[Configurable.Config]
    factory: Callable[..., Configurable.Config]
    fqns: list[str] | None
    description: str
    origin_module: str
    exact: bool = False

    def matches(self, fqn: str) -> bool:
        """Whether this override claims the node at ``fqn``.

        ``fqns`` is ``None`` (every instance of ``target_cls``) or a list of FQN
        globs (``fnmatch``; ``*`` also crosses ``.``); a node matches if any glob
        matches its FQN.
        """
        if self.fqns is None:
            return True
        return any(fnmatch(fqn, pattern) for pattern in self.fqns)


_REGISTRY: dict[str, Override] = {}


def override(
    name: str,
    *,
    target: type[Configurable.Config],
    fqns: list[str] | None = None,
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
            bare module FQN, e.g. ``"layers.0.feed_forward"``. (A general
            predicate selector may be added later if needed.)
        description: Human-readable description, surfaced in the override log.
        exact: If ``True``, claim only configs whose concrete type is exactly
            ``target``. The default ``False`` preserves subclass matching, useful
            when a replacement is valid for the full target contract.

    The decorated function takes the matched config as its first positional
    argument and returns its replacement. It may also declare keyword parameters
    (or ``**kwargs``); those are filled from the ``(module_path, kwargs)`` entry
    that activated the override, letting one override module be configured per
    actor (e.g. trainer vs. generator). A kwarg the factory does not accept
    raises ``TypeError`` at apply time (normal Python argument binding).

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


def _resolve_active(
    entries: list[tuple[str, dict[str, Any]]]
) -> list[tuple[Override, dict[str, Any]]]:
    """Pair each active override with the kwargs of the entry that activated it.

    Provenance is the module where the factory is defined (``fn.__module__``),
    matched against each listed import by exact name or submodule prefix. This
    keeps application strictly limited to the user's ``override.imports`` even
    when other code paths have registered overrides into the global table.

    When several listed modules are prefixes of one override's origin (e.g.
    ``"pkg"`` and ``"pkg.sub"``), the most specific (longest) one supplies its
    kwargs. An entry that carries kwargs but activates no override is a mistake
    (wrong module path, or its kwargs were shadowed by a more specific entry),
    so it raises rather than silently dropping the kwargs.
    """
    resolved: list[tuple[Override, dict[str, Any]]] = []
    used = [False] * len(entries)
    for ov in _REGISTRY.values():
        origin = ov.origin_module
        candidates = [
            i
            for i, (mod, _) in enumerate(entries)
            if origin == mod or origin.startswith(mod + ".")
        ]
        if not candidates:
            continue
        best = max(candidates, key=lambda i: len(entries[i][0]))
        used[best] = True
        resolved.append((ov, entries[best][1]))

    for i, (mod, kwargs) in enumerate(entries):
        if kwargs and not used[i]:
            raise ValueError(
                f"override.imports entry '{mod}' passed kwargs {sorted(kwargs)} "
                "but activated no override. Check the module path and that it "
                "registers an @override (and is not shadowed by a more specific "
                "listed module)."
            )
    return resolved


@dataclass
class _Claim:
    """One (override, matched node) pair, collected before any mutation.

    ``kwargs`` are the values from the ``override.imports`` entry that activated
    the override, forwarded to its factory when the claim is applied.
    """

    ov: Override
    fqn: str
    cfg: Configurable.Config
    parent: object | None
    attr: str | int | None
    kwargs: dict[str, Any] = field(default_factory=dict)


def _collect_claims(
    resolved: list[tuple[Override, dict[str, Any]]],
    config_root: Configurable.Config,
) -> list[_Claim]:
    """Traverse the original tree and gather every node each override claims.

    Collecting before mutating makes application order-independent: replacements
    are never re-traversed, so one override cannot affect another's matches.
    """
    claims: list[_Claim] = []
    for ov, kwargs in resolved:
        for fqn, cfg, parent, attr in config_root.traverse(ov.target_cls):
            if ov.exact and type(cfg) is not ov.target_cls:
                continue
            if ov.matches(fqn):
                claims.append(
                    _Claim(
                        ov=ov,
                        fqn=fqn,
                        cfg=cfg,
                        parent=parent,
                        attr=attr,
                        kwargs=kwargs,
                    )
                )
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
    # Each entry is a bare module path or a (module_path, kwargs) tuple; the
    # kwargs are forwarded to that module's factories (see OverrideConfig.imports).
    entries: list[tuple[str, dict[str, Any]]] = [
        (e, {}) if isinstance(e, str) else (e[0], e[1]) for e in override_config.imports
    ]
    for mod_path, _ in entries:
        try:
            importlib.import_module(mod_path)
        except ImportError as e:
            raise ImportError(
                f"Failed to import override module '{mod_path}': {e}"
            ) from e

    resolved = _resolve_active(entries)
    claims = _collect_claims(resolved, config_root)
    _check_node_conflicts(claims)

    replacements: list[str] = []
    for c in claims:
        # ``**{}`` for a bare (no-kwargs) entry is just ``factory(cfg)``; a kwarg
        # the factory does not accept raises TypeError here, naming the factory.
        new_cfg = c.ov.factory(c.cfg, **c.kwargs)
        if isinstance(c.parent, list) and isinstance(c.attr, int):
            c.parent[c.attr] = new_cfg
        elif isinstance(c.attr, str):
            setattr(c.parent, c.attr, new_cfg)
        else:
            raise ValueError(
                f"Override '{c.ov.name}' claims root config '{c.fqn}', "
                "which cannot be replaced in place."
            )
        kwargs_note = (
            " with " + ", ".join(f"{k}={v!r}" for k, v in c.kwargs.items())
            if c.kwargs
            else ""
        )
        replacements.append(
            f"[Override] {c.ov.name}: {c.fqn} "
            f"{type(c.cfg).__qualname__} -> {type(new_cfg).__qualname__}{kwargs_note}"
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
