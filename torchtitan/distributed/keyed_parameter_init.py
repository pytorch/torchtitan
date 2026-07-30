# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import contextlib
import hashlib
import sys
from collections.abc import Iterable, Iterator
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from functools import partial
from types import CodeType
from typing import Any

import torch
import torch.func._random as stateless_random
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.utils._python_dispatch import TorchDispatchMode

from torchtitan.components.checkpoint_utils import canonical_fqn

from .logical_shards import (
    LocalLogicalTensor,
    LogicalShardLayout,
    normalize_logical_tensor,
)


__all__ = [
    "ParameterInitRegistry",
    "capture_parameter_init_registry",
    "keyed_parameter_init",
    "keyed_parameter_init_is_active",
    "keyed_parameter_scope",
]


_PARAM_HASH_DOMAIN = b"torchtitan-keyed-parameter-v1"
_DRAW_HASH_DOMAIN = b"torchtitan-keyed-draw-v1"
_MODULE_ANNOTATIONS = "_keyed_parameter_init_annotations"
_IGNORED_SITE_MODULES = (
    "torch._compile",
    "torch._dynamo.eval_frame",
    "torch.compiler._compile",
    "torch.utils._python_dispatch",
)


def _stable_digest(domain: bytes, parts: Iterable[str]) -> bytes:
    digest = hashlib.sha256(domain)
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.digest()


def _fold_digest(key: torch.Tensor, digest: bytes) -> torch.Tensor:
    for offset in range(0, len(digest), 8):
        word = int.from_bytes(digest[offset : offset + 8], byteorder="big")
        key = stateless_random.fold_in(key, word)
    return key


@dataclass(frozen=True, slots=True)
class ParameterInitRegistry:
    """Canonical parameter identities captured before pipeline splitting."""

    canonical_fqns: tuple[str, ...]
    initializing_fqns: frozenset[str]


@dataclass(frozen=True, slots=True)
class _ParameterInitAnnotation:
    canonical_fqn: str
    alias_fqn: str
    initializes: bool


@dataclass(frozen=True, slots=True)
class _ParameterAlias:
    owner: nn.Module
    local_name: str
    fqn: str
    initializes: bool


@dataclass(frozen=True)
class _RegisteredParameterPath:
    owner: nn.Module
    local_name: str
    parameter: nn.Parameter
    fqn: str
    digest: bytes
    initializes: bool
    initial_logical: LocalLogicalTensor | None


@dataclass
class _KeyedInitTransaction:
    model_key: torch.Tensor
    parameter_paths: dict[tuple[int, str], _RegisteredParameterPath]
    completed_paths: set[tuple[int, str]] = field(default_factory=set)


@dataclass
class _ActiveParameter:
    registration: _RegisteredParameterPath
    parameter_key: torch.Tensor
    boundary_code: CodeType
    draw_counts: dict[bytes, int] = field(default_factory=dict)
    device_keys: dict[torch.device, torch.Tensor] = field(default_factory=dict)
    logical_tensors: dict[int, tuple[torch.Tensor, LocalLogicalTensor]] = field(
        default_factory=dict
    )

    def remember(self, tensor: torch.Tensor, logical: LocalLogicalTensor) -> None:
        self.logical_tensors[id(tensor)] = (tensor, logical)

    def known_logical_tensor(self, tensor: torch.Tensor) -> LocalLogicalTensor | None:
        entry = self.logical_tensors.get(id(tensor))
        if entry is None or entry[0] is not tensor:
            return None
        return entry[1]

    def logical_tensor(self, tensor: torch.Tensor) -> LocalLogicalTensor:
        logical = self.known_logical_tensor(tensor)
        if logical is None:
            initial_logical = self.registration.initial_logical
            if (
                initial_logical is not None
                and not isinstance(tensor, DTensor)
                and tensor is not initial_logical.local_tensor
                and torch._C._overlaps(tensor, initial_logical.local_tensor)
            ):
                raise ValueError(
                    "keyed parameter initialization cannot infer the logical "
                    "layout of an overlapping tensor view"
                )
            logical = normalize_logical_tensor(tensor)
            self.remember(tensor, logical)
        return logical

    def key_on(self, device: torch.device) -> torch.Tensor:
        key = self.device_keys.get(device)
        if key is None:
            key = self.parameter_key.to(device=device)
            self.device_keys[device] = key
        return key


_ACTIVE_TRANSACTION: ContextVar[_KeyedInitTransaction | None] = ContextVar(
    "keyed_parameter_init_transaction", default=None
)
_ACTIVE_PARAMETER: ContextVar[_ActiveParameter | None] = ContextVar(
    "keyed_parameter_init_parameter", default=None
)


def _normalize_models(
    models: nn.Module | Iterable[nn.Module],
) -> tuple[nn.Module, ...]:
    model_tuple = (models,) if isinstance(models, nn.Module) else tuple(models)
    if not model_tuple:
        raise ValueError("keyed_parameter_init requires at least one model")
    if not all(isinstance(model, nn.Module) for model in model_tuple):
        raise TypeError("keyed_parameter_init models must be nn.Module instances")
    return model_tuple


def _is_skip_initializer(initializer: Any) -> bool:
    # Import lazily because Module imports this integration module.
    from torchtitan.models.common.param_init import skip_param_init

    return initializer is skip_param_init


def _is_shape_dependent_initializer(initializer: Any) -> bool:
    while isinstance(initializer, partial):
        initializer = initializer.func
    return initializer in {
        nn.init.dirac_,
        nn.init.kaiming_normal_,
        nn.init.kaiming_uniform_,
        nn.init.orthogonal_,
        nn.init.sparse_,
        nn.init.xavier_normal_,
        nn.init.xavier_uniform_,
    }


def _is_full_local_tensor(logical: LocalLogicalTensor) -> bool:
    shape = tuple(logical.local_tensor.shape)
    zeros = (0,) * len(shape)
    return (
        logical.layout.global_shape == shape
        and logical.layout.global_offsets == (zeros,)
        and logical.layout.local_offsets == (zeros,)
        and logical.layout.local_sizes == (shape,)
    )


def _capture_parameter_init_registry(
    models: tuple[nn.Module, ...],
) -> ParameterInitRegistry:
    all_modules: dict[int, nn.Module] = {}
    for model in models:
        for module in model.modules():
            all_modules[id(module)] = module
    for module in all_modules.values():
        module.__dict__[_MODULE_ANNOTATIONS] = {}

    aliases_by_parameter: dict[int, tuple[nn.Parameter, list[_ParameterAlias]]] = {}
    for model in models:
        for module_fqn, owner in model.named_modules(remove_duplicate=False):
            direct_parameters = tuple(
                owner.named_parameters(recurse=False, remove_duplicate=False)
            )
            if not direct_parameters:
                continue
            param_init = getattr(owner, "_param_init", None)
            if param_init is None:
                owner_name = canonical_fqn(module_fqn) or "<root>"
                raise ValueError(
                    "keyed parameter initialization requires explicit param_init, "
                    f"but {owner_name} uses reset_parameters"
                )

            for local_name, parameter in direct_parameters:
                if local_name not in param_init:
                    owner_name = canonical_fqn(module_fqn) or "<root>"
                    raise ValueError(
                        f"no explicit initializer for {owner_name}.{local_name}"
                    )
                name = f"{module_fqn}.{local_name}" if module_fqn else local_name
                fqn = canonical_fqn(name)
                if not fqn:
                    raise ValueError(
                        "keyed parameter initialization found an empty FQN"
                    )
                alias = _ParameterAlias(
                    owner=owner,
                    local_name=local_name,
                    fqn=fqn,
                    initializes=not _is_skip_initializer(param_init[local_name]),
                )
                entry = aliases_by_parameter.get(id(parameter))
                if entry is None:
                    aliases_by_parameter[id(parameter)] = (parameter, [alias])
                else:
                    if entry[0] is not parameter:
                        raise RuntimeError(
                            "parameter identity was reused while building registry"
                        )
                    entry[1].append(alias)

    canonical_owners: dict[str, nn.Parameter] = {}
    digest_owners: dict[bytes, str] = {}
    initializing_fqns: set[str] = set()
    for parameter, aliases in aliases_by_parameter.values():
        initializing_aliases = [alias for alias in aliases if alias.initializes]
        initializing_paths = {
            (id(alias.owner), alias.local_name) for alias in initializing_aliases
        }
        if len(initializing_paths) > 1:
            fqns = sorted(alias.fqn for alias in initializing_aliases)
            raise ValueError(
                "tied parameters require exactly one non-skip initializer, "
                f"but found initializing aliases {fqns}"
            )
        selected = min(initializing_aliases or aliases, key=lambda alias: alias.fqn)
        canonical = selected.fqn
        owner = canonical_owners.setdefault(canonical, parameter)
        if owner is not parameter:
            raise ValueError(f"canonical parameter FQN {canonical!r} is not unique")

        digest = _stable_digest(_PARAM_HASH_DOMAIN, (canonical,))
        digest_owner = digest_owners.setdefault(digest, canonical)
        if digest_owner != canonical:
            raise RuntimeError(
                "parameter FQN digest collision between "
                f"{digest_owner!r} and {canonical!r}"
            )
        if initializing_aliases:
            initializing_fqns.add(canonical)

        aliases_by_path: dict[tuple[int, str], list[_ParameterAlias]] = {}
        for alias in aliases:
            aliases_by_path.setdefault((id(alias.owner), alias.local_name), []).append(
                alias
            )
        for path_aliases in aliases_by_path.values():
            path = path_aliases[0]
            annotation = _ParameterInitAnnotation(
                canonical_fqn=canonical,
                alias_fqn=min(alias.fqn for alias in path_aliases),
                initializes=any(alias is selected for alias in path_aliases)
                and bool(initializing_aliases),
            )
            annotations = getattr(path.owner, _MODULE_ANNOTATIONS)
            existing = annotations.get(path.local_name)
            if existing is not None and existing != annotation:
                raise ValueError(
                    f"parameter owner path {path.local_name!r} has conflicting "
                    "canonical annotations"
                )
            annotations[path.local_name] = annotation

    return ParameterInitRegistry(
        canonical_fqns=tuple(sorted(canonical_owners)),
        initializing_fqns=frozenset(initializing_fqns),
    )


def capture_parameter_init_registry(model: nn.Module) -> ParameterInitRegistry:
    """Capture global parameter identities before pipeline model splitting."""
    if not isinstance(model, nn.Module):
        raise TypeError("capture_parameter_init_registry expects an nn.Module")
    return _capture_parameter_init_registry((model,))


def _preflight_parameter_paths(
    models: tuple[nn.Module, ...],
    registry: ParameterInitRegistry,
) -> dict[tuple[int, str], _RegisteredParameterPath]:
    known_fqns = set(registry.canonical_fqns)
    if not registry.initializing_fqns <= known_fqns:
        raise ValueError("registry initializing_fqns must be canonical parameter FQNs")

    paths: dict[tuple[int, str], _RegisteredParameterPath] = {}
    paths_by_fqn: dict[str, list[_RegisteredParameterPath]] = {}
    seen_modules: set[int] = set()
    for model in models:
        for owner in model.modules():
            if id(owner) in seen_modules:
                continue
            seen_modules.add(id(owner))
            direct_parameters = dict(
                owner.named_parameters(recurse=False, remove_duplicate=False)
            )
            annotations = getattr(owner, _MODULE_ANNOTATIONS, None)
            if not direct_parameters:
                if annotations:
                    raise ValueError(
                        "keyed parameter annotations refer to removed parameters"
                    )
                continue
            if not isinstance(annotations, dict):
                raise ValueError(
                    "parameter owner is missing pre-pipeline keyed annotations"
                )
            stale_names = set(annotations) - set(direct_parameters)
            if stale_names:
                raise ValueError(
                    f"keyed parameter annotations refer to missing parameters {stale_names}"
                )

            param_init = getattr(owner, "_param_init", None)
            if param_init is None:
                raise ValueError(
                    "keyed parameter initialization does not support "
                    "reset_parameters fallback"
                )
            for local_name, parameter in direct_parameters.items():
                if local_name not in param_init:
                    raise ValueError(
                        f"no explicit initializer for local parameter {local_name!r}"
                    )
                annotation = annotations.get(local_name)
                if not isinstance(annotation, _ParameterInitAnnotation):
                    raise ValueError(
                        f"local parameter {local_name!r} has no keyed annotation"
                    )
                if annotation.canonical_fqn not in known_fqns:
                    raise ValueError(
                        f"parameter FQN {annotation.canonical_fqn!r} is not in registry"
                    )
                if annotation.initializes and _is_skip_initializer(
                    param_init[local_name]
                ):
                    raise ValueError(
                        f"initializing alias {annotation.alias_fqn!r} uses skip_param_init"
                    )

                initial_logical = (
                    normalize_logical_tensor(parameter)
                    if annotation.initializes
                    else None
                )
                if (
                    initial_logical is not None
                    and not isinstance(parameter, DTensor)
                    and not _is_full_local_tensor(initial_logical)
                    and _is_shape_dependent_initializer(param_init[local_name])
                ):
                    raise NotImplementedError(
                        "shape-dependent parameter initializers require a dense "
                        "tensor or a DTensor with a global public shape"
                    )

                digest = _stable_digest(_PARAM_HASH_DOMAIN, (annotation.canonical_fqn,))
                path = _RegisteredParameterPath(
                    owner=owner,
                    local_name=local_name,
                    parameter=parameter,
                    fqn=annotation.canonical_fqn,
                    digest=digest,
                    initializes=annotation.initializes,
                    initial_logical=initial_logical,
                )
                path_key = (id(owner), local_name)
                if path_key in paths:
                    raise RuntimeError("duplicate local parameter owner path")
                paths[path_key] = path
                paths_by_fqn.setdefault(path.fqn, []).append(path)

    for fqn, fqn_paths in paths_by_fqn.items():
        num_initializers = sum(path.initializes for path in fqn_paths)
        expected = fqn in registry.initializing_fqns
        if num_initializers != int(expected):
            raise ValueError(
                f"local parameter {fqn!r} has {num_initializers} initializing aliases"
            )
        selected_path = next(
            (path for path in fqn_paths if path.initializes), fqn_paths[0]
        )
        for path in fqn_paths:
            if path.parameter is not selected_path.parameter:
                setattr(path.owner, path.local_name, selected_path.parameter)
    return paths


class _ParameterScope(contextlib.AbstractContextManager[bool]):
    def __init__(
        self,
        transaction: _KeyedInitTransaction,
        registration: _RegisteredParameterPath,
        boundary_code: CodeType,
    ) -> None:
        self._transaction = transaction
        self._registration = registration
        self._boundary_code = boundary_code
        self._token: Token[_ActiveParameter | None] | None = None

    def __enter__(self) -> bool:
        if not self._registration.initializes:
            return False
        if _ACTIVE_PARAMETER.get() is not None:
            raise RuntimeError("keyed parameter initialization scopes cannot be nested")
        path_key = (id(self._registration.owner), self._registration.local_name)
        if path_key in self._transaction.completed_paths:
            raise RuntimeError(
                f"parameter initializer for {self._registration.fqn!r} ran twice"
            )

        parameter_key = _fold_digest(
            self._transaction.model_key, self._registration.digest
        )
        active = _ActiveParameter(
            registration=self._registration,
            parameter_key=parameter_key,
            boundary_code=self._boundary_code,
        )
        if self._registration.initial_logical is None:
            raise RuntimeError("initializing parameter has no logical shard metadata")
        active.remember(
            self._registration.parameter, self._registration.initial_logical
        )
        active.remember(
            self._registration.initial_logical.local_tensor,
            self._registration.initial_logical,
        )
        self._token = _ACTIVE_PARAMETER.set(active)
        return True

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._token is None:
            if self._registration.initializes:
                raise RuntimeError(
                    "keyed parameter initialization scope was not entered"
                )
            return
        _ACTIVE_PARAMETER.reset(self._token)
        self._token = None
        if exc_type is None:
            self._transaction.completed_paths.add(
                (id(self._registration.owner), self._registration.local_name)
            )


def keyed_parameter_init_is_active() -> bool:
    """Return whether the current context is a keyed initialization transaction."""
    return _ACTIVE_TRANSACTION.get() is not None


def keyed_parameter_scope(
    owner: nn.Module,
    local_name: str,
    parameter: nn.Parameter,
) -> contextlib.AbstractContextManager[bool]:
    """Associate an explicit parameter initializer with the active transaction."""
    transaction = _ACTIVE_TRANSACTION.get()
    if transaction is None:
        return contextlib.nullcontext(True)
    registration = transaction.parameter_paths.get((id(owner), local_name))
    if (
        registration is None
        or registration.owner is not owner
        or (registration.initializes and registration.parameter is not parameter)
    ):
        raise ValueError(
            f"parameter path {type(owner).__name__}.{local_name} is not registered "
            "in the active keyed transaction"
        )
    return _ParameterScope(transaction, registration, sys._getframe(1).f_code)


def _draw_site_digest(active: _ActiveParameter, func: Any) -> bytes:
    parts = [str(func)]
    frame = sys._getframe(1)
    found_boundary = False
    while frame is not None:
        module_name = frame.f_globals.get("__name__", "")
        if frame.f_code is active.boundary_code:
            parts.append(f"{module_name}:{frame.f_code.co_qualname}:{frame.f_lineno}")
            found_boundary = True
            break
        if module_name != __name__ and not module_name.startswith(
            _IGNORED_SITE_MODULES
        ):
            parts.append(f"{module_name}:{frame.f_code.co_qualname}:{frame.f_lineno}")
        frame = frame.f_back

    if not found_boundary:
        raise RuntimeError("random draw escaped its keyed parameter scope")
    return _stable_digest(_DRAW_HASH_DOMAIN, parts)


def _next_draw_key(
    active: _ActiveParameter, func: Any, device: torch.device
) -> torch.Tensor:
    site_digest = _draw_site_digest(active, func)
    iteration = active.draw_counts.get(site_digest, 0)
    active.draw_counts[site_digest] = iteration + 1
    draw_key = _fold_digest(active.key_on(device), site_digest)
    return stateless_random.fold_in(draw_key, iteration)


def _local_with_layout(
    tensor: torch.Tensor, layout: LogicalShardLayout
) -> LocalLogicalTensor:
    return LocalLogicalTensor(local_tensor=tensor, layout=layout)


def _drop_dimension(values: tuple[int, ...], dim: int) -> tuple[int, ...]:
    return values[:dim] + values[dim + 1 :]


def _propagate_select(
    active: _ActiveParameter,
    source: torch.Tensor,
    result: torch.Tensor,
    dim: int,
) -> None:
    logical = active.known_logical_tensor(source)
    if logical is None:
        return
    layout = logical.layout
    rank = len(layout.global_shape)
    dim %= rank

    local_dim_size = logical.local_tensor.shape[dim]
    if local_dim_size != layout.global_shape[dim] or any(
        global_offset[dim] != 0
        or local_offset[dim] != 0
        or local_size[dim] != local_dim_size
        for global_offset, local_offset, local_size in zip(
            layout.global_offsets,
            layout.local_offsets,
            layout.local_sizes,
            strict=True,
        )
    ):
        raise ValueError(
            "keyed initialization only supports select on an unsharded dimension"
        )

    selected_layout = LogicalShardLayout(
        global_shape=_drop_dimension(layout.global_shape, dim),
        global_offsets=tuple(
            _drop_dimension(offset, dim) for offset in layout.global_offsets
        ),
        local_offsets=tuple(
            _drop_dimension(offset, dim) for offset in layout.local_offsets
        ),
        local_sizes=tuple(_drop_dimension(size, dim) for size in layout.local_sizes),
    )
    active.remember(result, _local_with_layout(result, selected_layout))


def _propagate_layout(
    active: _ActiveParameter,
    func: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
) -> None:
    if not isinstance(result, torch.Tensor) or isinstance(result, DTensor):
        return

    if func in (
        torch.ops.aten.clone.default,
        torch.ops.aten.detach.default,
        torch.ops.aten.empty_like.default,
    ):
        logical = active.known_logical_tensor(args[0])
        if logical is not None:
            active.remember(result, _local_with_layout(result, logical.layout))
        return

    if func == torch.ops.aten.select.int:
        dim = args[1] if len(args) > 1 else kwargs["dim"]
        _propagate_select(active, args[0], result, dim)
        return

    if func == torch.ops.aten.where.self:
        logical_values = [active.known_logical_tensor(value) for value in args[1:3]]
        if logical_values == [None, None]:
            return
        if (
            logical_values[0] is None
            or logical_values[1] is None
            or logical_values[0].layout != logical_values[1].layout
        ):
            raise ValueError("torch.where combined incompatible logical shard layouts")
        layout = logical_values[0].layout
        active.remember(result, _local_with_layout(result, layout))


def _argument(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    index: int,
    name: str,
    default: float,
) -> Any:
    return args[index] if len(args) > index else kwargs.get(name, default)


def _keyed_random_(
    active: _ActiveParameter,
    func: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    target = args[0]
    if not isinstance(target, torch.Tensor):
        raise TypeError("in-place random operation target must be a tensor")
    if kwargs.get("generator") is not None:
        raise ValueError("keyed parameter initialization does not accept a generator")

    logical = active.logical_tensor(target)
    local_tensor = logical.local_tensor
    if local_tensor.is_meta:
        raise ValueError(
            "keyed parameter initialization requires materialized parameters"
        )
    key = _next_draw_key(active, func, local_tensor.device)
    layout = logical.layout

    if func == torch.ops.aten.normal_.default:
        stateless_random.normal_shards_(
            key,
            local_tensor,
            global_shape=layout.global_shape,
            global_offsets=layout.global_offsets,
            local_offsets=layout.local_offsets,
            local_sizes=layout.local_sizes,
            mean=_argument(args, kwargs, 1, "mean", 0.0),
            std=_argument(args, kwargs, 2, "std", 1.0),
        )
    else:
        stateless_random.uniform_shards_(
            key,
            local_tensor,
            global_shape=layout.global_shape,
            global_offsets=layout.global_offsets,
            local_offsets=layout.local_offsets,
            local_sizes=layout.local_sizes,
            low=_argument(args, kwargs, 1, "from", 0.0),
            high=_argument(args, kwargs, 2, "to", 1.0),
        )
    return target


class _KeyedParameterInitMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        active = _ACTIVE_PARAMETER.get()
        if active is not None:
            if func in (
                torch.ops.aten.normal_.default,
                torch.ops.aten.uniform_.default,
            ):
                return _keyed_random_(active, func, args, kwargs)
            if torch.Tag.nondeterministic_seeded in func.tags:
                raise NotImplementedError(
                    "keyed parameter initialization does not support random "
                    f"operator {func}"
                )

        result = func(*args, **kwargs)
        if active is not None:
            _propagate_layout(active, func, args, kwargs, result)
        return result


@contextlib.contextmanager
def keyed_parameter_init(
    models: nn.Module | Iterable[nn.Module],
    rng: stateless_random.StatefulPRNG,
    *,
    registry: ParameterInitRegistry | None = None,
) -> Iterator[None]:
    """Run one model initialization transaction with FQN-keyed random draws.

    The supported random operations are in-place ``normal_`` and ``uniform_``;
    this includes ``nn.init.trunc_normal_``. Initializers for a plain local shard
    must choose distribution arguments independently of its physical shape.
    DTensor initializers may use shape-dependent helpers because DTensor exposes
    its global shape.

    A canonical parameter FQN defines the numerical identity. Model transforms
    that replace one FQN with another, such as fused versus separate QKV
    parameters, intentionally define a different keyed numerical ground truth.

    If initialization raises, the RNG reservation is restored. Parameter writes
    that occurred before the error are not rolled back.
    """
    if _ACTIVE_TRANSACTION.get() is not None:
        raise RuntimeError(
            "keyed parameter initialization transactions cannot be nested"
        )

    from torchtitan.distributed.utils import get_spmd_backend

    if get_spmd_backend() == "spmd_types":
        raise NotImplementedError(
            "keyed parameter initialization requires logical shard metadata, "
            "which the spmd_types backend does not currently preserve"
        )

    model_tuple = _normalize_models(models)
    if registry is None:
        registry = _capture_parameter_init_registry(model_tuple)
    elif not isinstance(registry, ParameterInitRegistry):
        raise TypeError("registry must be a ParameterInitRegistry")
    parameter_paths = _preflight_parameter_paths(model_tuple, registry)

    saved_state = rng.get_state()
    try:
        model_key = rng.take_key()
        transaction = _KeyedInitTransaction(
            model_key=model_key,
            parameter_paths=parameter_paths,
        )
        token = _ACTIVE_TRANSACTION.set(transaction)
        try:
            with _KeyedParameterInitMode():
                yield
                expected_paths = {
                    path_key
                    for path_key, path in parameter_paths.items()
                    if path.initializes
                }
                missing_paths = expected_paths - transaction.completed_paths
                if missing_paths:
                    missing_fqns = sorted(
                        parameter_paths[path_key].fqn for path_key in missing_paths
                    )
                    raise RuntimeError(
                        "keyed parameter initialization did not run initializers "
                        f"for {missing_fqns}"
                    )
        finally:
            _ACTIVE_TRANSACTION.reset(token)
    except BaseException:
        rng.set_state(saved_state)
        raise
