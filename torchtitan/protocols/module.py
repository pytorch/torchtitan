# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.distributed.tensor import distribute_tensor, DTensor

from torchtitan.config import Configurable
from torchtitan.protocols.sharding import (
    resolve_placements,
    ShardingSpec,
    Unconstrained,
)


# Cache: maps nn.Module subclass -> created Module wrapper class.
# Module classes are typically created at import time and live for
# the process lifetime.
_created_classes: dict[type, type] = {}


class Module(nn.Module, Configurable):
    """Base class for all configurable nn.Module components.
    Combines nn.Module with Configurable, so subclasses only inherit from Module.

    ``init_states`` auto-recurses into children, then initializes the current
    module's parameters (via ``_param_init`` dict lookup) and buffers.
    Subclasses should NOT override ``init_states`` unless they need custom
    ordering (e.g., weight tying before init). Override ``_init_self_buffers``
    for buffer initialization.
    """

    _param_init: dict[str, Callable] | None = None
    _sharding_spec: ShardingSpec | None = None
    _pos_arg_list: list[str] | None = None

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        param_init: dict | None = None
        sharding_spec: ShardingSpec | None = None

        def build(self, **kwargs):
            # slots=True prevents super().build() from working; call explicitly.
            # Assignment is done here rather than in Module.__init__ because
            # there is no common Module.__init__ that all subclasses call.
            instance = Configurable.Config.build(self, **kwargs)
            if self.param_init is not None:
                instance._param_init = self.param_init
            if self.sharding_spec is not None:
                instance._sharding_spec = self.sharding_spec
            return instance

    def init_states(
        self,
        *,
        buffer_device: torch.device | None = None,
    ) -> None:
        """Initialize all states in the module tree.

        1. Recursively calls ``init_states`` on all direct Module children.
        2. Calls ``self._init_self_parameters()``.
        3. Calls ``self._init_self_buffers(...)``.

        Args:
            buffer_device: Device for buffer initialization (e.g., RoPE, MoE).
        """

        queue = list(self.children())
        while queue:
            child = queue.pop(0)
            if isinstance(child, Module):
                child.init_states(buffer_device=buffer_device)
            else:
                # Plain nn.Module (e.g., CheckpointWrapper, torch.compile
                # wrappers) — look inside for Module descendants.
                queue.extend(child.children())

        self._init_self_parameters()
        self._init_self_buffers(buffer_device=buffer_device)

    def _init_self_parameters(self) -> None:
        """Initialize this module's own parameters via ``_init_param``.

        Overridden internally by ``from_nn_module`` to delegate to
        ``reset_parameters``. Not intended for subclass override — configure
        parameter initialization via ``param_init`` on the Config instead.
        """
        for name, param in self.named_parameters(recurse=False):
            self._init_param(name, param)

    def _init_param(self, name: str, param: nn.Parameter) -> None:
        """Initialize a single parameter via dict lookup in ``_param_init``.

        Raises ``ValueError`` if ``_param_init`` is None or the name is missing.
        """
        if self._param_init is None:
            raise ValueError(
                f"No param_init found for parameter '{name}' in "
                f"{type(self).__name__}. Set param_init on this "
                f"module's Config or use skip_param_init."
            )
        if name not in self._param_init:
            raise ValueError(
                f"No initializer for parameter '{name}' in "
                f"{type(self).__name__}. "
                f"Available: {list(self._param_init.keys())}"
            )
        self._param_init[name](param)

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        """Initialize this module's own buffers.

        The default is a no-op. Override for device-aware buffer
        initialization (e.g., RoPE cache, MoE counters).

        Args:
            buffer_device: Target device for buffer creation/initialization.
        """
        pass

    @property
    def _pos_arg_names(self) -> list[str]:
        """Positional arg names of ``forward`` (excluding ``self``).

        Computed once from the class-level ``forward`` signature and cached.
        Must be accessed **before** ``forward`` is wrapped (i.e., in
        ``parallelize``), so the introspection sees the original signature.
        """
        if self._pos_arg_list is not None:
            return self._pos_arg_list
        sig = inspect.signature(type(self).forward)
        self._pos_arg_list = [
            p.name
            for p in sig.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            and p.name != "self"
        ]
        return self._pos_arg_list

    def parallelize(self, mesh: "torch.distributed.DeviceMesh") -> None:
        """Parallelize this module and all Module children recursively.

        For each module with a ``sharding_spec``:

        1. ``distribute_tensor`` on params per ``state_shardings``.
        2. Wrap ``self.forward`` with redistribution (+ ``local_map`` if needed).

        The wrapping order is: ``shard_inputs → [local_map →] fn → shard_outputs``.
        CP (applied before ``parallelize``) is captured inside ``local_map``.
        FSDP hooks on ``__call__`` fire around the wrapped ``forward``.
        """
        # Recurse children first (bottom-up, like sixlib)
        queue = list(self.children())
        while queue:
            child = queue.pop()
            if isinstance(child, Module):
                child.parallelize(mesh)
            else:
                # Look through non-Module wrappers (CheckpointWrapper, compile)
                queue.extend(child.children())

        spec = self._sharding_spec
        if spec is None:
            return

        assert mesh.mesh_dim_names is not None, "DeviceMesh must have named dims"
        mesh_dim_names = mesh.mesh_dim_names

        # 1. Distribute parameters
        for name, param in self.named_parameters(recurse=False):
            if name in spec.state_shardings:
                placements = resolve_placements(
                    spec.state_shardings[name], mesh_dim_names
                )
                self.register_parameter(
                    name,
                    nn.Parameter(distribute_tensor(param, mesh, list(placements))),
                )

        # Pre-cache positional arg names before wrapping forward, so
        # inspect.signature sees the original (or CP-wrapped) signature.
        # _shard_inputs uses the cached list instead of calling inspect every forward.
        _ = self._pos_arg_names  # noqa: F841

        # 2. Wrap forward with redistribution (+ local_map if needed)
        fn = self.forward  # capture current forward (may already be CP-wrapped)

        if spec.local_map is not None:
            from torch.distributed.tensor.experimental import local_map

            fn = local_map(
                fn,
                in_placements=spec.local_map.in_placements,
                out_placements=spec.local_map.out_placements,
                in_grad_placements=spec.local_map.in_grad_placements,
                device_mesh=mesh,
            )

        # Always wrap with redistribution
        captured_fn = fn
        captured_spec = spec
        captured_mesh = mesh
        captured_mod = self

        def with_redistribution(*args, **kwargs):
            args, kwargs = _shard_inputs(
                captured_mod, captured_mesh, captured_spec, args, kwargs
            )
            outputs = captured_fn(*args, **kwargs)
            return _shard_outputs(captured_mesh, captured_spec, outputs)

        self.forward = with_redistribution  # pyrefly: ignore [missing-attribute]

    @classmethod
    def from_nn_module(cls, nn_module_cls: type[nn.Module]) -> type["Module"]:
        """Create a ``Module``-protocol-compatible version of *nn_module_cls*.

        The returned class inherits from ``(nn_module_cls, Module)`` and has the
        same constructor signature as *nn_module_cls*.

        * If *nn_module_cls* defines ``reset_parameters``, the injected
          ``_init_self_parameters`` delegates to it.
        * Otherwise ``_init_self_parameters`` is the inherited default from
          ``Module``.

        Results are cached so that repeated calls with the same class return
        the identical class object.

        Usage::

            Conv2d = Module.from_nn_module(nn.Conv2d)
            LayerNorm = Module.from_nn_module(nn.LayerNorm)
            # Then use Conv2d / LayerNorm exactly like nn.Conv2d / nn.LayerNorm
        """
        if nn_module_cls in _created_classes:
            return _created_classes[nn_module_cls]

        attrs: dict[str, Any] = {}
        if hasattr(nn_module_cls, "reset_parameters"):

            def _init_self_parameters(self: Any) -> None:
                self.reset_parameters()

            attrs["_init_self_parameters"] = _init_self_parameters

        name = f"Module({nn_module_cls.__name__})"
        new_cls = type(name, (nn_module_cls, Module), attrs)
        new_cls.__module__ = __name__
        new_cls.__qualname__ = name
        _created_classes[nn_module_cls] = new_cls
        return new_cls


class ModuleList(nn.ModuleList, Module):
    """Module-protocol-compatible version of ``nn.ModuleList``."""

    pass


class ModuleDict(nn.ModuleDict, Module):
    """Module-protocol-compatible version of ``nn.ModuleDict``."""

    pass


class Sequential(nn.Sequential, Module):
    """Module-protocol-compatible version of ``nn.Sequential``."""

    pass


# ---------------------------------------------------------------------------
# Sharding helpers for Module.parallelize()
# ---------------------------------------------------------------------------


def _shard_inputs(
    mod: Module,
    mesh: "torch.distributed.DeviceMesh",
    spec: ShardingSpec,
    args: tuple,
    kwargs: dict,
) -> tuple[tuple, dict]:
    """Redistribute inputs (both positional and keyword) to desired placements.

    Merges positional args into a unified kwargs dict using the cached
    ``_pos_arg_names``, processes all inputs by name, then splits back
    into (args, kwargs) for the downstream forward call.

    Two-step process per input:
    1. If plain tensor, wrap as DTensor using ``input_layouts`` (annotation).
    2. If DTensor placements != ``in_shardings``, redistribute.
    """
    if spec.in_shardings is None and spec.input_layouts is None:
        return args, kwargs

    # Use pre-cached positional arg names (populated in parallelize()).
    pos_arg_names = [name for name in mod._pos_arg_names if name not in kwargs]

    # Merge positional args into a unified kwargs dict.
    new_kwargs = dict(zip(pos_arg_names, args))
    new_kwargs.update(kwargs)

    assert mesh.mesh_dim_names is not None
    mesh_dim_names = mesh.mesh_dim_names
    in_shardings = spec.in_shardings or {}
    input_layouts = spec.input_layouts or {}

    for name, value in new_kwargs.items():
        if not isinstance(value, torch.Tensor):
            continue

        # Step 1: Annotate plain tensor as DTensor using input_layouts
        if not isinstance(value, DTensor) and name in input_layouts:
            layout = resolve_placements(input_layouts[name], mesh_dim_names)
            value = DTensor.from_local(value, mesh, layout, run_check=False)

        # Step 2: Redistribute to desired placement if needed
        if name in in_shardings and isinstance(value, DTensor):
            desired = resolve_placements(in_shardings[name], mesh_dim_names)
            if any(isinstance(p, Unconstrained) for p in desired):
                continue
            if value.placements != desired:
                value = value.redistribute(placements=desired, async_op=True)

        new_kwargs[name] = value

    # Split back: pop positional args in order, remainder stays as kwargs.
    new_args = tuple(new_kwargs.pop(name) for name in pos_arg_names)
    return new_args, new_kwargs


def _shard_outputs(
    mesh: "torch.distributed.DeviceMesh",
    spec: ShardingSpec,
    outputs: Any,
) -> Any:
    """Redistribute output to desired placement."""
    if spec.out_shardings is None:
        return outputs
    assert mesh.mesh_dim_names is not None
    desired = resolve_placements(spec.out_shardings, mesh.mesh_dim_names)
    if any(isinstance(p, Unconstrained) for p in desired):
        return outputs
    if isinstance(outputs, DTensor) and outputs.placements != desired:
        outputs = outputs.redistribute(placements=desired, async_op=True)
    return outputs
