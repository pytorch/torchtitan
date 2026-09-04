# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import contextlib
import inspect
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import spmd_types as spmd
import torch
import torch.nn as nn
from spmd_types import SpmdType
from torch.utils._pytree import tree_map

from torchtitan.config import Configurable
from torchtitan.distributed.parallel_dims import MeshAxisName, ParallelDims
from torchtitan.distributed.spmd_types import (
    _per_axis_types,
    set_current_spmd_mesh,
    spmd_axes,
    spmd_distribute_tensor,
)
from torchtitan.protocols.sharding import ShardingConfig


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
    _sharding_config: ShardingConfig | None = None
    _pos_arg_list: list[str] | None = None
    _parallelized: bool = False

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        param_init: dict | None = None
        sharding_config: ShardingConfig | None = None

        def build(self, **kwargs):
            # slots=True prevents super().build() from working; call explicitly.
            # Assignment is done here rather than in Module.__init__ because
            # there is no common Module.__init__ that all subclasses call.
            instance = Configurable.Config.build(self, **kwargs)
            if self.param_init is not None:
                instance._param_init = self.param_init
            if self.sharding_config is not None:
                instance._sharding_config = self.sharding_config
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

        with self._preserve_buffer_spmd_types():
            self._init_self_buffers(buffer_device=buffer_device)

    def _apply(self, fn, recurse=True):
        """Override to preserve annotations across model.to_empty() in trainer.py"""
        with self._preserve_buffer_spmd_types():
            return super()._apply(fn, recurse=recurse)

    @contextlib.contextmanager
    def _preserve_buffer_spmd_types(self) -> Iterator[None]:
        """
        Preserve SPMD type annotations on buffers across reinitialization.

        ``to_empty()`` and ``_init_self_buffers()`` re-materialize buffer data,
        clobbering over SPMD annotations. Instead of attempting to typecheck over
        this, we save-restore annotations on their respective mesh axes.
        """
        saved = {
            fqn: SpmdType(
                dict(spmd.get_local_type(buf)),
                spmd.get_partition_spec(buf),
            )
            for fqn, buf in self.named_buffers()
            if spmd.has_local_type(buf)
        }
        try:
            yield
        finally:
            for fqn, buf in self.named_buffers():
                if fqn in saved and not spmd.has_local_type(buf):
                    spmd.assert_type(buf, saved[fqn])

    def _init_self_parameters(self) -> None:
        """Initialize this module's own direct parameters.

        Resolution order:

        1. If ``param_init`` is set, use per-parameter dict lookup via
           ``_init_param``.
        2. Otherwise, fall back to ``reset_parameters()`` if it is
           available on ``self`` (typically inherited from the
           underlying ``nn`` class, but a subclass override is also
           honored). This is the standard PyTorch convention used by
           ``nn.Linear``, ``nn.LayerNorm``, ``nn.Conv2d``, etc.
        3. Otherwise, raise if there are any own parameters.
        """
        if self._param_init is not None:
            for name, param in self.named_parameters(recurse=False):
                self._init_param(name, param)
            return

        reset = getattr(self, "reset_parameters", None)
        if callable(reset):
            reset()
            return

        own_param_names = [name for name, _ in self.named_parameters(recurse=False)]
        if own_param_names:
            raise ValueError(
                f"{type(self).__name__} has parameters {own_param_names} "
                "but neither param_init nor reset_parameters is available. "
                "Set param_init on the Config or define reset_parameters."
            )

    def _init_param(self, name: str, param: nn.Parameter) -> None:
        """Initialize a single parameter via dict lookup in ``_param_init``.

        Raises ``ValueError`` if ``_param_init`` is None or the name is missing.
        """
        if self._param_init is None:
            raise ValueError(
                f"No param_init found for parameter {name!r} in "
                f"{type(self).__name__}. Set param_init on this "
                f"module's Config or use skip_param_init."
            )
        if name not in self._param_init:
            raise ValueError(
                f"No initializer for parameter {name!r} in "
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

    def _cache_pos_arg_names(self) -> list[str]:
        """Return positional arg names of ``forward`` (excluding ``self``), cached.

        Must be called once **before** ``forward`` is wrapped in ``parallelize``
        so ``inspect.signature`` sees the unwrapped signature. Subsequent
        calls return the cached list.
        """
        if self._pos_arg_list is not None:
            return self._pos_arg_list
        # pyrefly sees self.forward = ... in parallelize() and thinks forward
        # is instance-only, but it's always defined on nn.Module subclasses.
        sig = inspect.signature(
            type(self).forward  # pyrefly: ignore[missing-attribute]
        )
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

    def parallelize(self, parallel_dims: ParallelDims) -> None:
        """Parallelize this module and all Module children recursively.

        For each module with a ``sharding_config``:

        1. Shard states (parameters and buffers).
        2. Wrap forward with boundary type checks and an optional local region.

        ``fully_shard`` hooks on ``__call__`` fire around the wrapped ``forward``.

        Each ``ShardingConfig`` field resolves its mesh independently via
        ``resolve_mesh()``.
        """
        if self._parallelized:
            raise ValueError(
                f"{type(self).__name__} has already been parallelized. "
                "Module.parallelize() must be called at most once per instance."
            )
        self._parallelized = True

        queue = list(self.children())
        while queue:
            child = queue.pop()
            if isinstance(child, Module):
                child.parallelize(parallel_dims)
            else:
                # Look through non-Module wrappers, e.g., CheckpointWrapper.
                queue.extend(child.children())

        # TODO(fegin): Change to assert once ALL Models are migrated to use _sharding_config.
        if self._sharding_config is None:
            return

        self._distribute_states(parallel_dims)
        self._cache_pos_arg_names()
        fn = self._maybe_wrap_with_local_region(self.forward)

        def forward_with_typechecking(*args, **kwargs):
            self._check_input_shardings(args, kwargs)
            outputs = fn(*args, **kwargs)
            self._check_output_shardings(outputs)
            return outputs

        self.forward = forward_with_typechecking

    def _check_input_shardings(self, args: tuple, kwargs: dict) -> None:
        """Check configured forward-input contracts when typechecking."""
        if not spmd.is_type_checking():
            return
        sharding_config = self._sharding_config
        assert sharding_config is not None
        in_shardings = sharding_config.in_shardings or {}
        pos_arg_names = [
            name for name in self._cache_pos_arg_names() if name not in kwargs
        ]
        named_args = dict(zip(pos_arg_names, args, strict=False))
        named_args.update(kwargs)
        for name, layout in in_shardings.items():
            value = named_args.get(name)
            if isinstance(value, torch.Tensor):
                spmd.assert_type(value, layout)

    def _check_output_shardings(self, outputs: Any) -> None:
        """Check configured forward-output contracts when typechecking."""
        if not spmd.is_type_checking():
            return
        sharding_config = self._sharding_config
        assert sharding_config is not None
        out_shardings = sharding_config.out_shardings
        if out_shardings is None:
            return

        def assert_leaf(output, layout):
            if layout is not None:
                spmd.assert_type(output, layout)
            return output

        tree_map(
            assert_leaf,
            outputs,
            out_shardings,
            is_leaf=lambda x: isinstance(x, SpmdType),
        )

    def _spmd_distribute_state(
        self,
        parallel_dims: ParallelDims,
        name: str,
        tensor: torch.Tensor,
        layout: SpmdType,
        *,
        is_param: bool,
    ) -> None:
        # Call get_optional_mesh with include_singleton_axes=True, so we're able to call assert_type()
        # using all axes, and defer size-1 axis filtering to spmd_types internals.
        mesh = parallel_dims.get_optional_mesh(
            [axis.value for axis in spmd_axes(layout)],
            include_singleton_axes=True,
        )
        assert mesh is not None
        assert mesh.mesh_dim_names is not None, "DeviceMesh must have named axes"

        requires_grad = tensor.requires_grad
        tensor = spmd_distribute_tensor(tensor, mesh, layout)
        if is_param:
            self.register_parameter(
                name, nn.Parameter(tensor, requires_grad=requires_grad)
            )
            registered = self._parameters[name]
        else:
            persistent = name not in self._non_persistent_buffers_set
            self.register_buffer(name, tensor, persistent=persistent)
            registered = self._buffers[name]

        # assert_type resolves SpmdType's string mesh axis names to concrete
        # runtime mesh-axis objects, so a mesh context is required here.
        with set_current_spmd_mesh(mesh):
            spmd.assert_type(registered, layout)

    def _validate_even_model_parallel_param_sharding(
        self,
        name: str,
        param: nn.Parameter,
        layout: SpmdType,
        parallel_dims: ParallelDims,
    ) -> None:
        """Reject parameter layouts that produce uneven TP or EP local shards."""
        axis_types = _per_axis_types(layout)
        axis_sizes = {
            MeshAxisName.TP: parallel_dims.tp,
            MeshAxisName.EP: parallel_dims.ep,
        }
        for axis_name, axis_size in axis_sizes.items():
            axis_type = axis_types.get(axis_name)
            if axis_size == 1 or not isinstance(axis_type, spmd.Shard):
                continue

            tensor_dim = axis_type.dim
            if tensor_dim < 0:
                tensor_dim += param.ndim
            if tensor_dim < 0 or tensor_dim >= param.ndim:
                raise ValueError(
                    f"{type(self).__name__}.{name} has invalid tensor dimension "
                    f"{axis_type.dim} in its {axis_name.value.upper()} sharding "
                    f"for parameter shape {tuple(param.shape)}."
                )
            if param.shape[tensor_dim] % axis_size == 0:
                continue

            raise ValueError(
                "spmd_types does not support uneven model-parallel parameter "
                f"sharding: {type(self).__name__}.{name} with shape "
                f"{tuple(param.shape)} "
                f"cannot be evenly sharded on tensor dimension {tensor_dim} "
                f"across model-parallel mesh axis {axis_name.value} with size "
                f"{axis_size}."
            )

    def _distribute_states(self, parallel_dims: ParallelDims) -> None:
        """Distribute params and buffers per ``state_shardings``.

        Each entry resolves its own mesh via ``resolve_mesh``, so different
        params on the same Module may live on different meshes.
        """
        sharding_config = self._sharding_config
        assert sharding_config is not None

        for name, param in self.named_parameters(recurse=False):
            spmd_layout = sharding_config.state_shardings.get(name)
            if spmd_layout is None:
                raise ValueError(
                    f"{type(self).__name__}.{name} has no placement declared "
                    "in sharding_config.state_shardings."
                )
            self._validate_even_model_parallel_param_sharding(
                name,
                param,
                spmd_layout,
                parallel_dims,
            )
            self._spmd_distribute_state(
                parallel_dims,
                name,
                param,
                spmd_layout,
                is_param=True,
            )

        for name, buffer in self.named_buffers(recurse=False):
            spmd_layout = sharding_config.state_shardings.get(name)
            if spmd_layout is None:
                raise ValueError(
                    f"{type(self).__name__}.{name} (buffer) has no placement "
                    "declared in sharding_config.state_shardings."
                )
            if buffer is None:
                # ``register_buffer(name, None)`` reserves a slot to be filled
                # by ``init_states`` later; nothing to distribute yet.
                continue
            self._spmd_distribute_state(
                parallel_dims,
                name,
                buffer,
                spmd_layout,
                is_param=False,
            )

    def _maybe_wrap_with_local_region(
        self,
        fn: Callable,
    ) -> Callable:
        """Wrap ``fn`` with a local-tensor region if configured.

        Input and output layouts come from the module boundary contracts.
        """
        sharding_config = self._sharding_config
        assert sharding_config is not None
        if not sharding_config.local_spmd:
            return fn

        in_shardings = sharding_config.in_shardings or {}
        pos_args = self._cache_pos_arg_names()
        out_shardings = sharding_config.out_shardings
        if out_shardings is None:
            raise AssertionError(
                f"{type(self).__name__}: local_spmd is set but "
                "out_shardings is None."
            )
        missing_in = [name for name in pos_args if name not in in_shardings]
        if missing_in:
            raise AssertionError(
                f"{type(self).__name__}: local_spmd is set but in_shardings "
                f"is missing entries for: {missing_in}"
            )
        in_named: list[SpmdType] = [in_shardings[name] for name in pos_args]

        return self._spmd_apply_local_region(fn, in_named, out_shardings)

    def _spmd_apply_local_region(
        self,
        fn: Callable,
        in_named: list[SpmdType],
        out_shardings: SpmdType | tuple[SpmdType | None, ...],
    ) -> Callable:
        """Apply spmd_types local_map for a local-tensor compute region."""
        in_types = tuple(
            (layout.local_type, layout.partition_spec) for layout in in_named
        )
        out_types = tree_map(
            lambda layout: (layout.local_type, layout.partition_spec),
            out_shardings,
            is_leaf=lambda x: isinstance(x, SpmdType),
        )
        return spmd.no_typecheck(
            in_types=in_types,
            out_types=out_types,
        )(fn)


class ModuleList(nn.ModuleList, Module):
    """Module-protocol-compatible version of ``nn.ModuleList``."""

    pass


class ModuleDict(nn.ModuleDict, Module):
    """Module-protocol-compatible version of ``nn.ModuleDict``."""

    pass


class Sequential(nn.Sequential, Module):
    """Module-protocol-compatible version of ``nn.Sequential``."""

    pass
