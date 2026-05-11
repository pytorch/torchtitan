# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import inspect
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.distributed.tensor import DeviceMesh, distribute_tensor, DTensor
from torch.distributed.tensor.experimental import local_map

import spmd_types as spmd
from torchtitan.config import Configurable
from torchtitan.distributed.spmd_state import is_spmd_active, mesh as spmd_mesh, spmd_state
from torchtitan.protocols.sharding import (
    LocalMapConfig,
    resolve_placements,
    ShardingConfig,
)

if TYPE_CHECKING:
    from torchtitan.distributed.parallel_dims import ParallelDims


@contextmanager
def preserve_buffer_spmd(model: nn.Module):
    """
    Init time:
    1. shard & apply spmd_types annotations on meta tensor state.
    2. apply FSDP, which saves-restores spmd_types on parameters.
    3. Initialize weights w/ ``to_empty()`` + ``init_weights()``, which loses annotations.

    FSDP restores annotations for params; use this around 3. for buffer spmd_types.
    """
    saved = {}
    for fqn, buf in model.named_buffers():
        if spmd.has_local_type(buf):
            saved[fqn] = dict(spmd.get_local_type(buf))
    yield
    for fqn, buf in model.named_buffers():
        if fqn in saved and not spmd.has_local_type(buf):
            spmd.assert_type(buf, saved[fqn])


def _resolve_named(named):
    """Resolve MeshAxisName → MeshAxis, dropping inactive axes."""
    m = spmd_mesh()
    types = {}
    for axis_name, value in named.items():
        if (ax := getattr(m, axis_name.value)) is None:
            continue
        types[ax] = value
    return types


def named_placement_to_spmd(named, ndim=None):
    """
    Resolve NamedPlacement to {MeshAxis: spmd_type},
    with optional S(*)->V+PartitionSpec normalization for ``spmd.assert_type`` usage.

    When ``ndim`` is None, returns raw types without normalization —
    used by redistribute callers that compare per-axis types directly.

    When ``ndim`` is provided and multiple active axes shard the same
    tensor dim, S(dim) entries are converted to V and a PartitionSpec
    is built using canonical axis ordering.

    Returns ``(types, partition_spec)``.
    """
    types = _resolve_named(named)

    if ndim is None:
        return types, None

    # Check for multi-axis-same-dim collisions
    dim_to_axes: dict[int, list] = {}
    for ax, t in types.items():
        if isinstance(t, spmd.Shard):
            dim_to_axes.setdefault(t.dim, []).append(ax)

    has_collision = any(len(axes) > 1 for axes in dim_to_axes.values())
    if not has_collision:
        return types, None

    # Build PartitionSpec using canonical axis ordering for collisions.
    axis_order = spmd_state().axis_order
    order_idx = {ax: i for i, ax in enumerate(axis_order)}

    pspec_entries: dict[int, object] = {}
    for dim, axes in dim_to_axes.items():
        if len(axes) > 1:
            sorted_axes = tuple(sorted(axes, key=lambda a: order_idx.get(a, 0)))
            pspec_entries[dim] = sorted_axes
            for ax in axes:
                types[ax] = spmd.V
        else:
            pspec_entries[dim] = axes[0]
            types[axes[0]] = spmd.V

    spec_args = [pspec_entries.get(d) for d in range(ndim)]
    partition_spec = spmd.PartitionSpec(*spec_args)

    return types, partition_spec


def redistribute_spmd_per_axis(
    x: torch.Tensor,
    src_types: spmd.PerMeshAxisSpmdTypes,
    dst_types: spmd.PerMeshAxisSpmdTypes,
) -> torch.Tensor:
    """Redistribute a tensor per-axis where src != dst."""
    state = spmd_state()
    for axis, dst_t in dst_types.items():
        src_t = src_types.get(axis)
        if src_t is not None and src_t != dst_t:
            pg = state.pg_for_axis(axis)
            # bwd = {"op_dtype": torch.float32} if x.dtype != torch.float32 else None
            # x = spmd.redistribute(x, pg, src=src_t, dst=dst_t, backward_options=bwd)
            x = spmd.redistribute(x, pg, src=src_t, dst=dst_t)
    return x


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
    _sharding_config: ShardingConfig | None = None
    _pos_arg_list: list[str] | None = None

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

        # _init_self_buffers often re-assigns (e.g. ``self.cache = self._precompute()``),
        # replacing any DTensor entry with a plain tensor. Restore DTensor-ness
        # by re-distributing the freshly computed global tensor to the original
        # placements. ``distribute_tensor`` supports Replicate / Shard / Partial.
        dtensor_meta = {
            name: (buf.device_mesh, buf.placements)
            for name, buf in self._buffers.items()
            if isinstance(buf, DTensor)
        }
        self._init_self_buffers(buffer_device=buffer_device)
        for name, (mesh, placements) in dtensor_meta.items():
            new_buf = self._buffers.get(name)
            if new_buf is None or isinstance(new_buf, DTensor):
                continue
            persistent = name not in self._non_persistent_buffers_set
            self.register_buffer(
                name,
                distribute_tensor(new_buf, mesh, list(placements)),
                persistent=persistent,
            )

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

    # ------------------------------------------------------------------
    # parallelize: distribute states, wrap local boundary, wrap forward
    # ------------------------------------------------------------------

    def parallelize(
        self, mesh_or_parallel_dims: DeviceMesh | ParallelDims
    ) -> None:
        """Parallelize this module and all Module children recursively.

        Accepts either a ``DeviceMesh`` (DTensor path) or ``ParallelDims``
        (SPMD path, when ``full_spmd_types=True``).

        For each module with a ``sharding_config``:

        1. Distribute params and buffers per ``state_shardings``.
        2. Wrap ``self.forward`` with local_map boundary + redistribution.

        fully_shard hooks on ``__call__`` fire around the wrapped ``forward``.
        CP (applied before ``parallelize``) is captured inside ``local_map``.
        """
        # Recurse children first
        queue = list(self.children())
        while queue:
            child = queue.pop()
            if isinstance(child, Module):
                child.parallelize(mesh_or_parallel_dims)
            else:
                # Look through non-Module wrappers, e.g., CheckpointWrapper
                queue.extend(child.children())

        sharding_config = self._sharding_config
        if sharding_config is None:
            return

        # Step 1: Distribute parameters and buffers
        for name, param in self.named_parameters(recurse=False):
            if name not in sharding_config.state_shardings:
                continue
            self.distribute_state(
                name,
                param,
                sharding_config.state_shardings[name],
                mesh_or_parallel_dims,
                is_param=True,
            )

        for name, buffer in self.named_buffers(recurse=False):
            if name not in sharding_config.state_shardings or buffer is None:
                continue
            self.distribute_state(
                name,
                buffer,
                sharding_config.state_shardings[name],
                mesh_or_parallel_dims,
                is_param=False,
            )

        # Cache positional arg names before wrapping forward
        self._cache_pos_arg_names()

        # Step 2: Wrap forward with local_map boundary
        fn = self.forward
        if sharding_config.local_map is not None:
            fn = self.local_map(
                fn, sharding_config.local_map, mesh_or_parallel_dims
            )

        # Step 3: Wrap forward with input/output redistribution
        def with_redistribution(*args, **kwargs):
            args, kwargs = self._shard_inputs(mesh_or_parallel_dims, args, kwargs)
            outputs = fn(*args, **kwargs)
            return self._shard_outputs(mesh_or_parallel_dims, outputs)

        self.forward = with_redistribution

    def distribute_state(
        self,
        name: str,
        tensor: torch.Tensor,
        named_placement: dict,
        mesh_or_pd: DeviceMesh | ParallelDims,
        *,
        is_param: bool,
    ) -> None:
        """Distribute a single parameter or buffer.

        DTensor values → ``distribute_tensor``.
        SPMD values → physical TP shard (if applicable) + ``spmd.assert_type``.
        """
        if is_spmd_active():
            self.distribute_state_spmd(name, tensor, named_placement, is_param=is_param)
        else:
            mesh = (
                mesh_or_pd
                if isinstance(mesh_or_pd, DeviceMesh)
                else mesh_or_pd.world_mesh
            )
            assert mesh.mesh_dim_names is not None
            placements = resolve_placements(named_placement, mesh.mesh_dim_names)
            distributed = distribute_tensor(tensor, mesh, list(placements))
            if is_param:
                self.register_parameter(name, nn.Parameter(distributed))
            else:
                persistent = name not in self._non_persistent_buffers_set
                self.register_buffer(name, distributed, persistent=persistent)

    def distribute_state_spmd(
        self,
        name: str,
        tensor: torch.Tensor,
        named_placement: dict,
        *,
        is_param: bool,
    ) -> None:
        """SPMD path: physically shard where needed, then annotate."""
        m = spmd_mesh()
        state = spmd_state()

        # Validate and collect shards. Values must be raw per-axis types.
        shard_dims: dict[int, str] = {}
        for axis_name, value in named_placement.items():
            assert isinstance(
                value, (spmd.Shard, spmd.PerMeshAxisLocalSpmdType)
            ), (
                f"Expected per-axis spmd type for state {name!r} on axis "
                f"{axis_name.value!r}, got {type(value).__name__}: {value!r}"
            )
            if value is spmd.P:
                raise ValueError(
                    f"Partial is not valid for state {name!r} "
                    f"on axis {axis_name.value!r}."
                )
            if isinstance(value, spmd.Shard):
                assert value.dim not in shard_dims, (
                    f"State {name!r}: axes {shard_dims[value.dim]!r} and "
                    f"{axis_name.value!r} both shard dim {value.dim}. "
                    f"Multi-axis sharding of a single state dim is not yet supported."
                )
                shard_dims[value.dim] = axis_name.value

        for axis_name, value in named_placement.items():
            if (ax := getattr(m, axis_name.value)) is None:
                continue
            if isinstance(value, spmd.Shard):
                pg = state.pg_for_axis(ax)
                assert pg is not None
                tensor = spmd.shard(tensor, pg, src=spmd.I, dst=value)

        # register state
        if is_param:
            self.register_parameter(name, nn.Parameter(tensor))
        else:
            persistent = name not in self._non_persistent_buffers_set
            self.register_buffer(name, tensor, persistent=persistent)

        # annotate the registered param/buffer (not the input tensor)
        registered = self._parameters[name] if is_param else self._buffers[name]
        types, partition_spec = named_placement_to_spmd(
            named_placement, ndim=registered.ndim
        )
        if types:
            spmd.assert_type(registered, types, partition_spec=partition_spec)

    def local_map(
        self,
        fn: Callable,
        local_map_config: LocalMapConfig,
        mesh_or_pd: DeviceMesh | ParallelDims,
    ) -> Callable:
        """Wrap forward with local_map (DTensor) or spmd.local_map (SPMD).

        Duck-types on placement values in the config to dispatch.
        """
        if is_spmd_active():
            return self.local_map_spmd(fn, local_map_config)
        else:
            mesh = (
                mesh_or_pd
                if isinstance(mesh_or_pd, DeviceMesh)
                else mesh_or_pd.world_mesh
            )
            return self.local_map_dtensor(fn, local_map_config, mesh)

    def local_map_dtensor(
        self, fn: Callable, lm: LocalMapConfig, mesh: DeviceMesh
    ) -> Callable:
        assert mesh.mesh_dim_names is not None
        mesh_axis_names = mesh.mesh_dim_names
        in_placements = tuple(
            resolve_placements(p, mesh_axis_names) for p in lm.in_placements
        )
        out_placements = tuple(
            resolve_placements(p, mesh_axis_names) for p in lm.out_placements
        )
        in_grad_placements = tuple(
            resolve_placements(p, mesh_axis_names) for p in lm.in_grad_placements
        ) if lm.in_grad_placements is not None else in_placements
        return local_map(
            fn,
            in_placements=in_placements,
            out_placements=out_placements,
            in_grad_placements=in_grad_placements,
            device_mesh=mesh,
        )

    def local_map_spmd(self, fn: Callable, lm: LocalMapConfig) -> Callable:
        assert lm.in_placements is not None, (
            "SPMD local_map requires explicit in_placements"
        )
        in_ndims = lm.in_ndims or (None,) * len(lm.in_placements)
        resolved_in = tuple(
            None if p is None else named_placement_to_spmd(p, ndim=nd)
            for p, nd in zip(lm.in_placements, in_ndims)
        )
        assert len(lm.out_placements) == 1, "spmd.local_map accepts 1 output"
        outp = lm.out_placements[0]
        out_nd = lm.out_ndims[0] if lm.out_ndims else None
        resolved_out = None if outp is None else named_placement_to_spmd(outp, ndim=out_nd)

        @spmd.local_map(in_types=resolved_in, out_types=resolved_out)
        def body(*args, **kwargs):
            return fn(*args, **kwargs)

        return body

    def _shard_inputs(
        self,
        mesh_or_pd: DeviceMesh | ParallelDims,
        args: tuple,
        kwargs: dict,
    ) -> tuple[tuple, dict]:
        """Redistribute inputs to desired placements."""
        sharding_config = self._sharding_config
        assert sharding_config is not None

        if (
            sharding_config.in_dst_shardings is None
            and sharding_config.in_src_shardings is None
        ):
            return args, kwargs

        # Use pre-cached positional arg names (populated in parallelize()) to
        # merge positional args into a unified kwargs dict.
        pos_arg_names = [
            name for name in self._cache_pos_arg_names() if name not in kwargs
        ]
        new_kwargs = dict(zip(pos_arg_names, args))
        new_kwargs.update(kwargs)

        in_dst_shardings = sharding_config.in_dst_shardings or {}
        in_src_shardings = sharding_config.in_src_shardings or {}

        if is_spmd_active():
            self._shard_inputs_spmd(new_kwargs, in_src_shardings, in_dst_shardings)
        else:
            mesh = (
                mesh_or_pd
                if isinstance(mesh_or_pd, DeviceMesh)
                else mesh_or_pd.world_mesh
            )
            self._shard_inputs_dtensor(new_kwargs, in_src_shardings, in_dst_shardings, mesh)

        new_args = tuple(new_kwargs.pop(name) for name in pos_arg_names)
        return new_args, new_kwargs

    def _shard_inputs_dtensor(
        self,
        kwargs: dict,
        in_src_shardings: dict,
        in_dst_shardings: dict,
        mesh: DeviceMesh,
    ) -> None:
        assert mesh.mesh_dim_names is not None
        mesh_axis_names = mesh.mesh_dim_names

        for name, value in kwargs.items():
            if not isinstance(value, torch.Tensor):
                continue

            # Step 1: Annotate plain tensor as DTensor
            if not isinstance(value, DTensor) and name in in_src_shardings:
                layout = resolve_placements(in_src_shardings[name], mesh_axis_names)
                value = DTensor.from_local(value, mesh, layout, run_check=False)

            # Step 2: Redistribute to desired placement
            if name in in_dst_shardings and isinstance(value, DTensor):
                desired = resolve_placements(in_dst_shardings[name], mesh_axis_names)
                if value.placements != desired:
                    value = value.redistribute(placements=desired, async_op=True)

            kwargs[name] = value

    def _shard_inputs_spmd(
        self,
        kwargs: dict,
        in_src_shardings: dict,
        in_dst_shardings: dict,
    ) -> None:
        for name, value in kwargs.items():
            if not isinstance(value, torch.Tensor):
                continue
            if name in in_src_shardings and name in in_dst_shardings:
                src_types, _ = named_placement_to_spmd(in_src_shardings[name])
                dst_types, _ = named_placement_to_spmd(in_dst_shardings[name])
                value = redistribute_spmd_per_axis(value, src_types, dst_types)
            kwargs[name] = value

    def _shard_outputs(
        self,
        mesh_or_pd: DeviceMesh | ParallelDims,
        outputs: Any,
    ) -> Any:
        """Redistribute output to desired placement."""
        sharding_config = self._sharding_config
        assert sharding_config is not None

        if sharding_config.out_dst_shardings is None:
            return outputs

        if is_spmd_active():
            return self._shard_outputs_spmd(outputs)
        else:
            mesh = (
                mesh_or_pd
                if isinstance(mesh_or_pd, DeviceMesh)
                else mesh_or_pd.world_mesh
            )
            return self._shard_outputs_dtensor(outputs, mesh)

    def _shard_outputs_dtensor(
        self, outputs: Any, mesh: DeviceMesh
    ) -> Any:
        sharding_config = self._sharding_config
        assert sharding_config is not None
        assert mesh.mesh_dim_names is not None
        desired = resolve_placements(
            sharding_config.out_dst_shardings, mesh.mesh_dim_names
        )
        if isinstance(outputs, DTensor) and outputs.placements != desired:
            outputs = outputs.redistribute(placements=desired, async_op=True)
        return outputs

    def _shard_outputs_spmd(self, outputs: Any) -> Any:
        sharding_config = self._sharding_config
        assert sharding_config is not None
        if isinstance(outputs, torch.Tensor):
            out_src, out_dst = sharding_config.out_src_shardings, sharding_config.out_dst_shardings
            if out_src is None:
                out_src = out_dst
            src_types, _ = named_placement_to_spmd(out_src)
            dst_types, _ = named_placement_to_spmd(out_dst)
            outputs = redistribute_spmd_per_axis(outputs, src_types, dst_types)
        return outputs

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
