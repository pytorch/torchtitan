# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import distribute_tensor, DTensor

import spmd_types as spmd
from spmd_types.types import shard_types_to_partition_spec
from torchtitan.config import Configurable
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.spmd_state import (
    current_mesh,
    is_spmd_active,
    set_current_mesh,
)
from torchtitan.protocols.sharding import (
    NamedPlacement,
    NamedPartitionSpec,
    PlacementLike,
    PlacementSpec,
    ShardingConfig,
)
from torchtitan.protocols.types import MeshAxisName


@contextmanager
def preserve_buffer_spmd(model: nn.Module) -> Iterator[None]:
    """
    Init time:
    1. shard & apply spmd_types annotations on meta tensor state.
    2. apply FSDP, which saves-restores spmd_types on parameters.
    3. Initialize weights w/ ``to_empty()`` + ``init_weights()``, which loses annotations.

    FSDP restores annotations for params; use this around 3. for buffer spmd_types.
    """
    saved: dict[str, spmd.LocalSpmdType] = {}
    for fqn, buf in model.named_buffers():
        if spmd.has_local_type(buf):
            saved[fqn] = dict(spmd.get_local_type(buf))
    yield
    for fqn, buf in model.named_buffers():
        if fqn in saved and not spmd.has_local_type(buf):
            spmd.assert_type(buf, saved[fqn])


def _active_axis_names(
    axis_name: MeshAxisName,
    mesh_names: dict,
) -> tuple[MeshAxisName, ...]:
    if axis_name == MeshAxisName.DP and MeshAxisName.DP not in mesh_names:
        return tuple(
            axis
            for axis in (MeshAxisName.DP_REPLICATE, MeshAxisName.DP_SHARD)
            if axis in mesh_names
        )
    return (axis_name,) if axis_name in mesh_names else ()


def named_placement_to_spmd(named: NamedPlacement) -> NamedPlacement:
    """Drop axes that are not present in the active spmd_types mesh."""
    mesh_names = spmd.current_mesh_names()
    if mesh_names is None:
        return {}
    resolved: NamedPlacement = {}
    for axis_name, value in named.items():
        for active_axis_name in _active_axis_names(axis_name, mesh_names):
            resolved[active_axis_name] = value
    return resolved


def _resolve_named_partition_spec(
    named: NamedPartitionSpec,
    ndim: int | None = None,
) -> tuple[spmd.PartitionSpec | None, NamedPlacement]:
    if ndim is not None and len(named) != ndim:
        raise ValueError(
            f"partition_spec length {len(named)} must match tensor ndim {ndim}."
        )
    mesh_names = spmd.current_mesh_names()
    if mesh_names is None:
        return None, {}

    entries = []
    shard_types: NamedPlacement = {}
    for dim, entry in enumerate(named):
        active_axes: list[MeshAxisName] = []
        for axis_name in () if entry is None else (
            entry if isinstance(entry, tuple) else (entry,)
        ):
            for active_axis_name in _active_axis_names(axis_name, mesh_names):
                if active_axis_name in shard_types:
                    raise ValueError(
                        f"Axis {active_axis_name} appears multiple times in "
                        "partition_spec."
                    )
                active_axes.append(active_axis_name)
                shard_types[active_axis_name] = spmd.S(dim)
        if not active_axes:
            entries.append(None)
        elif len(active_axes) == 1:
            entries.append(active_axes[0])
        else:
            entries.append(tuple(active_axes))

    partition_spec = spmd.PartitionSpec(*entries) if shard_types else None
    return partition_spec, shard_types


def _validate_placement_spec(
    spec: PlacementSpec,
    resolved_placement: NamedPlacement,
    shard_types: NamedPlacement,
) -> None:
    for axis_name, value in spec.placement.items():
        if isinstance(value, spmd.Shard):
            raise ValueError(
                "PlacementSpec.placement must use local SPMD types only. "
                f"Got {value} on axis {axis_name}."
            )
    for axis_name in shard_types:
        if resolved_placement.get(axis_name) is not spmd.V:
            raise ValueError(
                "PlacementSpec.placement must set every partition_spec axis to "
                f"spmd.V. Axis {axis_name} has "
                f"{resolved_placement.get(axis_name)!r}."
            )


def placement_to_spmd(
    placement: PlacementLike,
    ndim: int | None = None,
) -> NamedPlacement:
    if isinstance(placement, PlacementSpec):
        resolved = named_placement_to_spmd(placement.placement)
        _, shard_types = _resolve_named_partition_spec(placement.partition_spec, ndim)
        _validate_placement_spec(placement, resolved, shard_types)
        return {**resolved, **shard_types}
    return named_placement_to_spmd(placement)


def named_placement_to_assert_type(
    named: NamedPlacement,
    ndim: int,
) -> tuple[spmd.LocalSpmdType, spmd.PartitionSpec | None]:
    """Lower ``NamedPlacement`` to ``assert_type`` args for a concrete tensor."""
    resolved = named_placement_to_spmd(named)
    local_type: spmd.LocalSpmdType = {}
    shard_types: NamedPlacement = {}
    for axis_name, value in resolved.items():
        if isinstance(value, spmd.Shard):
            dim = value.dim if value.dim >= 0 else ndim + value.dim
            if dim < 0 or dim >= ndim:
                raise ValueError(
                    f"NamedPlacement shards tensor dim {value.dim}, but tensor ndim "
                    f"is {ndim}."
                )
            for existing_axis_name, existing_value in shard_types.items():
                existing_dim = (
                    existing_value.dim
                    if existing_value.dim >= 0
                    else ndim + existing_value.dim
                )
                if existing_dim == dim:
                    raise ValueError(
                        "NamedPlacement cannot determine partition order when "
                        f"tensor dim {dim} is sharded on multiple mesh axes "
                        f"({existing_axis_name}, {axis_name}). Use PlacementSpec "
                        "with an explicit partition_spec."
                    )
            local_type[axis_name] = spmd.V
            shard_types[axis_name] = value
        else:
            local_type[axis_name] = value

    partition_spec = None
    if shard_types:
        partition_spec = shard_types_to_partition_spec(
            shard_types,
            ndim,
        )
    return local_type, partition_spec


def placement_to_assert_type(
    placement: PlacementLike,
    ndim: int,
) -> tuple[spmd.LocalSpmdType, spmd.PartitionSpec | None]:
    if isinstance(placement, PlacementSpec):
        resolved = named_placement_to_spmd(placement.placement)
        partition_spec, shard_types = _resolve_named_partition_spec(
            placement.partition_spec,
            ndim,
        )
        _validate_placement_spec(placement, resolved, shard_types)
        return resolved, partition_spec
    return named_placement_to_assert_type(placement, ndim)


def redistribute_spmd_per_axis(
    x: torch.Tensor,
    src_types: NamedPlacement,
    dst_types: NamedPlacement,
) -> torch.Tensor:
    """Redistribute a tensor per-axis where src != dst."""
    for axis_name, dst_t in dst_types.items():
        src_t = src_types.get(axis_name)
        if src_t is not None and src_t != dst_t:
            mesh = current_mesh()
            assert mesh is not None
            pg = mesh.get_group(axis_name)
            bwd = {"op_dtype": x.dtype}
            x = spmd.redistribute(
                x,
                pg,
                src=src_t,
                dst=dst_t,
                backward_options=bwd,
            )
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

    def parallelize(self, parallel_dims: ParallelDims) -> None:
        """Parallelize this module and all Module children recursively.

        For each module with a ``sharding_config``:

        1. Shard states (parameters and buffers).
        2. Wrap the forward with:
            ``reshard inputs -> [optional local_spmd] forward -> reshard outputs``.

        ``fully_shard`` hooks on ``__call__`` fire around the wrapped ``forward``.

        Each ``ShardingConfig`` field resolves its mesh independently via
        ``resolve_mesh()`` and resolves SPMD placements per axis.
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

        sharding_config = self._sharding_config
        if sharding_config is None:
            return
        mesh = parallel_dims.resolve_mesh(sharding_config.axes())

        with set_current_mesh(mesh):
            for name, param in self.named_parameters(recurse=False):
                if name not in sharding_config.state_shardings:
                    continue
                self.distribute_state(
                    name,
                    param,
                    sharding_config.state_shardings[name],
                    is_param=True,
                )

            for name, buffer in self.named_buffers(recurse=False):
                if name not in sharding_config.state_shardings or buffer is None:
                    continue
                self.distribute_state(
                    name,
                    buffer,
                    sharding_config.state_shardings[name],
                    is_param=False,
                )

            if sharding_config.state_shardings_compute:
                self._convert_to_compute_state_with_hook(
                    sharding_config.state_shardings_compute
                )

            self._cache_pos_arg_names()

            fn = self.forward
            if sharding_config.local_spmd:
                fn = self.local_spmd(fn)

        def with_redistribution(*args: Any, **kwargs: Any) -> Any:
            args, kwargs = self._shard_inputs(args, kwargs)
            outputs = fn(*args, **kwargs)
            return self._shard_outputs(outputs)

        self.forward = with_redistribution

    @staticmethod
    def _convert_state(
        tensor: torch.Tensor,
        mesh,
        rest_types: NamedPlacement,
        compute_types: NamedPlacement,
    ) -> torch.Tensor:
        device_type = tensor.device.type
        bwd = {
            "op_dtype": torch.get_autocast_dtype(device_type)
            if torch.is_autocast_enabled(device_type)
            else tensor.dtype,
            "out_dtype": tensor.dtype,
        }
        for axis_name, dst_t in compute_types.items():
            src_t = rest_types.get(axis_name)
            if src_t is None or src_t == dst_t:
                continue
            pg = mesh.get_group(axis_name)
            if dist.get_world_size(pg) > 1:
                tensor = spmd.convert(
                    tensor,
                    pg,
                    src=src_t,
                    dst=dst_t,
                    backward_options=bwd,
                )
        return tensor

    def _convert_to_compute_state_with_hook(
        self,
        state_shardings_compute: dict[str, PlacementLike],
    ) -> None:
        """Temporarily convert configured states to their forward placements."""
        if not is_spmd_active():
            return
        mesh = current_mesh()
        assert mesh is not None

        sharding_config = self._sharding_config
        assert sharding_config is not None
        state_types = {
            name: (
                placement_to_spmd(sharding_config.state_shardings[name]),
                placement_to_spmd(compute_named),
            )
            for name, compute_named in state_shardings_compute.items()
        }
        originals: dict[str, tuple[torch.Tensor, bool]] = {}

        def get_state(module: nn.Module, name: str) -> tuple[torch.Tensor, bool]:
            if name in module._parameters:
                return module._parameters[name], True
            return module._buffers[name], False

        def set_state(
            module: nn.Module,
            name: str,
            tensor: torch.Tensor,
            *,
            is_param: bool,
        ) -> None:
            if is_param:
                module._parameters[name] = tensor
            else:
                module._buffers[name] = tensor

        def pre_hook(module, args):
            originals.clear()
            for name, (rest_types, compute_types) in state_types.items():
                tensor, is_param = get_state(module, name)
                originals[name] = (tensor, is_param)
                converted = self._convert_state(tensor, mesh, rest_types, compute_types)
                set_state(module, name, converted, is_param=is_param)

        def post_hook(module, args, output):
            for name, (tensor, is_param) in originals.items():
                set_state(module, name, tensor, is_param=is_param)
            originals.clear()

        self.register_forward_pre_hook(pre_hook, with_kwargs=False)
        self.register_forward_hook(post_hook, always_call=True)

    def distribute_state(
        self,
        name: str,
        tensor: torch.Tensor,
        named_placement: PlacementLike,
        *,
        is_param: bool,
    ) -> None:
        """Distribute a single parameter or buffer.

        SPMD values → physical TP shard (if applicable) + ``spmd.assert_type``.
        """
        mesh = current_mesh()
        local_type, partition_spec = (
            placement_to_assert_type(named_placement, tensor.ndim)
            if mesh
            else ({}, None)
        )
        named_placement = placement_to_spmd(named_placement, tensor.ndim) if mesh else {}
        # Validate and collect shards. Values must be raw per-axis types.
        shard_dims: dict[int, str] = {}
        for axis_name, value in named_placement.items():
            assert isinstance(value, (spmd.Shard, spmd.PerMeshAxisLocalSpmdType)), (
                f"Expected per-axis spmd type for state {name!r} on axis "
                f"{axis_name!r}, got {type(value).__name__}: {value!r}"
            )
            if isinstance(value, spmd.Shard):
                assert value.dim not in shard_dims, (
                    f"State {name!r}: axes {shard_dims[value.dim]!r} and "
                    f"{axis_name!r} both shard dim {value.dim}. "
                    f"Multi-axis sharding of a single state dim is not yet supported."
                )
                shard_dims[value.dim] = str(axis_name)

        for axis_name, value in named_placement.items():
            if isinstance(value, spmd.Shard):
                assert mesh is not None
                pg = mesh.get_group(axis_name)
                tensor = spmd.shard(tensor, pg, src=spmd.I, dst=value)

        # register state
        if is_param:
            self.register_parameter(name, nn.Parameter(tensor))
        else:
            persistent = name not in self._non_persistent_buffers_set
            self.register_buffer(name, tensor, persistent=persistent)

        # annotate the registered param/buffer (not the input tensor)
        registered = self._parameters[name] if is_param else self._buffers[name]
        if local_type:
            spmd.assert_type(
                registered,
                local_type,
                partition_spec=partition_spec,
            )

    def local_spmd(
        self,
        fn: Callable[..., Any],
    ) -> Callable[..., Any]:
        sharding_config = self._sharding_config
        assert sharding_config is not None
        pos_arg_names = self._cache_pos_arg_names()
        in_dst = sharding_config.in_dst_shardings or {}
        resolved_in: tuple[PlacementLike | None, ...] = tuple(
            in_dst[name] if name in in_dst else None
            for name in pos_arg_names
        )
        out_src = sharding_config.out_src_shardings
        if out_src is None:
            resolved_out: tuple[PlacementLike | None, ...] = ()
        elif isinstance(out_src, tuple):
            resolved_out = out_src
        else:
            resolved_out = (out_src,)

        def assert_types(
            tensors: Iterable[Any],
            specs: Iterable[PlacementLike | None],
        ) -> None:
            if not spmd.is_type_checking():
                return
            for tensor, types in zip(tensors, specs):
                if isinstance(tensor, torch.Tensor) and types is not None and types:
                    local_type, partition_spec = placement_to_assert_type(
                        types,
                        tensor.ndim,
                    )
                    spmd.assert_type(
                        tensor,
                        local_type,
                        partition_spec=partition_spec,
                    )

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            assert_types(args, resolved_in)
            if kwargs:
                assert_types(
                    kwargs.values(),
                    tuple(
                        in_dst[name] if name in in_dst else None
                        for name in kwargs
                    ),
                )

            if spmd.is_type_checking():
                with spmd.typecheck(local=True):
                    result = fn(*args, **kwargs)
            else:
                result = fn(*args, **kwargs)

            outputs = (result,) if isinstance(result, torch.Tensor) else result
            assert_types(outputs, resolved_out)
            return result

        return wrapper

    def _shard_inputs(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Redistribute inputs to desired placements.

        Per input present in ``in_src_shardings`` / ``in_dst_shardings``:
        resolve a mesh from that input's NamedPlacements, then:
        1. If plain tensor and ``in_src_shardings`` declared, wrap as
           DTensor via ``DTensor.from_local`` on that mesh.
        2. If ``in_dst_shardings`` declared, redistribute on the same mesh.
        """
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

        for name, value in new_kwargs.items():
            if not isinstance(value, torch.Tensor):
                continue
            if name in in_src_shardings and name in in_dst_shardings:
                src_types = placement_to_spmd(in_src_shardings[name], value.ndim)
                dst_types = placement_to_spmd(in_dst_shardings[name], value.ndim)
                value = redistribute_spmd_per_axis(value, src_types, dst_types)
            new_kwargs[name] = value

        new_args = tuple(new_kwargs.pop(name) for name in pos_arg_names)
        return new_args, new_kwargs

    def _shard_outputs(
        self,
        outputs: Any,
    ) -> Any:
        """Redistribute output to desired placement.

        TODO: Currently only handles a single DTensor output. Extend to
        support nested outputs (tuples, dicts) when models with
        multi-tensor forward returns (e.g., Flux, MoE) adopt config-based
        sharding. ``out_dst_shardings`` would also need to become a nested
        structure.
        """
        sharding_config = self._sharding_config
        assert sharding_config is not None

        if sharding_config.out_dst_shardings is None:
            return outputs

        if isinstance(outputs, torch.Tensor):
            out_src, out_dst = (
                sharding_config.out_src_shardings,
                sharding_config.out_dst_shardings,
            )
            if out_src is None:
                out_src = out_dst
            src_types = placement_to_spmd(out_src, outputs.ndim)
            dst_types = placement_to_spmd(out_dst, outputs.ndim)
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
