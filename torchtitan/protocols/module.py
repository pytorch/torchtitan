# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

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
    LocalSpmdConfig,
    NamedPlacement,
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


def named_placement_to_spmd(named: NamedPlacement) -> NamedPlacement:
    """Drop axes that are not present in the active spmd_types mesh."""
    mesh_names = spmd.current_mesh_names()
    if mesh_names is None:
        return dict(named)
    resolved = dict(named)
    if MeshAxisName.DP in resolved and MeshAxisName.DP not in mesh_names:
        dp_value = resolved.pop(MeshAxisName.DP)
        for axis_name in (MeshAxisName.DP_REPLICATE, MeshAxisName.DP_SHARD):
            if axis_name in mesh_names:
                resolved[axis_name] = dp_value
    return {
        axis_name: value
        for axis_name, value in resolved.items()
        if axis_name in mesh_names
    }


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
            local_type[axis_name] = spmd.V
            shard_types[axis_name] = value
        else:
            local_type[axis_name] = value

    partition_spec = None
    if shard_types:
        partition_spec = shard_types_to_partition_spec(
            shard_types,
            ndim,
            axis_order=tuple(resolved.keys()),
        )
    return local_type, partition_spec


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
            ``reshard inputs -> [optional local_map] forward -> reshard outputs``.

        ``fully_shard`` hooks on ``__call__`` fire around the wrapped ``forward``.

        Each ``ShardingConfig`` field resolves its mesh independently via
        ``resolve_mesh()`` and resolves its placements independently via
        ``resolve_placements()``.
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

            if sharding_config.state_tp_ir:
                self._install_spmd_tp_ir_param_hook(sharding_config.state_tp_ir)

            self._cache_pos_arg_names()

            fn = self.forward
            if sharding_config.local_spmd is not None:
                fn = self.local_spmd(fn, sharding_config.local_spmd)

        def with_redistribution(*args: Any, **kwargs: Any) -> Any:
            args, kwargs = self._shard_inputs(args, kwargs)
            outputs = fn(*args, **kwargs)
            return self._shard_outputs(outputs)

        self.forward = with_redistribution

    def _install_spmd_tp_ir_param_hook(self, param_names: set[str]) -> None:
        """Convert configured I@tp params to R@tp for forward compute."""
        if not is_spmd_active():
            return
        mesh = current_mesh()
        assert mesh is not None
        if "tp" not in mesh.mesh_dim_names:
            return
        tp_pg = mesh.get_group("tp")
        if dist.get_world_size(tp_pg) == 1:
            return

        param_names = set(param_names)
        original_params: dict[str, torch.Tensor] = {}

        def pre_hook(module, args):
            original_params.clear()
            for name in param_names:
                param = module._parameters[name]
                original_params[name] = param
                device_type = param.device.type
                op_dtype = (
                    torch.get_autocast_dtype(device_type)
                    if torch.is_autocast_enabled(device_type)
                    else param.dtype
                )
                bwd = {"op_dtype": op_dtype, "out_dtype": param.dtype}
                module._parameters[name] = spmd.convert(
                    param,
                    tp_pg,
                    src=spmd.I,
                    dst=spmd.R,
                    backward_options=bwd,
                )

        def post_hook(module, args, output):
            for name, param in original_params.items():
                module._parameters[name] = param
            original_params.clear()

        self.register_forward_pre_hook(pre_hook, with_kwargs=False)
        self.register_forward_hook(post_hook, always_call=True)

    def distribute_state(
        self,
        name: str,
        tensor: torch.Tensor,
        named_placement: NamedPlacement,
        *,
        is_param: bool,
    ) -> None:
        """Distribute a single parameter or buffer.

        SPMD values → physical TP shard (if applicable) + ``spmd.assert_type``.
        """
        named_placement = named_placement_to_spmd(named_placement)
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
                mesh = current_mesh()
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
        types = named_placement_to_spmd(named_placement)
        if types:
            spmd.assert_type(registered, types)

    def local_spmd(
        self,
        fn: Callable[..., Any],
        lm: LocalSpmdConfig,
    ) -> Callable[..., Any]:
        sharding_config = self._sharding_config
        assert sharding_config is not None
        pos_arg_names = self._cache_pos_arg_names()
        in_dst = sharding_config.in_dst_shardings or {}
        resolved_in: tuple[NamedPlacement | None, ...] = tuple(
            named_placement_to_spmd(in_dst[name]) if name in in_dst else None
            for name in pos_arg_names
        )
        out_src = sharding_config.out_src_shardings
        if out_src is None:
            resolved_out: tuple[NamedPlacement | None, ...] = ()
        elif isinstance(out_src, tuple):
            resolved_out = tuple(named_placement_to_spmd(p) for p in out_src)
        else:
            resolved_out = (named_placement_to_spmd(out_src),)

        def assert_types(
            tensors: Iterable[Any],
            specs: Iterable[NamedPlacement | None],
        ) -> None:
            if not spmd.is_type_checking():
                return
            for t, types in zip(tensors, specs):
                if isinstance(t, torch.Tensor) and types is not None and types:
                    local_type, partition_spec = named_placement_to_assert_type(
                        types,
                        t.ndim,
                    )
                    spmd.assert_type(t, local_type, partition_spec=partition_spec)

        def body(*args: Any, **kwargs: Any) -> Any:
            assert_types(args, resolved_in)
            if spmd.is_type_checking():
                with spmd.typecheck(local=True):
                    result = fn(*args, **kwargs)
            else:
                result = fn(*args, **kwargs)
            outputs = (result,) if isinstance(result, torch.Tensor) else result
            assert_types(outputs, resolved_out)
            return result

        return body

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
                src_types = named_placement_to_spmd(in_src_shardings[name])
                dst_types = named_placement_to_spmd(in_dst_shardings[name])
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
            src_types = named_placement_to_spmd(out_src)
            dst_types = named_placement_to_spmd(out_dst)
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
