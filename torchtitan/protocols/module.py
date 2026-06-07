# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import spmd_types as spmd
import torch
import torch.nn as nn
from torch.distributed.tensor import distribute_tensor, DTensor
from torch.distributed.tensor.experimental import local_map
from torch.distributed.tensor.placement_types import Placement
from torch.utils._pytree import tree_map

from torchtitan.config import Configurable
from torchtitan.distributed.spmd_types import (
    redistribute_spmd_per_axis,
    set_current_spmd_mesh,
    spmd_layout_to_assert_type,
)
from torchtitan.protocols.sharding import resolve_placements, ShardingConfig
from torchtitan.protocols.types import SpmdLayout

if TYPE_CHECKING:
    from torchtitan.distributed.parallel_dims import ParallelDims


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

        # TODO(fegin): Change to assert once ALL Models are migrated to use _sharding_config.
        if self._sharding_config is None:
            return

        self._shard_states(parallel_dims)
        self._cache_pos_arg_names()
        fn = self._maybe_wrap_with_local_map(self.forward, parallel_dims)

        def forward_with_redistribution(*args, **kwargs):
            args, kwargs = self._redistribute_inputs(parallel_dims, args, kwargs)
            outputs = fn(*args, **kwargs)
            return self._redistribute_outputs(parallel_dims, outputs)

        self.forward = forward_with_redistribution

    def _register_spmd_state(
        self,
        parallel_dims: ParallelDims,
        name: str,
        tensor: torch.Tensor,
        layout: SpmdLayout,
        *,
        is_param: bool,
    ) -> None:
        mesh = parallel_dims._global_meshes["spmd_dense_for_fwdbwd"]
        assert mesh.mesh_dim_names is not None, "DeviceMesh must have named axes"

        for axis_name, axis_type in layout.shard_types().items():
            axis = axis_name.value
            axis_size = (
                mesh.size(mesh.mesh_dim_names.index(axis))
                if axis in mesh.mesh_dim_names
                else 1
            )
            if isinstance(axis_type, spmd.Shard) and axis_size > 1:
                tensor = spmd.shard(
                    tensor,
                    mesh.get_group(axis),
                    src=spmd.I,
                    dst=axis_type,
                )

        if is_param:
            self.register_parameter(name, nn.Parameter(tensor))
            registered = self._parameters[name]
        else:
            persistent = name not in self._non_persistent_buffers_set
            self.register_buffer(name, tensor, persistent=persistent)
            registered = self._buffers[name]

        # assert_type resolves SpmdLayout's string mesh axis names to concrete
        # runtime mesh-axis objects, so a mesh context is required here.
        with set_current_spmd_mesh(mesh):
            local_type, partition_spec = spmd_layout_to_assert_type(layout)
            spmd.assert_type(
                registered,
                local_type,
                partition_spec=partition_spec,
            )

    def _maybe_wrap_with_local_spmd(self, fn: Callable) -> Callable:
        sharding_config = self._sharding_config
        assert sharding_config is not None
        in_dst = sharding_config.in_dst_shardings or {}
        out_src = sharding_config.out_src_shardings
        if sharding_config.local_map is None:
            return fn
        if not in_dst:
            raise ValueError(
                f"{type(self).__name__}: local SPMD requires in_dst_shardings."
            )
        if out_src is None:
            raise ValueError(
                f"{type(self).__name__}: local SPMD requires out_src_shardings."
            )

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not spmd.is_type_checking():
                return fn(*args, **kwargs)

            sig = inspect.signature(fn)
            bound = sig.bind_partial(*args, **kwargs)
            checked_names = [name for name in bound.arguments if name in in_dst]
            checked_values = tuple(bound.arguments[name] for name in checked_names)
            in_types = tuple(
                spmd_layout_to_assert_type(in_dst[name]) for name in checked_names
            )
            out_types = tree_map(
                spmd_layout_to_assert_type,
                out_src,
                is_leaf=lambda x: isinstance(x, SpmdLayout),
            )

            def local_map_call(*checked_args: Any) -> Any:
                for name, value in zip(checked_names, checked_args):
                    bound.arguments[name] = value
                return fn(*bound.args, **bound.kwargs)

            return spmd.local_map(
                in_types=in_types,
                out_types=out_types,
            )(local_map_call)(*checked_values)

        return wrapper

    def _shard_states(self, parallel_dims: ParallelDims) -> None:
        """Distribute params and buffers per ``state_shardings``.

        Each entry resolves its own mesh via ``resolve_mesh``, so different
        params on the same Module may live on different meshes. An
        already-DTensor param/buffer indicates it was distributed by a
        sibling (e.g. weight tying); skip but verify placements agree.
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
            if parallel_dims.spmd_backend == "spmd_types":
                self._register_spmd_state(
                    parallel_dims,
                    name,
                    param,
                    spmd_layout,
                    is_param=True,
                )
                continue
            axes = spmd_layout.axes()
            mesh = parallel_dims.resolve_mesh(axes)
            if mesh is None:
                continue
            placements = resolve_placements(spmd_layout, mesh)
            if isinstance(param, DTensor):
                if tuple(param.placements) != tuple(placements):
                    raise ValueError(
                        f"{type(self).__name__}.{name} is already a DTensor with "
                        f"placements {param.placements}, but its sharding_config "
                        f"expects {placements}. This usually means a tied parameter "
                        "is referenced by two modules with conflicting sharding "
                        "configs."
                    )
                continue
            self.register_parameter(
                name,
                nn.Parameter(
                    distribute_tensor(param, mesh, list(placements)),
                    requires_grad=param.requires_grad,
                ),
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
            if parallel_dims.spmd_backend == "spmd_types":
                self._register_spmd_state(
                    parallel_dims,
                    name,
                    buffer,
                    spmd_layout,
                    is_param=False,
                )
                continue
            axes = spmd_layout.axes()
            mesh = parallel_dims.resolve_mesh(axes)
            if mesh is None:
                continue
            placements = resolve_placements(spmd_layout, mesh)
            persistent = name not in self._non_persistent_buffers_set
            self.register_buffer(
                name,
                distribute_tensor(buffer, mesh, list(placements)),
                persistent=persistent,
            )

    def _maybe_wrap_with_local_map(
        self,
        fn: Callable,
        parallel_dims: ParallelDims,
    ) -> Callable:
        """Wrap ``fn`` with ``local_map`` if ``sharding_config.local_map`` is set.

        Input placements come from ``in_dst_shardings`` (the same dict
        ``_redistribute_inputs`` uses to pre-align inputs); output placements
        from ``out_src_shardings``; only ``in_grad_placements`` lives on
        ``LocalMapConfig``. ``local_map`` takes a single ``device_mesh``, so
        all SpmdLayouts must resolve to the same mesh.
        """
        sharding_config = self._sharding_config
        assert sharding_config is not None
        if parallel_dims.spmd_backend == "spmd_types":
            return self._maybe_wrap_with_local_spmd(fn)

        if sharding_config.local_map is None:
            return fn

        lm = sharding_config.local_map
        in_dst = sharding_config.in_dst_shardings or {}
        pos_args = self._cache_pos_arg_names()
        out_src = sharding_config.out_src_shardings
        if out_src is None:
            raise AssertionError(
                f"{type(self).__name__}: local_map is set but "
                "out_src_shardings is None."
            )
        if isinstance(out_src, tuple):
            out_src_list: list[SpmdLayout] = [p for p in out_src if p is not None]
        else:
            out_src_list = [out_src]

        missing_in = [name for name in pos_args if name not in in_dst]
        if missing_in:
            raise AssertionError(
                f"{type(self).__name__}: local_map is set but in_dst_shardings "
                f"is missing entries for: {missing_in}"
            )
        in_named: list[SpmdLayout] = [in_dst[name] for name in pos_args]
        out_named: list[SpmdLayout] = out_src_list
        # in_grad_placements may contain None for non-tensor args; filter
        # them out for mesh resolution -- local_map passes None through.
        grad_named: list[SpmdLayout | None] = list(lm.in_grad_placements)

        resolved_mesh = parallel_dims.resolve_shared_mesh(
            in_named + out_named + grad_named
        )
        if resolved_mesh is None:
            return fn

        out_placements: tuple[tuple[Placement, ...], ...] = tuple(
            resolve_placements(p, resolved_mesh) for p in out_named
        )
        return local_map(
            fn,
            in_placements=tuple(resolve_placements(p, resolved_mesh) for p in in_named),
            out_placements=out_placements,
            in_grad_placements=tuple(
                resolve_placements(p, resolved_mesh) if p is not None else None
                for p in grad_named
            ),
            device_mesh=resolved_mesh,
        )

    def _redistribute_inputs(
        self,
        parallel_dims: ParallelDims,
        args: tuple,
        kwargs: dict,
    ) -> tuple[tuple, dict]:
        """Redistribute inputs to desired placements.

        Per input present in ``in_src_shardings`` / ``in_dst_shardings``:
        resolve a mesh from that input's SpmdLayouts, then:
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
            src_spmd_layout = in_src_shardings.get(name)
            dst_spmd_layout = in_dst_shardings.get(name)

            if parallel_dims.spmd_backend == "spmd_types":
                if src_spmd_layout is None:
                    if dst_spmd_layout is not None:
                        raise ValueError(
                            f"{type(self).__name__}.{name}: SPMD input "
                            "redistribution requires explicit in_src_shardings."
                        )
                    continue

                # SPMD source placements are part of the config contract: assert
                # before redistributing so typechecking catches boundary drift.
                local_type, partition_spec = spmd_layout_to_assert_type(
                    src_spmd_layout
                )
                spmd.assert_type(
                    value,
                    local_type,
                    partition_spec=partition_spec,
                )

                if dst_spmd_layout is None:
                    new_kwargs[name] = value
                    continue
                value = redistribute_spmd_per_axis(
                    value,
                    src_spmd_layout.shard_types(),
                    dst_spmd_layout.shard_types(),
                )
                new_kwargs[name] = value
                continue

            mesh = parallel_dims.resolve_shared_mesh(
                [src_spmd_layout, dst_spmd_layout]
            )
            if mesh is None:
                continue

            if (
                not isinstance(value, DTensor)
                and parallel_dims.spmd_backend == "full_dtensor"
            ):
                raise ValueError("Got a plain Tensor under the full_dtensor mode.")

            if not isinstance(value, DTensor) and src_spmd_layout is not None:
                layout = resolve_placements(src_spmd_layout, mesh)
                value = DTensor.from_local(
                    value,
                    mesh,
                    layout,
                    run_check=False,
                )

            if dst_spmd_layout is not None and isinstance(value, DTensor):
                desired = resolve_placements(dst_spmd_layout, mesh)
                if value.placements != desired:
                    value = value.redistribute(placements=desired, async_op=True)

            new_kwargs[name] = value

        new_args = tuple(new_kwargs.pop(name) for name in pos_arg_names)
        return new_args, new_kwargs

    def _redistribute_outputs(self, parallel_dims: ParallelDims, outputs: Any) -> Any:
        """Redistribute output to desired placement.

        TODO: Currently only handles a single DTensor output. Extend to
        support nested outputs (tuples, dicts) when models with
        multi-tensor forward returns (e.g., Flux, MoE) adopt config-based
        sharding. ``out_dst_shardings`` would also need to become a nested
        structure.
        """
        sharding_config = self._sharding_config
        assert sharding_config is not None

        out_spmd_layout = sharding_config.out_dst_shardings
        if parallel_dims.spmd_backend == "spmd_types":
            if not isinstance(outputs, torch.Tensor):
                return outputs

            out_src = sharding_config.out_src_shardings
            if out_src is None:
                if sharding_config.out_dst_shardings is not None:
                    raise ValueError(
                        f"{type(self).__name__}: SPMD output redistribution "
                        "requires explicit out_src_shardings."
                    )
                return outputs
            # SPMD source placements are part of the config contract: assert
            # before redistributing so typechecking catches boundary drift.
            local_type, partition_spec = spmd_layout_to_assert_type(out_src)
            spmd.assert_type(
                outputs,
                local_type,
                partition_spec=partition_spec,
            )

            out_dst = sharding_config.out_dst_shardings
            if out_dst is None:
                return outputs
            return redistribute_spmd_per_axis(
                outputs,
                out_src.shard_types(),
                out_dst.shard_types(),
            )

        mesh = parallel_dims.resolve_shared_mesh([out_spmd_layout])
        if mesh is None:
            return outputs

        if out_spmd_layout is not None:
            desired = resolve_placements(out_spmd_layout, mesh)
            if isinstance(outputs, DTensor) and outputs.placements != desired:
                outputs = outputs.redistribute(placements=desired, async_op=True)

        return outputs


class ModuleList(nn.ModuleList, Module):
    """Module-protocol-compatible version of ``nn.ModuleList``."""

    pass


class ModuleDict(nn.ModuleDict, Module):
    """Module-protocol-compatible version of ``nn.ModuleDict``."""

    pass


class Sequential(nn.Sequential, Module):
    """Module-protocol-compatible version of ``nn.Sequential``."""

    pass
