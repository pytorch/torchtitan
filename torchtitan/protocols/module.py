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
from torch.distributed.tensor.experimental import local_map

from torchtitan.config import Configurable
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.protocols.sharding import (
    demote_degenerate_shards,
    resolve_mesh,
    resolve_placements,
    resolve_shared_mesh,
    ShardingConfig,
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

        if self._sharding_config is None:
            return

        self._shard_states(parallel_dims)
        self._cache_pos_arg_names()
        fn = self._maybe_wrap_with_local_map(self.forward, parallel_dims)

        def with_redistribution(*args, **kwargs):
            args, kwargs = self._shard_inputs(parallel_dims, args, kwargs)
            outputs = fn(*args, **kwargs)
            return self._shard_outputs(parallel_dims, outputs)

        self.forward = with_redistribution

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
            named_placements = sharding_config.state_shardings.get(name)
            if named_placements is None:
                continue
            axes = named_placements.keys()
            mesh, mesh_axis_names = resolve_mesh(axes, parallel_dims)
            if mesh is None:
                continue
            placements = demote_degenerate_shards(
                resolve_placements(named_placements, mesh_axis_names), mesh
            )
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
                nn.Parameter(distribute_tensor(param, mesh, list(placements))),
            )

        for name, buffer in self.named_buffers(recurse=False):
            named_placements = sharding_config.state_shardings.get(name)
            if named_placements is None or buffer is None:
                continue
            axes = named_placements.keys()
            mesh, mesh_axis_names = resolve_mesh(axes, parallel_dims)
            if mesh is None:
                continue
            placements = demote_degenerate_shards(
                resolve_placements(named_placements, mesh_axis_names), mesh
            )
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

        ``local_map`` takes a single ``device_mesh``, so all (non-``None``)
        NamedPlacements within one LocalMapConfig must resolve to the same
        mesh. ``resolve_shared_mesh`` enforces that and returns the
        resolved mesh. ``None`` entries mark non-tensor args (e.g. ints,
        lists) and pass through unchanged.
        """
        sharding_config = self._sharding_config
        assert sharding_config is not None
        if sharding_config.local_map is None:
            return fn

        lm = sharding_config.local_map
        resolved_mesh, mesh_axis_names = resolve_shared_mesh(
            lm.in_placements + lm.out_placements + lm.in_grad_placements,
            parallel_dims,
        )
        if resolved_mesh is None:
            return fn
        mesh = resolved_mesh  # narrow for closure capture

        def _resolve(p):
            if p is None:
                return None
            return demote_degenerate_shards(
                resolve_placements(p, mesh_axis_names), mesh
            )

        in_placements = tuple(_resolve(p) for p in lm.in_placements)
        out_placements = tuple(_resolve(p) for p in lm.out_placements)
        in_grad_placements = tuple(_resolve(p) for p in lm.in_grad_placements)
        return local_map(
            fn,
            in_placements=in_placements,
            out_placements=out_placements,
            in_grad_placements=in_grad_placements,
            device_mesh=mesh,
            # Under full_dtensor, callers feed DTensors sharded on the full
            # SPMD mesh; local_map redistributes to the per-arg placements
            # declared above. Legacy path keeps the strict default so
            # placement mismatches surface as errors.
            redistribute_inputs=parallel_dims.full_dtensor,
        )

    def _shard_inputs(
        self,
        parallel_dims: ParallelDims,
        args: tuple,
        kwargs: dict,
    ) -> tuple[tuple, dict]:
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

        pos_arg_names = [
            name for name in self._cache_pos_arg_names() if name not in kwargs
        ]
        new_kwargs = dict(zip(pos_arg_names, args))
        new_kwargs.update(kwargs)

        in_dst_shardings = sharding_config.in_dst_shardings or {}
        in_src_shardings = sharding_config.in_src_shardings or {}
        in_grad_shardings = sharding_config.local_input_grad_placements or {}

        for name, value in new_kwargs.items():
            if not isinstance(value, torch.Tensor):
                continue
            src_named_placements = in_src_shardings.get(name)
            dst_named_placements = in_dst_shardings.get(name)
            grad_named_placements = in_grad_shardings.get(name)
            mesh, mesh_axis_names = resolve_shared_mesh(
                [src_named_placements, dst_named_placements, grad_named_placements],
                parallel_dims,
            )
            if mesh is None:
                continue

            if not isinstance(value, DTensor) and src_named_placements is not None:
                layout = demote_degenerate_shards(
                    resolve_placements(src_named_placements, mesh_axis_names), mesh
                )
                grad_placements: tuple | None = None
                if grad_named_placements is not None:
                    grad_placements = demote_degenerate_shards(
                        resolve_placements(grad_named_placements, mesh_axis_names),
                        mesh,
                    )
                value = DTensor.from_local(
                    value,
                    mesh,
                    layout,
                    run_check=False,
                    grad_placements=grad_placements,
                )

            if dst_named_placements is not None and isinstance(value, DTensor):
                desired = demote_degenerate_shards(
                    resolve_placements(dst_named_placements, mesh_axis_names), mesh
                )
                if value.placements != desired:
                    value = value.redistribute(placements=desired, async_op=True)

            new_kwargs[name] = value

        new_args = tuple(new_kwargs.pop(name) for name in pos_arg_names)
        return new_args, new_kwargs

    def _shard_outputs(self, parallel_dims: ParallelDims, outputs: Any) -> Any:
        """Redistribute output to desired placement.

        TODO: Currently only handles a single DTensor output. Extend to
        support nested outputs (tuples, dicts) when models with
        multi-tensor forward returns (e.g., Flux, MoE) adopt config-based
        sharding. ``out_dst_shardings`` would also need to become a nested
        structure.
        """
        sharding_config = self._sharding_config
        assert sharding_config is not None

        out_named_placements = sharding_config.out_dst_shardings
        out_grad_named_placements = sharding_config.local_output_grad_placements
        mesh, mesh_axis_names = resolve_shared_mesh(
            [out_named_placements, out_grad_named_placements], parallel_dims
        )
        if mesh is None:
            return outputs

        if out_named_placements is not None:
            desired = demote_degenerate_shards(
                resolve_placements(out_named_placements, mesh_axis_names), mesh
            )
            if isinstance(outputs, DTensor) and outputs.placements != desired:
                outputs = outputs.redistribute(placements=desired, async_op=True)

        # Unwrap DTensor output to local tensor with the declared backward
        # gradient placement. Mirrors NoParallel(local_output_grad_placements
        # =...): the module returns a local tensor; in backward, the
        # upstream local d_output is wrapped back as a DTensor with the
        # declared placement (e.g. Partial to skip a downstream all-reduce).
        if out_grad_named_placements is not None and isinstance(outputs, DTensor):
            grad_placements = demote_degenerate_shards(
                resolve_placements(out_grad_named_placements, mesh_axis_names), mesh
            )
            outputs = outputs.to_local(grad_placements=grad_placements)
        return outputs

    @classmethod
    def from_nn_module(cls, nn_module_cls: type[nn.Module]) -> type["Module"]:
        """Create a ``Module``-protocol-compatible version of *nn_module_cls*.

        The returned class inherits from ``(nn_module_cls, Module)`` and has the
        same constructor signature as *nn_module_cls*. It also has an
        auto-generated ``Config`` subclass of ``Module.Config`` that
        forwards positional / keyword args to the underlying
        ``nn_module_cls`` constructor at ``build`` time and applies
        ``sharding_config`` / ``param_init`` to the resulting instance --
        so call sites that need declarative sharding can use the
        Config-based path while call sites that don't can keep direct
        construction.

        * If *nn_module_cls* defines ``reset_parameters``, the injected
          ``_init_self_parameters`` delegates to it.
        * Otherwise ``_init_self_parameters`` is the inherited default from
          ``Module``.

        Results are cached so that repeated calls with the same class return
        the identical class object.

        Usage::

            LayerNorm = Module.from_nn_module(nn.LayerNorm)
            # Direct construction (no declarative sharding):
            norm = LayerNorm(dim, eps)
            # Config-based construction (with declarative sharding):
            norm = LayerNorm.Config(
                sharding_config=norm_config(enable_sp=False),
            ).build(dim, eps)
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

        # Auto-generated Config: inherits sharding_config / param_init from
        # Module.Config; build forwards args to nn_module_cls and applies
        # the slots. dataclass(slots=True) prevents ad-hoc attributes; the
        # inherited fields are the only state the Config carries.
        @dataclass(kw_only=True, slots=True)
        class _AutoConfig(Module.Config):
            def build(self, *args: Any, **kwargs: Any) -> Any:
                instance = new_cls(*args, **kwargs)
                if self.sharding_config is not None:
                    instance._sharding_config = self.sharding_config
                if self.param_init is not None:
                    instance._param_init = self.param_init
                return instance

        _AutoConfig.__name__ = "Config"
        _AutoConfig.__qualname__ = f"{name}.Config"
        _AutoConfig.__module__ = __name__
        new_cls.Config = _AutoConfig  # pyrefly: ignore [missing-attribute]

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
