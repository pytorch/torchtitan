# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Iterator
from contextlib import contextmanager
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
from torchtitan.distributed.utils import current_mesh, set_current_mesh
from torchtitan.protocols.sharding import (
    active_spmd_placement,
    is_placement_like,
    NamedPlacement,
    NamedPlacementSpmd,
    placement_axes,
    placement_to_spmd_assert_type,
    PlacementLike,
    resolve_placements,
    ShardingConfig,
)

if TYPE_CHECKING:
    from torchtitan.distributed.parallel_dims import ParallelDims


@contextmanager
def preserve_buffer_spmd(model: nn.Module) -> Iterator[None]:
    """Preserve spmd_types annotations on buffers replaced during initialization."""
    saved: dict[str, Any] = {}
    for fqn, buf in model.named_buffers():
        if spmd.has_local_type(buf):
            saved[fqn] = dict(spmd.get_local_type(buf))
    yield
    for fqn, buf in model.named_buffers():
        if fqn in saved and not spmd.has_local_type(buf):
            spmd.assert_type(buf, saved[fqn])


def redistribute_spmd_per_axis(
    x: torch.Tensor,
    src_types: NamedPlacementSpmd,
    dst_types: NamedPlacementSpmd,
) -> torch.Tensor:
    """Redistribute a local tensor along axes whose SPMD type changes."""
    mesh = current_mesh()
    if mesh is None:
        return x

    for axis_name, dst_t in dst_types.items():
        src_t = src_types.get(axis_name)
        if src_t is None or src_t == dst_t:
            continue
        x = spmd.redistribute(
            x,
            mesh.get_group(axis_name),
            src=src_t,
            dst=dst_t,
            backward_options={"op_dtype": x.dtype},
        )
    return x


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
        if (
            self._sharding_config.state_shardings_compute
            and parallel_dims.spmd_backend == "spmd"
        ):
            self._convert_to_compute_state_with_hook(
                parallel_dims,
                self._sharding_config.state_shardings_compute,
            )
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
        placement: PlacementLike,
        *,
        is_param: bool,
    ) -> None:
        mesh = parallel_dims.resolve_mesh(placement_axes(placement))
        if mesh is None:
            return

        with set_current_mesh(mesh):
            local_type, partition_spec = placement_to_spmd_assert_type(placement)
            for axis_name, axis_type in active_spmd_placement(placement).items():
                if isinstance(axis_type, spmd.Shard):
                    tensor = spmd.shard(
                        tensor,
                        mesh.get_group(axis_name),
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

            if local_type:
                spmd.assert_type(
                    registered,
                    local_type,
                    partition_spec=partition_spec,
                )

    @staticmethod
    def _convert_spmd_state(
        tensor: torch.Tensor,
        mesh,
        rest_types: NamedPlacementSpmd,
        compute_types: NamedPlacementSpmd,
    ) -> torch.Tensor:
        """Convert a param/buffer from rest placement to compute placement."""
        device_type = tensor.device.type
        backward_options = {
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
            if torch.distributed.get_world_size(pg) > 1:
                tensor = spmd.convert(
                    tensor,
                    pg,
                    src=src_t,
                    dst=dst_t,
                    backward_options=backward_options,
                )
        return tensor

    def _convert_to_compute_state_with_hook(
        self,
        parallel_dims: ParallelDims,
        state_shardings_compute: dict[str, PlacementLike],
    ) -> None:
        """Temporarily retype params/buffers for local SPMD forward compute."""
        sharding_config = self._sharding_config
        assert sharding_config is not None

        state_types = {}
        for name, compute_placement in state_shardings_compute.items():
            rest_placement = sharding_config.state_shardings.get(name)
            if rest_placement is None:
                raise ValueError(
                    f"{type(self).__name__}.{name} has compute placement but "
                    "no state_shardings entry."
                )
            mesh = parallel_dims.resolve_shared_mesh(
                [rest_placement, compute_placement]
            )
            if mesh is None:
                continue
            with set_current_mesh(mesh):
                state_types[name] = (
                    mesh,
                    active_spmd_placement(rest_placement),
                    active_spmd_placement(compute_placement),
                )
        if not state_types:
            return

        originals: dict[str, tuple[torch.Tensor, bool]] = {}

        def get_state(module: nn.Module, name: str) -> tuple[torch.Tensor, bool]:
            if name in module._parameters:
                return module._parameters[name], True
            if name in module._buffers:
                return module._buffers[name], False
            raise ValueError(f"{type(module).__name__} has no state named {name!r}.")

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
            for name, (mesh, rest_types, compute_types) in state_types.items():
                tensor, is_param = get_state(module, name)
                originals[name] = (tensor, is_param)
                converted = self._convert_spmd_state(
                    tensor,
                    mesh,
                    rest_types,
                    compute_types,
                )
                set_state(module, name, converted, is_param=is_param)

        def post_hook(module, args, output):
            for name, (tensor, is_param) in originals.items():
                set_state(module, name, tensor, is_param=is_param)
            originals.clear()

        self.register_forward_pre_hook(pre_hook, with_kwargs=False)
        self.register_forward_hook(post_hook, always_call=True)

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
                placement_to_spmd_assert_type(in_dst[name]) for name in checked_names
            )
            out_types = tree_map(
                placement_to_spmd_assert_type,
                out_src,
                is_leaf=is_placement_like,
            )

            def local_map_call(*checked_args: Any) -> Any:
                for name, value in zip(checked_names, checked_args):
                    bound.arguments[name] = value
                return fn(*bound.args, **bound.kwargs)

            return spmd.local_map(
                in_types=in_types,
                out_types=out_types,
                local_typecheck=True,
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
            named_placements = sharding_config.state_shardings.get(name)
            if named_placements is None:
                raise ValueError(
                    f"{type(self).__name__}.{name} has no placement declared "
                    "in sharding_config.state_shardings."
                )
            if parallel_dims.spmd_backend == "spmd":
                self._register_spmd_state(
                    parallel_dims,
                    name,
                    param,
                    named_placements,
                    is_param=True,
                )
                continue
            axes = placement_axes(named_placements)
            mesh = parallel_dims.resolve_mesh(axes)
            if mesh is None:
                continue
            placements = resolve_placements(named_placements, mesh)
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
            named_placements = sharding_config.state_shardings.get(name)
            if named_placements is None:
                raise ValueError(
                    f"{type(self).__name__}.{name} (buffer) has no placement "
                    "declared in sharding_config.state_shardings."
                )
            if buffer is None:
                # ``register_buffer(name, None)`` reserves a slot to be filled
                # by ``init_states`` later; nothing to distribute yet.
                continue
            if parallel_dims.spmd_backend == "spmd":
                self._register_spmd_state(
                    parallel_dims,
                    name,
                    buffer,
                    named_placements,
                    is_param=False,
                )
                continue
            axes = placement_axes(named_placements)
            mesh = parallel_dims.resolve_mesh(axes)
            if mesh is None:
                continue
            placements = resolve_placements(named_placements, mesh)
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
        all NamedPlacements must resolve to the same mesh.
        """
        sharding_config = self._sharding_config
        assert sharding_config is not None
        if parallel_dims.spmd_backend == "spmd":
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
            out_src_list: list[NamedPlacement] = [p for p in out_src if p is not None]
        else:
            out_src_list = [out_src]

        missing_in = [name for name in pos_args if name not in in_dst]
        if missing_in:
            raise AssertionError(
                f"{type(self).__name__}: local_map is set but in_dst_shardings "
                f"is missing entries for: {missing_in}"
            )
        in_named: list[NamedPlacement] = [in_dst[name] for name in pos_args]
        out_named: list[NamedPlacement] = out_src_list
        # in_grad_placements may contain None for non-tensor args; filter
        # them out for mesh resolution -- local_map passes None through.
        grad_named: list[NamedPlacement | None] = list(lm.in_grad_placements)

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

        for name, value in new_kwargs.items():
            if not isinstance(value, torch.Tensor):
                continue
            src_named_placements = in_src_shardings.get(name)
            dst_named_placements = in_dst_shardings.get(name)
            mesh = parallel_dims.resolve_shared_mesh(
                [src_named_placements, dst_named_placements]
            )
            if mesh is None:
                continue

            if parallel_dims.spmd_backend == "spmd":
                if src_named_placements is None or dst_named_placements is None:
                    continue
                with set_current_mesh(mesh):
                    value = redistribute_spmd_per_axis(
                        value,
                        active_spmd_placement(src_named_placements),
                        active_spmd_placement(dst_named_placements),
                    )
                new_kwargs[name] = value
                continue

            if (
                not isinstance(value, DTensor)
                and parallel_dims.spmd_backend == "full_dtensor"
            ):
                raise ValueError("Got a plain Tensor under the full_dtensor mode.")

            if not isinstance(value, DTensor) and src_named_placements is not None:
                layout = resolve_placements(src_named_placements, mesh)
                value = DTensor.from_local(
                    value,
                    mesh,
                    layout,
                    run_check=False,
                )

            if dst_named_placements is not None and isinstance(value, DTensor):
                desired = resolve_placements(dst_named_placements, mesh)
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

        out_named_placements = sharding_config.out_dst_shardings
        if parallel_dims.spmd_backend == "spmd":
            out_dst = out_named_placements
            if out_dst is None or not isinstance(outputs, torch.Tensor):
                return outputs

            out_src = sharding_config.out_src_shardings or out_dst
            if not is_placement_like(out_src):
                return outputs

            mesh = parallel_dims.resolve_shared_mesh([out_src, out_dst])
            if mesh is None:
                return outputs
            with set_current_mesh(mesh):
                return redistribute_spmd_per_axis(
                    outputs,
                    active_spmd_placement(out_src),
                    active_spmd_placement(out_dst),
                )

        mesh = parallel_dims.resolve_shared_mesh([out_named_placements])
        if mesh is None:
            return outputs

        if out_named_placements is not None:
            desired = resolve_placements(out_named_placements, mesh)
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
