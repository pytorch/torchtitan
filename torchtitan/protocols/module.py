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
from spmd_types.runtime import get_local_type, get_partition_spec, has_local_type
from torch.distributed.tensor import distribute_tensor, DTensor
from torch.distributed.tensor.experimental import local_map
from torch.distributed.tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.utils._pytree import tree_map

from torchtitan.config import Configurable
from torchtitan.distributed.parallel_dims import ParallelDims, SpmdLayout
from torchtitan.distributed.spmd_types import (
    set_current_spmd_mesh,
    spmd_distribute_tensor,
)
from torchtitan.distributed.utils import (
    check_dtensor_placements_match,
    get_spmd_backend,
)
from torchtitan.protocols.sharding import (
    RedistributionSpec,
    resolve_placements,
    ShardingConfig,
)


def _placement_from_spmd_type(
    axis_type: spmd.PerMeshAxisSpmdType,
) -> Placement:
    if axis_type == spmd.R or axis_type == spmd.I:
        return Replicate()
    if axis_type == spmd.P:
        return Partial()
    assert isinstance(axis_type, spmd.Shard)
    return Shard(axis_type.dim)


def _resolve_placements_after_redist(
    layout: SpmdLayout,
    redist: RedistributionSpec.Config,
    mesh,
) -> tuple[Placement, ...]:
    src_axis_type = layout.per_axis_spmd_types().get(redist.axis)
    if src_axis_type is not None and src_axis_type != redist.src:
        raise ValueError(
            "RedistributionSpec src does not match source sharding for "
            f"axis {redist.axis.value!r}: {redist.src!r} vs {src_axis_type!r}."
        )

    placements = list(resolve_placements(layout, mesh))
    mesh_axis_names = mesh.mesh_dim_names or ()
    axis = redist.axis.value
    if axis not in mesh_axis_names:
        return tuple(placements)

    axis_idx = mesh_axis_names.index(axis)
    if mesh.size(axis_idx) > 1:
        placements[axis_idx] = _placement_from_spmd_type(redist.dst)
    return tuple(placements)


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
        with self._preserve_buffer_spmd_types():
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
        if get_spmd_backend() != "spmd_types":
            yield
            return

        saved = {
            fqn: SpmdLayout(
                # pyrefly: ignore [bad-argument-type]
                axis_types=get_local_type(buf),
                partition_spec=get_partition_spec(buf),
            )
            for fqn, buf in self.named_buffers()
            if has_local_type(buf)
        }
        try:
            yield
        finally:
            for fqn, buf in self.named_buffers():
                if fqn in saved and not has_local_type(buf):
                    layout = saved[fqn]
                    spmd.assert_type(
                        buf,
                        layout.axis_types,
                        partition_spec=layout.partition_spec,
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

        self._distribute_states(parallel_dims)
        self._cache_pos_arg_names()
        fn = self._maybe_wrap_with_local_region(self.forward, parallel_dims)

        def forward_with_redistribution(*args, **kwargs):
            args, kwargs = self._redistribute_inputs(parallel_dims, args, kwargs)
            outputs = fn(*args, **kwargs)
            return self._redistribute_outputs(parallel_dims, outputs)

        self.forward = forward_with_redistribution

    def _spmd_distribute_state(
        self,
        parallel_dims: ParallelDims,
        name: str,
        tensor: torch.Tensor,
        layout: SpmdLayout,
        *,
        is_param: bool,
    ) -> None:
        # Call get_optional_mesh with include_singleton_axes=True, so we're able to call assert_type()
        # using all axes, and defer size-1 axis filtering to spmd_types internals.
        mesh = parallel_dims.get_optional_mesh(
            [axis.value for axis in layout.axes()], include_singleton_axes=True
        )
        assert mesh is not None
        assert mesh.mesh_dim_names is not None, "DeviceMesh must have named axes"

        tensor = spmd_distribute_tensor(tensor, mesh, layout)
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
            spmd.assert_type(
                registered,
                layout.axis_types,
                layout.partition_spec,
            )

    def _distribute_states(self, parallel_dims: ParallelDims) -> None:
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
                self._spmd_distribute_state(
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
                self._spmd_distribute_state(
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

    def _maybe_wrap_with_local_region(
        self,
        fn: Callable,
        parallel_dims: ParallelDims,
    ) -> Callable:
        """Wrap ``fn`` with a local-tensor region if local_map config is set.

        Output placements come from ``out_src_shardings``. DTensor uses
        ``local_map``; the spmd_types backend uses the analogous local SPMD
        wrapper. Input placements are inferred so this boundary does not
        duplicate the input redistribution contract.
        """
        sharding_config = self._sharding_config
        assert sharding_config is not None
        if sharding_config.local_map is None:
            return fn

        out_src = sharding_config.out_src_shardings
        if out_src is None:
            raise AssertionError(
                f"{type(self).__name__}: local_map is set but "
                "out_src_shardings is None."
            )

        if parallel_dims.spmd_backend == "spmd_types":
            return self._spmd_apply_local_map(fn, out_src)
        return self._apply_local_map(fn, parallel_dims, out_src)

    def _apply_local_map(
        self,
        fn: Callable,
        parallel_dims: ParallelDims,
        out_src: SpmdLayout | tuple[SpmdLayout | None, ...],
    ) -> Callable:
        """Apply DTensor local_map for a local-tensor compute region."""
        sharding_config = self._sharding_config
        assert sharding_config is not None
        lm = sharding_config.local_map
        assert lm is not None

        if isinstance(out_src, tuple):
            out_named: list[SpmdLayout] = [p for p in out_src if p is not None]
        else:
            out_named = [out_src]
        # in_grad_placements may contain None for non-tensor args; filter
        # them out for mesh resolution -- local_map passes None through.
        grad_named = lm.in_grad_placements

        resolved_mesh = parallel_dims.resolve_shared_mesh(
            out_named + (list(grad_named) if grad_named else [])
        )
        if resolved_mesh is None:
            return fn

        out_placements: tuple[tuple[Placement, ...], ...] = tuple(
            resolve_placements(p, resolved_mesh) for p in out_named
        )
        in_grad_placements = (
            tuple(
                resolve_placements(p, resolved_mesh) if p is not None else None
                for p in grad_named
            )
            if grad_named is not None
            else None
        )
        return local_map(
            fn,
            out_placements=out_placements,
            in_grad_placements=in_grad_placements,
            device_mesh=resolved_mesh,
        )

    def _spmd_apply_local_map(
        self,
        fn: Callable,
        out_src: SpmdLayout | tuple[SpmdLayout | None, ...],
    ) -> Callable:
        """Apply spmd_types local_map for a local-tensor compute region."""
        out_types = tree_map(
            lambda layout: (layout.axis_types, layout.partition_spec),
            out_src,
            is_leaf=lambda x: isinstance(x, SpmdLayout),
        )
        return spmd.local_map(
            out_types=out_types,
        )(fn)

    def _redistribute_inputs(
        self,
        parallel_dims: ParallelDims,
        args: tuple,
        kwargs: dict,
    ) -> tuple[tuple, dict]:
        """Redistribute inputs to desired placements.

        Per input present in ``in_src_shardings`` / ``in_redist``:
        resolve a mesh from that input's SpmdLayouts, then:
        1. If plain tensor and ``in_src_shardings`` declared, wrap as
           DTensor via ``DTensor.from_local`` on that mesh.
        2. If ``in_redist`` declared, redistribute on the same mesh.
        """
        sharding_config = self._sharding_config
        assert sharding_config is not None

        if (
            sharding_config.in_src_shardings is None
            and sharding_config.in_redist is None
        ):
            return args, kwargs

        pos_arg_names = [
            name for name in self._cache_pos_arg_names() if name not in kwargs
        ]
        new_kwargs = dict(zip(pos_arg_names, args, strict=False))
        new_kwargs.update(kwargs)

        in_src_shardings = sharding_config.in_src_shardings or {}
        in_redist = sharding_config.in_redist or {}

        for name, value in new_kwargs.items():
            if not isinstance(value, torch.Tensor):
                continue
            src_spmd_layout = in_src_shardings.get(name)
            redist = in_redist.get(name)

            if parallel_dims.spmd_backend == "spmd_types":
                if src_spmd_layout is None:
                    if redist is not None:
                        raise ValueError(
                            f"{type(self).__name__}.{name}: SPMD input "
                            "redistribution requires explicit in_src_shardings."
                        )
                    continue

                # SPMD source placements are part of the config contract: assert
                # before redistributing so typechecking catches placement mismatch.
                # Gate assertion so compile doesn't error.
                if spmd.is_type_checking():
                    spmd.assert_type(
                        value,
                        src_spmd_layout.axis_types,
                        src_spmd_layout.partition_spec,
                    )

                if redist is not None:
                    new_kwargs[name] = redist.build()(value)
                    continue

                new_kwargs[name] = value
                continue

            if src_spmd_layout is None:
                if redist is not None:
                    raise ValueError(
                        f"{type(self).__name__}.{name}: input redistribution "
                        "requires explicit in_src_shardings."
                    )
                continue

            mesh = parallel_dims.resolve_shared_mesh([src_spmd_layout])
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

            if isinstance(value, DTensor) and src_spmd_layout is not None:
                expected = resolve_placements(src_spmd_layout, mesh)
                if not check_dtensor_placements_match(
                    value.placements, expected, value.ndim
                ):
                    raise ValueError(
                        f"{type(self).__name__}.{name}: input DTensor has "
                        f"placements {value.placements}, but in_src_shardings "
                        f"expects {expected}."
                    )

            if redist is not None and isinstance(value, DTensor):
                desired = _resolve_placements_after_redist(
                    src_spmd_layout, redist, mesh
                )
                if not check_dtensor_placements_match(
                    value.placements, desired, value.ndim
                ):
                    value = value.redistribute(placements=desired, async_op=True)

            new_kwargs[name] = value

        new_args = tuple(new_kwargs.pop(name) for name in pos_arg_names)
        return new_args, new_kwargs

    def _redistribute_outputs(self, parallel_dims: ParallelDims, outputs: Any) -> Any:
        """Redistribute output to desired placement.

        TODO: Currently only handles a single DTensor output. Extend to
        support nested outputs (tuples, dicts) when models with
        multi-tensor forward returns (e.g., Flux, MoE) adopt config-based
        sharding.
        """
        sharding_config = self._sharding_config
        assert sharding_config is not None

        out_src = sharding_config.out_src_shardings
        out_redist = sharding_config.out_redist
        if out_redist is None:
            return outputs

        if parallel_dims.spmd_backend == "spmd_types":
            if not isinstance(outputs, torch.Tensor):
                return outputs

            if out_src is None:
                raise ValueError(
                    f"{type(self).__name__}: SPMD output redistribution "
                    "requires explicit out_src_shardings."
                )
            if isinstance(out_src, tuple):
                raise ValueError(
                    f"{type(self).__name__}: SPMD output redistribution only "
                    "supports a single tensor output."
                )
            # SPMD source placements are part of the config contract: assert
            # before redistributing so typechecking catches placement mismatch.
            # Gate assertion so compile doesn't error.
            if spmd.is_type_checking():
                spmd.assert_type(
                    outputs,
                    out_src.axis_types,
                    out_src.partition_spec,
                )

            if out_redist is not None:
                return out_redist.build()(outputs)

        if isinstance(out_src, tuple):
            raise ValueError(
                f"{type(self).__name__}: output redistribution only supports "
                "a single tensor output."
            )
        if out_src is None:
            raise ValueError(
                f"{type(self).__name__}: output redistribution requires explicit "
                "out_src_shardings."
            )

        mesh = parallel_dims.resolve_shared_mesh([out_src])
        if mesh is None:
            return outputs

        if isinstance(outputs, DTensor) and out_src is not None:
            expected = resolve_placements(out_src, mesh)
            if not check_dtensor_placements_match(
                outputs.placements, expected, outputs.ndim
            ):
                raise ValueError(
                    f"{type(self).__name__}: output DTensor has placements "
                    f"{outputs.placements}, but out_src_shardings expects "
                    f"{expected}."
                )

        desired = _resolve_placements_after_redist(out_src, out_redist, mesh)
        if isinstance(outputs, DTensor) and not check_dtensor_placements_match(
            outputs.placements, desired, outputs.ndim
        ):
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
