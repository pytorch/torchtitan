# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FlexShard (meta-pytorch/flex_shard) as a ``ShardingBackend``, in place of
FSDP2's ``fully_shard``.

With ``parallelism.fsdp_backend=FlexShardBackend.Config()``, ``apply_to_decoder``
gives FlexShard one ``fsdp2_compatible`` bucket for each unit FSDP2 would pass to
``fully_shard``, with the same placements, mixed precision and reshard policy.
The buckets' collectives then match FSDP2's byte for byte, so training from the
same parameters is bitwise identical.

FlexShard parameters and grads are plain local shards, with no DTensor.
``apply_to_decoder`` wraps each FlexShard parameter's ``param_init`` function
with flex_shard's ``init_from_full``: it fills the full parameter and the shard
keeps its part, as one device would initialize it, while each of FSDP2's DTensor
shards draws its own random values. ``flex_shard_grad_norm`` reduces a bucket's
grad norm as DTensor does. One more difference changes only memory and
communication: FSDP2 skips the embedding's backward all-gather
(``set_unshard_in_backward``), which FlexShard can't yet, so it gathers the
embedding again in backward, with the same values. FlexShard is imported only
when it is selected.
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from importlib.util import find_spec
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.distributed.sharding_backend import ShardingBackend

if TYPE_CHECKING:
    from torchtitan.components.checkpointer import CheckpointManager
    from torchtitan.components.loss import Loss
    from torchtitan.components.optim import Optim
    from torchtitan.config import DebugConfig, PrecisionConfig, TrainingConfig
    from torchtitan.config.parallelism import ParallelismConfig
    from torchtitan.distributed.fsdp import ShardingUnit
    from torchtitan.distributed.parallelism_context import ParallelismContext
    from torchtitan.models.base.model import BaseModel
    from torchtitan.models.common.decoder import Decoder

logger = logging.getLogger(__name__)

__all__ = [
    "FlexShardBackend",
    "flex_shard_grad_norm",
]


def flex_shard_grad_norm(
    params: list[torch.Tensor],
    *,
    mesh: DeviceMesh,
    norm_type: float,
    error_if_nonfinite: bool,
    foreach: bool | None,
) -> torch.Tensor:
    """The norm of the grads of FlexShard parameters ``params``, whose buckets
    are on the 1D ``mesh`` (``flex_shard.get_mesh``).

    The grads are plain local shards over ``mesh``. This reduces them in the
    order ``get_total_norm`` does for DTensor grads sharded over ``mesh``, so the
    result is bitwise equal: each grad's local norm, reduced over the mesh, then
    the norm of those norms.
    """
    grads = [p.grad for p in params if p.grad is not None]
    # From flex_shard's examples/deepseek_v3/train.py.
    if foreach is False:
        norms = torch.stack([torch.linalg.vector_norm(g, norm_type) for g in grads])
    else:
        norms = torch.stack(torch._foreach_norm(grads, norm_type))
    if math.isinf(norm_type):
        dist.all_reduce(norms, op=dist.ReduceOp.MAX, group=mesh.get_group())
    else:
        norms **= norm_type
        dist.all_reduce(norms, group=mesh.get_group())
        norms **= 1.0 / norm_type
    total_norm = torch.linalg.vector_norm(norms, norm_type)
    if error_if_nonfinite and not torch.isfinite(total_norm):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from `parameters` "
            "is non-finite, so it cannot be clipped. To disable this error and "
            "scale the gradients by the non-finite norm anyway, set "
            "`error_if_nonfinite=False`"
        )
    return total_norm


def _raise_if_unsupported(unsupported: list[str]) -> None:
    if unsupported:
        raise ValueError(
            "parallelism.fsdp_backend=FlexShardBackend.Config() does not support "
            f"{', '.join(unsupported)} yet."
        )


class FlexShardBackend(ShardingBackend):
    """FlexShard, bitwise identical to FSDP2's ``fully_shard`` for what it
    supports. ``validate_model`` and ``validate_trainer`` reject the rest. Each
    rejection is temporary: it goes once FlexShard supports the feature."""

    @dataclass(kw_only=True, slots=True)
    class Config(ShardingBackend.Config):
        pass

    def validate_model(
        self,
        model_config: "BaseModel.Config",
        *,
        parallelism: "ParallelismConfig",
        training: "TrainingConfig",
        debug: "DebugConfig",
    ) -> None:
        from torchtitan.distributed.accelerator_graph import (
            accelerator_graphs_supported,
        )
        from torchtitan.models.base.module import Module
        from torchtitan.models.common.decoder import Decoder
        from torchtitan.quantization.utils import has_quantization

        # TODO: Remove each rejection once FlexShard supports the feature.
        unsupported = [
            f"parallelism.{name}={degree}"
            for name, degree in (
                (
                    "data_parallel_replicate_degree",
                    parallelism.data_parallel_replicate_degree,
                ),
                ("tensor_parallel_degree", parallelism.tensor_parallel_degree),
                ("context_parallel_degree", parallelism.context_parallel_degree),
                ("expert_parallel_degree", parallelism.expert_parallel_degree),
                ("pipeline_parallel_degree", parallelism.pipeline_parallel_degree),
            )
            if degree != 1
        ]
        if parallelism.fsdp_symm_mem_scope is not None:
            unsupported.append(
                f"parallelism.fsdp_symm_mem_scope={parallelism.fsdp_symm_mem_scope!r}"
            )
        if parallelism.fsdp_enable_cpu_offloading:
            unsupported.append("parallelism.fsdp_enable_cpu_offloading=True")
        if parallelism.fsdp_unshard_in_current_stream:
            unsupported.append("parallelism.fsdp_unshard_in_current_stream=True")
        if parallelism.fsdp_defer_gradient_reduction:
            # FlexShard has no equivalent of FSDP2's set_is_last_backward.
            unsupported.append("parallelism.fsdp_defer_gradient_reduction=True")

        # Decoder._apply_fsdp is the only path that applies fsdp_backend, so a
        # model that overrides it would shard with FSDP2.
        model_cls = model_config._owner
        if getattr(model_cls, "_apply_fsdp", None) is not Decoder._apply_fsdp:
            name = (
                type(model_config).__qualname__
                if model_cls is None
                else model_cls.__name__
            )
            unsupported.append(f"models that override Decoder._apply_fsdp ({name})")
        # FlexShard's Shard placement all-gathers a bucket in one dtype.
        if any(
            module_config.preserve_dtype_params
            for _, module_config, _, _ in model_config.traverse(
                Module.Config, recurse=True
            )
        ):
            unsupported.append("preserve_dtype_params")
        # FlexShard parameters are plain local shards, and only param_init
        # initializes them as one device would: apply_to_decoder wraps it with
        # init_from_full. reset_parameters() would draw every rank's shard as the
        # whole parameter, with the same values.
        if any(
            module_config.param_init is None
            and callable(getattr(module_config._owner, "reset_parameters", None))
            for _, module_config, _, _ in model_config.traverse(
                Module.Config, recurse=True
            )
        ):
            unsupported.append(
                "modules initialized by reset_parameters() (set param_init)"
            )
        # FlexShard stores parameters as plain tensors, which would drop the
        # quantized all-gather that MXFP8 and NVFP4 weights implement for FSDP2.
        if has_quantization(model_config):
            unsupported.append("MXFP8 and NVFP4 weights")

        if not training.disable_accelerator_graphs and accelerator_graphs_supported():
            unsupported.append(
                "accelerator graphs (set training.disable_accelerator_graphs=True)"
            )
        if debug.spmd_typechecking:
            unsupported.append("debug.spmd_typechecking")
        _raise_if_unsupported(unsupported)
        # Checked here because flex_shard is imported only once the model is
        # sharded, after distributed init and the model build.
        if find_spec("flex_shard") is None:
            raise ImportError(
                "parallelism.fsdp_backend=FlexShardBackend.Config() needs flex_shard "
                "(github.com/meta-pytorch/flex_shard), which isn't installed. "
                ".github/workflows/unit_test_gpu.yaml installs the commit torchtitan "
                "is tested with."
            )

    def validate_trainer(
        self,
        *,
        loss: "Loss.Config",
        checkpointer: "CheckpointManager.Config | None",
        optim: "Optim.Config",
        trainer: type | None,
    ) -> None:
        from torchtitan.components.loss import ChunkedLossWrapper
        from torchtitan.components.optim import DistMuon, GradientClipper
        from torchtitan.trainer import Trainer

        # TODO: Remove each rejection once FlexShard supports the feature. They
        # read the loss, checkpointer, optimizers and trainer, which
        # validate_model_training_config (config/validation.py), where
        # validate_model runs, doesn't receive, so TrainingEngine.Config's
        # __post_init__ calls this method instead.
        unsupported = []
        if isinstance(loss, ChunkedLossWrapper.Config):
            unsupported.append("ChunkedLossWrapper")
        if checkpointer is not None:
            unsupported.append("checkpointing")
        if optim.ema is not None:
            unsupported.append("optim.ema")
        if any(isinstance(o, DistMuon.Config) for o in optim.optimizer.optimizers):
            # DistMuon needs DTensor parameters.
            unsupported.append("DistMuon")
        # Only GradientClipper is known to clip through the sharding backend.
        # Another clipper might clip FlexShard's plain local shards by each rank's
        # own norm, with no error.
        if type(optim.gradient_clipper) is not GradientClipper.Config:
            unsupported.append(
                "gradient clippers other than GradientClipper "
                f"({type(optim.gradient_clipper).__qualname__})"
            )

        # Other trainers shard, reduce or publish parameters in ways that FlexShard
        # doesn't support yet: GraphTrainer shards with SimpleFSDP, TorchFT's
        # cross-replica all-reduce hooks FSDP2's parameter groups, and the RL
        # trainer publishes state_dict(), which holds local shards, as full weights.
        if trainer is not Trainer:
            name = (
                "None"
                if trainer is None
                else f"{trainer.__module__}.{trainer.__qualname__}"
            )
            unsupported.append(
                f"trainers other than torchtitan.trainer.Trainer ({name})"
            )
        _raise_if_unsupported(unsupported)

    def apply_to_decoder(
        self,
        model: "Decoder",
        *,
        parallelism_context: "ParallelismContext",
        precision: "PrecisionConfig",
        parallelism: "ParallelismConfig",
    ) -> None:
        from torchtitan.config import TORCH_DTYPE_MAP
        from torchtitan.distributed.fsdp import (
            decoder_sharding_units,
            resolve_fsdp_mesh,
        )

        dp_mesh, _ = resolve_fsdp_mesh(parallelism_context)
        embedding, norm_and_lm_head, blocks = decoder_sharding_units(
            model,
            dp_mesh=dp_mesh,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            pp_enabled=parallelism_context.pp_enabled,
        )
        _apply_flex_shard(
            model,
            dp_mesh=dp_mesh,
            units=[unit for unit in (embedding, norm_and_lm_head) if unit is not None]
            + blocks,
            param_dtype=TORCH_DTYPE_MAP[precision.param_dtype],
            reduce_dtype=TORCH_DTYPE_MAP[precision.reduce_dtype],
        )

    def clip_grad_norm_(
        self,
        parameters: list[torch.nn.Parameter],
        max_norm: float,
        *,
        foreach: bool | None,
        parallelism_context: "ParallelismContext",
    ) -> torch.Tensor:
        # validate_model has checked that flex_shard is installed.
        # pyrefly: ignore [missing-import]
        import flex_shard

        from torchtitan.distributed import utils as dist_utils

        # FlexShard grads are plain local shards over their bucket's mesh.
        params_by_mesh: dict[DeviceMesh, list[torch.Tensor]] = defaultdict(list)
        for p in parameters:
            if p.grad is not None:
                mesh = flex_shard.get_mesh(p)
                # The root bucket holds every parameter that no other bucket does.
                assert mesh is not None, tuple(p.shape)
                params_by_mesh[mesh].append(p)
        norms = [
            flex_shard_grad_norm(
                params,
                mesh=mesh,
                norm_type=2.0,
                error_if_nonfinite=False,
                foreach=foreach,
            )
            for mesh, params in params_by_mesh.items()
        ]
        # A rank without grads, e.g. a PP rank, contributes a norm of 0.
        if not norms:
            norms = [torch.nn.utils.get_total_norm([], foreach=foreach)]
        return dist_utils.clip_grads_with_mesh_norms_(
            list(params_by_mesh.values()),
            norms,
            max_norm=max_norm,
            norm_type=2.0,
            foreach=foreach,
            pp_mesh=parallelism_context.get_optional_mesh("pp"),
        )


def _apply_flex_shard(
    model: "Decoder",
    *,
    dp_mesh: DeviceMesh,
    units: list["ShardingUnit"],
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
) -> None:
    """Shard ``model`` with FlexShard (meta-pytorch/flex_shard), one
    ``fsdp2_compatible`` bucket per sharding unit.

    Each bucket names the unit's modules, with its placements and reshard
    setting, and a root bucket takes the parameters left over, as
    ``fully_shard(model)`` does. Gradients are summed over ``dp_mesh`` without
    division, as ``disable_fsdp_gradient_division`` makes FSDP2 do. The buckets'
    collectives then match FSDP2's byte for byte. Each sharded parameter's
    ``param_init`` function is wrapped with flex_shard's ``init_from_full``, so
    the shards start as one device's initialization.
    """
    # flex_shard is an optional dependency, needed only for this backend.
    # pyrefly: ignore [missing-import]
    from flex_shard import (
        BucketSpec,
        custom_placements,
        flex_shard,
        get_flex_shard_global_layouts,
        init_from_full,
        MixedPrecisionPolicy,
    )

    from torchtitan.models.base.module import Module
    from torchtitan.models.common.param_init import skip_param_init

    # validate_model allows only dp_shard.
    assert dp_mesh.mesh_dim_names == ("dp_shard",), dp_mesh.mesh_dim_names
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    # Shard dims other than 0, which FSDP2 gets from shard_placement_fn.
    shard_dims = {
        param: placement.dim
        for unit in units
        for param, placement in unit.placements.items()
    }

    def placement_fn(named_params, mesh):
        del mesh
        return {
            fqn: (custom_placements.Shard(shard_dims.get(param, 0)),)
            for fqn, param in named_params
        }

    def bucket(patterns: list[str], reshard_after_forward: bool) -> BucketSpec:
        return BucketSpec(
            patterns,
            placement_fn=placement_fn,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            gradient_divide_factor=1.0,
            reshard_after_forward=reshard_after_forward,
            fsdp2_compatible=True,
        )

    module_fqns = {module: fqn for fqn, module in model.named_modules()}
    buckets = [
        bucket(
            [module_fqns[module] for module in unit.modules],
            unit.reshard_after_forward,
        )
        for unit in units
    ]
    # fully_shard(model) puts the remaining parameters in the root unit, which
    # FSDP2 does not reshard after forward.
    params_in_units = {
        param
        for unit in units
        for module in unit.modules
        for param in module.parameters()
    }
    root_fqns = [
        fqn for fqn, param in model.named_parameters() if param not in params_in_units
    ]
    if root_fqns:
        buckets.append(bucket(root_fqns, reshard_after_forward=False))

    flex_shard(model, buckets=buckets)
    # FlexShard parameters are plain local shards. Wrap each one's param_init
    # function so that it fills the full parameter and the shard keeps its part:
    # one device's initialization, whatever the sharding. to_empty() replaces
    # the parameters but keeps the modules, so the wrapped functions survive it.
    # validate_model rejects modules without param_init, whose
    # reset_parameters() would bypass this.
    for fqn, layout in get_flex_shard_global_layouts(model).items():
        module_fqn, _, name = fqn.rpartition(".")
        module = model.get_submodule(module_fqn)
        assert isinstance(module, Module) and module._param_init is not None, fqn
        # A skipped init, e.g. tok_embeddings' with tied weights, which lm_head
        # initializes, stays a no-op on the shard.
        if module._param_init.get(name, skip_param_init) is not skip_param_init:
            # A new dict: modules built from one Config share its param_init.
            module._param_init = {
                **module._param_init,
                name: partial(
                    init_from_full, layout=layout, init_fn=module._param_init[name]
                ),
            }
    logger.info("Applied FlexShard to the model")
