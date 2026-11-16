# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data parallelism with FlexShard in place of FSDP2's ``fully_shard``.

With ``parallelism.fsdp_backend=FlexShardConfig()``, ``apply_fsdp_to_decoder``
gives FlexShard (meta-pytorch/flex_shard) one ``fsdp2_compatible`` bucket for
each group it would pass to ``fully_shard``, with the same placements, mixed
precision and reshard policy. The buckets' collectives then match FSDP2's byte
for byte, so training from the same parameters is bitwise identical.

With expert parallelism, ``fully_shard`` gives an MoE block one group per mesh,
and the block gets one bucket per group: its routed experts on ``edp_shard``,
the rest on ``dp_shard``. EP has already split each routed-expert parameter over
``ep``, and torchtitan declares that split as the parameter's outer layout.

FlexShard parameters and grads are plain local shards, with no DTensor.
``init_param`` initializes each parameter as one device would, while each of
FSDP2's DTensor shards draws its own random values. ``flex_shard_grad_norm_mesh``
gives the mesh a grad's norm is reduced over. Two more differences change only
memory and communication. FSDP2 skips the embedding's backward all-gather
(``set_unshard_in_backward``), which FlexShard can't yet, so it gathers the
embedding again in backward, with the same values. And an MoE block's expert
bucket hooks the routed experts, not the block, so FlexShard reshards them as
soon as the routed experts finish forward or backward, while FSDP2 keeps the
block's expert group unsharded until the whole block finishes. FlexShard is
imported only when it is selected.
"""

import sys
import weakref
from collections.abc import Callable
from importlib.util import find_spec
from typing import Any, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

if TYPE_CHECKING:
    from torchtitan.models.base.model import BaseModel
    from torchtitan.training_engine import TrainingEngine

__all__ = [
    "flex_shard_grad_norm_mesh",
    "has_flex_shard_params",
    "init_param",
    "record_grad_norm_mesh",
    "record_param_layouts",
    "validate_flex_shard_training",
]

# Where each FlexShard parameter's local shard sits in the full parameter, by
# module and parameter name, for init_param. to_empty() replaces FlexShard's
# parameters but keeps the modules.
# TODO: Read each parameter's layout from flex_shard instead, once its
# parameters keep their layouts through to_empty().
_PARAM_LAYOUTS: weakref.WeakKeyDictionary[
    nn.Module, dict[str, Any]
] = weakref.WeakKeyDictionary()

# The mesh to reduce a grad norm over for the parameters of the FlexShard buckets
# on a mesh, when it isn't that mesh, for flex_shard_grad_norm_mesh. Keyed by the
# mesh the buckets hold, which keeps the entry alive as long as the model.
_GRAD_NORM_MESHES: weakref.WeakKeyDictionary[
    DeviceMesh, DeviceMesh
] = weakref.WeakKeyDictionary()


def record_param_layouts(model: nn.Module) -> None:
    """Record where each of ``model``'s FlexShard parameters sits in the full
    parameter, for ``init_param``. Call it after ``flex_shard(model)``.

    flex_shard composes each layout with the outer layout the parameter
    declared before ``flex_shard(model)``, such as expert parallelism's split of
    the routed experts.
    """
    # pyrefly: ignore [missing-import]
    from flex_shard import get_flex_shard_global_layouts

    for fqn, layout in get_flex_shard_global_layouts(model).items():
        module_fqn, _, name = fqn.rpartition(".")
        _PARAM_LAYOUTS.setdefault(model.get_submodule(module_fqn), {})[name] = layout


def init_param(
    module: nn.Module,
    name: str,
    param: torch.Tensor,
    init_fn: Callable[[torch.Tensor], object],
) -> None:
    """Initialize ``module``'s parameter ``name``, ``param``, with ``init_fn``.

    A FlexShard parameter is a plain local shard. ``init_fn`` fills a tensor of
    the full parameter's shape, and the shard keeps its part (flex_shard's
    ``init_from_full``). With the same RNG state on every rank, the shards are
    one device's initialization, whatever the sharding. Each call briefly holds
    one full parameter: under expert parallelism, a routed-expert parameter with
    all experts.
    """
    layout = _PARAM_LAYOUTS.get(module, {}).get(name)
    if layout is None:
        init_fn(param)
        return
    # pyrefly: ignore [missing-import]
    from flex_shard import init_from_full

    init_from_full(param, layout, init_fn)


def has_flex_shard_params(module: nn.Module) -> bool:
    """Whether FlexShard sharded any of ``module``'s own parameters, which only
    ``init_param`` initializes as one device would."""
    return module in _PARAM_LAYOUTS


def record_grad_norm_mesh(bucket_mesh: DeviceMesh, mesh: DeviceMesh) -> None:
    """Record that ``flex_shard_grad_norm_mesh`` gives the 1D ``mesh`` for the
    parameters of the FlexShard buckets on ``bucket_mesh``.

    For parameters that an outer sharding split over more ranks before FlexShard
    sharded each part over ``bucket_mesh``, e.g. routed experts under expert
    parallelism.
    """
    # Meshes compare by value. An equal mesh that an earlier model recorded
    # would stay the key, and the entry would go when that model does.
    _GRAD_NORM_MESHES.pop(bucket_mesh, None)
    _GRAD_NORM_MESHES[bucket_mesh] = mesh


def flex_shard_grad_norm_mesh(param: torch.Tensor) -> DeviceMesh | None:
    """The mesh ``param``'s grad norm is reduced over if it's a FlexShard
    parameter, else None.

    A FlexShard parameter and its grad are plain local shards over the mesh of
    their bucket, or over the mesh ``record_grad_norm_mesh`` recorded for it.
    """
    # FlexShard parameters exist only once flex_shard is imported, which
    # happens only when it is selected.
    flex_shard = sys.modules.get("flex_shard")
    mesh = None if flex_shard is None else flex_shard.get_mesh(param)
    return None if mesh is None else _GRAD_NORM_MESHES.get(mesh, mesh)


def validate_flex_shard_training(
    config: "TrainingEngine.Config", model_config: "BaseModel.Config"
) -> None:
    """Reject what FlexShard doesn't support yet, before anything is built.

    ``TrainingEngine.__init__`` calls this for every trainer, with the config
    after recipes and overrides.
    """
    from torchtitan.components.loss import ChunkedLossWrapper
    from torchtitan.components.optim.optimizer import DistMuon
    from torchtitan.distributed.accelerator_graph import accelerator_graphs_supported
    from torchtitan.models.base.module import Module
    from torchtitan.models.common.decoder import Decoder
    from torchtitan.models.common.moe import MoE, RoutedExperts
    from torchtitan.models.common.moe.token_dispatcher import AllToAllTokenDispatcher
    from torchtitan.quantization.utils import has_quantization
    from torchtitan.trainer import Trainer

    parallelism = config.parallelism
    unsupported = [
        f"parallelism.{name}={degree}"
        for name, degree in (
            (
                "data_parallel_replicate_degree",
                parallelism.data_parallel_replicate_degree,
            ),
            ("tensor_parallel_degree", parallelism.tensor_parallel_degree),
            ("context_parallel_degree", parallelism.context_parallel_degree),
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

    # Decoder._apply_fsdp is the only path that passes fsdp_backend on, so a
    # model that overrides it would shard with FSDP2.
    model_cls = model_config._owner
    if getattr(model_cls, "_apply_fsdp", None) is not Decoder._apply_fsdp:
        name = (
            type(model_config).__qualname__ if model_cls is None else model_cls.__name__
        )
        unsupported.append(f"models that override Decoder._apply_fsdp ({name})")
    if isinstance(model_config, Decoder.Config) and model_config.enable_weight_tying:
        # TODO: Parallelization unties tok_embeddings and lm_head, and only
        # init_states ties them again, after FlexShard has captured both.
        # Allow tied weights once _parallelize keeps them tied.
        unsupported.append("enable_weight_tying")
    # FlexShard's Shard placement all-gathers a bucket in one dtype.
    if any(
        module_config.preserve_dtype_params
        for _, module_config, _, _ in model_config.traverse(Module.Config, recurse=True)
    ):
        unsupported.append("preserve_dtype_params")
    # FlexShard stores parameters as plain tensors, which would drop the
    # quantized all-gather that MXFP8 and NVFP4 weights implement for FSDP2.
    if has_quantization(model_config):
        unsupported.append("MXFP8 and NVFP4 weights")
    if parallelism.expert_parallel_degree > 1:
        # Only RoutedExperts with AllToAllTokenDispatcher has been checked
        # against FSDP2 under EP. FlexShard's expert bucket treats every
        # routed-expert parameter as split over ep, but output_postprocess
        # parameters are replicated over ep (models/common/moe/SHARDING.md),
        # so their grads would never be summed over it.
        for _, moe, _, _ in model_config.traverse(MoE.Config):
            routed_experts = moe.routed_experts
            if type(routed_experts) is not RoutedExperts.Config:
                feature = type(routed_experts).__qualname__
            elif (
                type(routed_experts.token_dispatcher)
                is not AllToAllTokenDispatcher.Config
            ):
                feature = type(routed_experts.token_dispatcher).__qualname__
            elif routed_experts.output_postprocess is not None:
                feature = "routed-expert output_postprocess"
            else:
                continue
            entry = f"{feature} with expert parallelism"
            if entry not in unsupported:
                unsupported.append(entry)

    if (
        not config.training.disable_accelerator_graphs
        and accelerator_graphs_supported()
    ):
        unsupported.append(
            "accelerator graphs (set training.disable_accelerator_graphs=True)"
        )
    if isinstance(config.loss, ChunkedLossWrapper.Config):
        unsupported.append("ChunkedLossWrapper")
    if config.checkpointer is not None:
        unsupported.append("checkpointing")
    if config.optim.ema is not None:
        unsupported.append("optim.ema")
    if any(isinstance(o, DistMuon.Config) for o in config.optim.optimizer.optimizers):
        # DistMuon needs DTensor parameters.
        unsupported.append("DistMuon")
    if config.debug.spmd_typechecking:
        unsupported.append("debug.spmd_typechecking")

    # Other trainers shard, reduce or publish parameters in ways that FlexShard
    # doesn't support yet: GraphTrainer shards with SimpleFSDP, TorchFT's
    # cross-replica all-reduce hooks FSDP2's parameter groups, and the RL
    # trainer publishes state_dict(), which holds local shards, as full weights.
    trainer = type(config)._owner
    if trainer is not Trainer:
        name = (
            "None"
            if trainer is None
            else f"{trainer.__module__}.{trainer.__qualname__}"
        )
        unsupported.append(f"trainers other than torchtitan.trainer.Trainer ({name})")
    if unsupported:
        raise ValueError(
            "parallelism.fsdp_backend=FlexShardConfig() does not support "
            f"{', '.join(unsupported)} yet."
        )
    # Checked here because flex_shard is imported only once the model is
    # sharded, after distributed init and the model build.
    if find_spec("flex_shard") is None:
        raise ImportError(
            "parallelism.fsdp_backend=FlexShardConfig() needs flex_shard "
            "(github.com/meta-pytorch/flex_shard), which isn't installed. "
            ".github/workflows/unit_test_gpu.yaml installs the commit torchtitan "
            "is tested with."
        )
