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
``apply_fsdp_to_decoder`` wraps each FlexShard parameter's ``param_init``
function with flex_shard's ``init_from_full``: it fills the full parameter and
the shard keeps its part, as one device would initialize it, while each of
FSDP2's DTensor shards draws its own random values. ``flex_shard_grad_norm``
reduces a bucket's grad norm as DTensor does. Two more differences change only
memory and communication. FSDP2 skips the embedding's backward all-gather
(``set_unshard_in_backward``), which FlexShard can't yet, so it gathers the
embedding again in backward, with the same values. And an MoE block's expert
bucket hooks the routed experts, not the block, so FlexShard reshards them as
soon as the routed experts finish forward or backward, while FSDP2 keeps the
block's expert group unsharded until the whole block finishes.
``flex_shard_partial_backwards`` keeps ``lm_head``'s bucket unsharded across
``ChunkedLossWrapper``'s chunks, as ``fsdp_partial_backwards`` keeps FSDP2's
group. FlexShard is imported only when it is selected.

FlexShard's state dicts hold plain local shards too, so each declares where its
shards sit in the full tensors (DCP's ``CheckpointableTensor``): the model's,
through ``declare_state_dict_layouts``, and the optimizer's, through
``declare_optimizer_state_layouts``. DCP then saves the same checkpoints as for
FSDP2's DTensors, and either backend loads the other's.
"""

import math
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from importlib.util import find_spec
from typing import Any, cast, TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

if TYPE_CHECKING:
    from torchtitan.models.base.model import BaseModel
    from torchtitan.training_engine import TrainingEngine

__all__ = [
    "declare_optimizer_state_layouts",
    "declare_state_dict_layouts",
    "flex_shard_grad_norm",
    "flex_shard_partial_backwards",
    "has_flex_shard_bucket",
    "record_buckets",
    "record_grad_norm_mesh",
    "reshard_flex_shard",
    "validate_flex_shard_training",
]

# The mesh to reduce a grad norm over for the parameters of the FlexShard buckets
# on a mesh, when it isn't that mesh, for flex_shard_grad_norm. Keyed by the
# mesh the buckets hold, which keeps the entry alive as long as the model.
_GRAD_NORM_MESHES: weakref.WeakKeyDictionary[
    DeviceMesh, DeviceMesh
] = weakref.WeakKeyDictionary()

# For flex_shard_partial_backwards, reshard_flex_shard and
# declare_optimizer_state_layouts: the models FlexShard sharded, and for the
# first, the bucket of each module whose FSDP2 group is one bucket, e.g.
# lm_head's, which also holds the final norm. A bucket holds its model, which
# holds the module, so the bucket is referenced weakly to let the entry go.
_FLEX_SHARD_MODELS: weakref.WeakSet[nn.Module] = weakref.WeakSet()
_MODULE_BUCKETS: weakref.WeakKeyDictionary[
    nn.Module, weakref.ref[Any]
] = weakref.WeakKeyDictionary()


def record_grad_norm_mesh(bucket_mesh: DeviceMesh, mesh: DeviceMesh) -> None:
    """Record that ``flex_shard_grad_norm`` reduces the grads of the parameters
    of the FlexShard buckets on ``bucket_mesh`` over the 1D ``mesh``.

    For parameters that an outer sharding split over more ranks before FlexShard
    sharded each part over ``bucket_mesh``, e.g. routed experts under expert
    parallelism.
    """
    # Meshes compare by value. An equal mesh that an earlier model recorded
    # would stay the key, and the entry would go when that model does.
    _GRAD_NORM_MESHES.pop(bucket_mesh, None)
    _GRAD_NORM_MESHES[bucket_mesh] = mesh


def flex_shard_grad_norm(
    params: list[torch.Tensor],
    mesh: DeviceMesh,
    norm_type: float,
    error_if_nonfinite: bool,
    foreach: bool | None,
) -> torch.Tensor:
    """The norm of the grads of FlexShard parameters ``params``, whose buckets
    are on the 1D ``mesh`` (``flex_shard.get_mesh``).

    The grads are plain local shards over ``mesh``. This reduces them over
    ``mesh``, or over the 1D mesh ``record_grad_norm_mesh`` recorded for it, in
    the order ``get_total_norm`` does for DTensor grads sharded over that mesh,
    so the result is bitwise equal: each grad's local norm, reduced over the
    mesh, then the norm of those norms.
    """
    mesh = _GRAD_NORM_MESHES.get(mesh, mesh)
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


def record_buckets(model: nn.Module, module_buckets: dict[nn.Module, Any]) -> None:
    """Record that FlexShard sharded ``model``, and the bucket storage of each
    module in ``module_buckets``, for ``flex_shard_partial_backwards``,
    ``reshard_flex_shard`` and ``declare_optimizer_state_layouts``. Call it
    after ``flex_shard(model)``."""
    _FLEX_SHARD_MODELS.add(model)
    for module, bucket in module_buckets.items():
        _MODULE_BUCKETS[module] = weakref.ref(bucket)


def has_flex_shard_bucket(module: nn.Module) -> bool:
    """Whether FlexShard sharded ``module``'s FSDP2 group as one bucket, as it
    does ``lm_head``'s, for ``flex_shard_partial_backwards``."""
    return module in _MODULE_BUCKETS


@contextmanager
def flex_shard_partial_backwards(
    module: nn.Module, *, run_backward: bool, outer_backward_follows: bool
) -> Iterator[Callable[[], None]]:
    """``fsdp_partial_backwards`` for a module with a FlexShard bucket, with
    the collectives FSDP2 issues for its group. Yields ``before_last_forward``.

    A call of ``module`` on its own leaves its bucket's forward open, so
    FlexShard runs no post-forward for it and finishes the bucket as soon as
    the call's input grads are ready: with gradient sync, it reduce-scatters
    and reshards; without, it keeps the grads, and reshards only with
    ``reshard_after_backward``. That backward finishes no other bucket, so
    unlike FSDP2, the other buckets need no changes. Gradient sync is on
    outside, since ``validate_flex_shard_training`` rejects deferred gradient
    reduction, so the last call's backward reduce-scatters the grads of all
    calls, as FSDP2's does.

    Without backward, the bucket's forward stays open, as FSDP2 leaves its
    group's; ``post_validate_fsdp_for_accelerator_graph`` closes both.
    """
    if run_backward and not outer_backward_follows:
        # Calls whose inputs don't require grad have no trigger that finishes
        # the bucket. FSDP2 completes each call's grouped forward instead.
        raise ValueError(
            "parallelism.fsdp_backend=FlexShardConfig() does not support "
            "ChunkedLossWrapper with a frozen decoder yet."
        )
    bucket = _MODULE_BUCKETS[module]()
    assert bucket is not None, "FlexShard buckets live as long as their model."
    reshard_after_backward = bucket.reshard_after_backward
    # TODO: Gradient sync is set to True, not restored, because FlexShard has
    # no public getter for it, and validate_flex_shard_training rejects
    # deferred gradient reduction, so it's always on outside. Allowing deferral
    # needs the caller's setting restored, and FlexShard's state of grads kept
    # without sync, to leave the last reduce-scatter to the decoder backward
    # when earlier microbatches kept grads, as FSDP2 does.
    try:
        bucket.set_requires_gradient_sync(False)
        bucket.set_reshard_after_backward(False)
        yield lambda: bucket.set_requires_gradient_sync(True)
    finally:
        bucket.set_requires_gradient_sync(True)
        bucket.set_reshard_after_backward(reshard_after_backward)
        # The last call's backward has resharded the bucket, unless there was
        # no backward.
        bucket.reshard()


def reshard_flex_shard(model: nn.Module) -> None:
    """If FlexShard sharded ``model``, reshard all its buckets and close the
    forwards they left open, with FlexShard's ``reshard()``."""
    if model in _FLEX_SHARD_MODELS:
        cast(Any, model).reshard()


def declare_state_dict_layouts(model: nn.Module) -> None:
    """Make ``model.state_dict()`` declare where each FlexShard parameter's
    local shard sits in the full parameter. Call it after ``flex_shard(model)``.

    Without the layouts, DCP saves and loads each local shard as the whole
    parameter. Every state dict ``CheckpointManager`` saves or loads comes from
    ``model.state_dict()``, including the one it takes again after a load.
    """
    # pyrefly: ignore [missing-import]
    from flex_shard import set_state_dict_global_layouts

    # TODO: Drop once a FlexShard module's state_dict() declares its layouts.
    def hook(module, state_dict, prefix, local_metadata) -> None:
        set_state_dict_global_layouts(
            module,
            {
                key.removeprefix(prefix): value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            },
        )

    model.register_state_dict_post_hook(hook)


def declare_optimizer_state_layouts(
    optimizer: torch.optim.Optimizer, model: nn.Module
) -> None:
    """Make ``optimizer.state_dict()`` declare where each state of a FlexShard
    parameter sits in the full state, if FlexShard sharded ``model``, which
    holds the parameters.

    ``step`` declares no layout, so DCP saves it as replicated, as for FSDP2.
    flex_shard raises ``NotImplementedError`` for optimizers other than Adam
    and AdamW.
    """
    if model in _FLEX_SHARD_MODELS:
        # pyrefly: ignore [missing-import]
        from flex_shard import register_optimizer_checkpoint_hook

        register_optimizer_checkpoint_hook(optimizer, model)


def validate_flex_shard_training(
    config: "TrainingEngine.Config", model_config: "BaseModel.Config"
) -> None:
    """Reject what FlexShard doesn't support yet, before anything is built.

    ``TrainingEngine.__init__`` calls this for every trainer, with the config
    after recipes and overrides.
    """
    from torchtitan.components.checkpointer import CheckpointManager
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
        # TODO: When allowing it, change flex_shard_partial_backwards, which
        # sets lm_head's bucket's gradient sync to True instead of restoring it.
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
    # FlexShard parameters are plain local shards, and only param_init
    # initializes them as one device would: apply_fsdp_to_decoder wraps it with
    # init_from_full. reset_parameters() would draw every rank's shard as the
    # whole parameter, with the same values.
    if any(
        module_config.param_init is None
        and callable(getattr(module_config._owner, "reset_parameters", None))
        for _, module_config, _, _ in model_config.traverse(Module.Config, recurse=True)
    ):
        unsupported.append("modules initialized by reset_parameters() (set param_init)")
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
    if config.checkpointer is not None:
        # FlexShard's state dicts hold local shards that declare their layouts
        # to DCP. Each of these would lose the layouts and treat every rank's
        # shard as the whole tensor, without an error.
        checkpointer = config.checkpointer
        if not isinstance(checkpointer, CheckpointManager.Config):
            # TODO: Allow once torch_checkpointing's resharder reads the
            # layouts, not only DTensor placements.
            unsupported.append("checkpointers other than CheckpointManager")
        elif checkpointer.async_mode == "async_with_pinned_mem":
            # TODO: Allow once DCP's process-based async save keeps the
            # layouts, which torch.multiprocessing drops on the way to the save
            # process.
            unsupported.append("checkpointer.async_mode='async_with_pinned_mem'")
        # The state dict adapters need DTensor parameters.
        if checkpointer.initial_load_in_hf:
            unsupported.append("checkpointer.initial_load_in_hf=True")
        if checkpointer.last_save_in_hf:
            unsupported.append("checkpointer.last_save_in_hf=True")
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
