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

FlexShard parameters and grads are plain local shards, with no DTensor.
``apply_fsdp_to_decoder`` wraps each FlexShard parameter's ``param_init``
function with flex_shard's ``init_from_full``: it fills the full parameter and
the shard keeps its part, as one device would initialize it, while each of
FSDP2's DTensor shards draws its own random values. ``flex_shard_grad_norm_mesh``
gives the mesh a grad's norm is reduced over. One more difference changes only
memory and communication: FSDP2 skips the embedding's backward all-gather
(``set_unshard_in_backward``), which FlexShard can't yet, so it gathers the
embedding again in backward, with the same values. FlexShard is imported only
when it is selected.
"""

import sys
from importlib.util import find_spec
from typing import TYPE_CHECKING

import torch
from torch.distributed.device_mesh import DeviceMesh

if TYPE_CHECKING:
    from torchtitan.models.base.model import BaseModel
    from torchtitan.training_engine import TrainingEngine

__all__ = [
    "flex_shard_grad_norm_mesh",
    "validate_flex_shard_training",
]


def flex_shard_grad_norm_mesh(param: torch.Tensor) -> DeviceMesh | None:
    """The mesh ``param`` is sharded over if it's a FlexShard parameter, else None.

    A FlexShard parameter and its grad are plain local shards over the mesh of
    their bucket.
    """
    # FlexShard parameters exist only once flex_shard is imported, which
    # happens only when it is selected.
    flex_shard = sys.modules.get("flex_shard")
    return None if flex_shard is None else flex_shard.get_mesh(param)


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
