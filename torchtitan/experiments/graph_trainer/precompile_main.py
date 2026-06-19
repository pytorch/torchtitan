# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Single-process precompile entry point for graph_trainer.

Uses compile-on-one-rank (CooR) to generate a rank-agnostic compiled
artifact from a single process, which can then be loaded by all ranks
during torchrun training. This avoids the need to run torchrun with N
GPUs just for precompilation.

Usage (aot_fx_trace mode):
    python -m torchtitan.experiments.graph_trainer.precompile_main \
        --module graph_trainer.llama3 \
        --config graph_trainer_llama3_debugmodel \
        --compile.mode aot_fx_trace \
        --compile.precompile_artifact_dir /tmp/fx_trace_artifacts
"""

import contextlib
from typing import Any, cast

import torch
import torch.distributed as dist

from torchtitan.components.loss import ChunkedLoss
from torchtitan.config import ConfigManager, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.common_utils import (
    maybe_register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.precompile import _FX_TRACE_ARTIFACT_KEY
from torchtitan.experiments.graph_trainer.storage import DiskStorageAdapter
from torchtitan.models.common.attention import FlexAttention, VarlenAttention
from torchtitan.models.common.decoder import Decoder
from torchtitan.tools import utils
from torchtitan.tools.logging import logger


def _common_setup(config):
    """Common setup for precompile: fake PG, CooR, model build."""
    compile_config = config.compile

    if not compile_config.precompile_artifact_dir:
        raise ValueError(
            "precompile_main requires --compile.precompile_artifact_dir to be set."
        )

    parallelism = config.parallelism
    dp_replicate = parallelism.data_parallel_replicate_degree
    dp_shard = parallelism.data_parallel_shard_degree
    cp = parallelism.context_parallel_degree
    tp = parallelism.tensor_parallel_degree
    pp = parallelism.pipeline_parallel_degree

    # dp_shard=-1 means "use remaining ranks" which can't be inferred
    # in single-process mode. The compiled graph bakes in tensor shapes
    # that depend on dp_shard, so the exact value must match training.
    if dp_shard < 0:
        raise ValueError(
            "precompile_main requires an explicit "
            "--parallelism.data_parallel_shard_degree (not -1). "
            "Set it to the value you will use during torchrun training."
        )
    world_size = dp_replicate * dp_shard * cp * tp * pp

    logger.info(f"Initializing single-process precompile with world_size={world_size}")

    # rank must be 0 because --virtual-local-rank maps every torchrun rank
    # to local rank 0, so the precompiled artifact needs to match that setup.
    # Fake backend produces correct collective output shapes without real
    # communication, letting us trace distributed ops on a single process.
    dist.init_process_group("fake", rank=0, world_size=world_size)

    # CooR must be enabled globally (not just during tracing) so that the
    # parallelization phase (TP, FSDP mesh setup) also uses symbolic
    # coordinates rather than hardcoding rank-specific values.
    import torch.distributed.config as dist_config

    dist_config.compile_on_one_rank = True

    # Match the deterministic mode that the training loop will use.
    # The backward graph captures use_deterministic_algorithms() at
    # compile time and asserts it matches at runtime.
    if config.debug.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parallel_dims = ParallelDims(
        dp_shard=dp_shard,
        dp_replicate=dp_replicate,
        cp=cp,
        tp=tp,
        pp=pp,
        ep=parallelism.expert_parallel_degree,
        world_size=world_size,
    )
    parallel_dims.build_mesh()

    model_spec = config.model_spec
    if model_spec is None:
        raise ValueError(
            "model_spec must be set. Pass --module to specify the model "
            "(e.g. --module graph_trainer.llama3)."
        )

    # TODO: Factor the model setup below with the training path so precompile
    # and training share a single implementation of build/parallelize/init.
    model_config = model_spec.model
    model_config.update_from_config(config=config)

    logger.info(f"Building {model_spec.name} {model_spec.flavor} on meta device")
    with (
        torch.device("meta"),
        utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]),
    ):
        model = model_config.build()

    model.verify_module_protocol()

    # For aot_fx_trace, apply_compile inside parallelize_fn is a no-op
    # (returns model unchanged), so we pass the real compile_config.
    model = model_spec.parallelize_fn(
        model,
        parallel_dims=parallel_dims,
        training=config.training,
        parallelism=parallelism,
        compile_config=compile_config,
        ac_config=config.activation_checkpoint,
        dump_folder=config.dump_folder,
    )

    # CooR must be disabled during init_weights because DTensor RNG ops
    # (weight initialization seeding) raise NotImplementedError under
    # compile_on_one_rank=True. Re-enable for the tracing phase after.
    device_type = utils.device_type
    model.to_empty(device=device_type)
    dist_config.compile_on_one_rank = False
    try:
        with torch.no_grad():
            model.init_weights(buffer_device=None)
    finally:
        dist_config.compile_on_one_rank = True
    model.train()

    logger.info("Model parallelized and materialized")

    tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)

    return (
        model,
        model_config,
        model_spec,
        compile_config,
        parallel_dims,
        device,
        tokenizer,
    )


def _prepare_loss_for_precompile(model, loss_fn) -> None:
    """Match Trainer's post-parallelization loss setup for precompile tracing."""
    if not isinstance(loss_fn, ChunkedLoss):
        return

    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise ValueError("Model must have lm_head for ChunkedLoss precompile")

    loss_fn.set_lm_head(lm_head)
    model._skip_lm_head = True


def _precompile_aot_fx_trace(
    config,
    model,
    model_config,
    model_spec,
    compile_config,
    parallel_dims,
    device,
    tokenizer,
):
    """aot_fx_trace mode precompilation: make_fx tracing + Inductor."""
    from torchtitan.experiments.graph_trainer.make_fx_tracer import minimal_fx_tracer
    from torchtitan.experiments.graph_trainer.precompile import (
        compute_config_fingerprint,
        precompile_fx_trace_save,
    )
    from torchtitan.experiments.graph_trainer.trainer import make_fwd_bwd_step

    loss_fn = config.loss.build(compile_config=compile_config)
    _prepare_loss_for_precompile(model, loss_fn)

    fwd_bwd_fn = make_fwd_bwd_step(model, loss_fn)

    seq_len = config.training.seq_len
    local_batch_size = config.training.local_batch_size
    vocab_size = model_config.vocab_size

    dummy_inputs = torch.randint(
        0, vocab_size, (local_batch_size, seq_len), device=device
    )
    dummy_labels = torch.randint(
        0, vocab_size, (local_batch_size, seq_len), device=device
    )
    # The trainer computes global_valid_tokens via dist_sum (an
    # all-reduce + .item()), which returns a Python float. Use the
    # same type here so make_fx bakes it as a graph constant — not a
    # graph input — identical to the non-precompile runtime trace.
    global_batch_size = (
        local_batch_size
        * parallel_dims.dp_shard
        * parallel_dims.dp_replicate
        * parallel_dims.cp
    )
    dummy_global_valid_tokens = float(global_batch_size * seq_len)
    extra_kwargs: dict[str, Any] = {}

    if isinstance(model_config, Decoder.Config) and model_config.layers:
        attn_config = model_config.layers[0].attention
        inner_attention = attn_config.inner_attention

        positions = torch.arange(
            0, dummy_inputs.shape[1], dtype=torch.int32, device=dummy_inputs.device
        ).expand(dummy_inputs.shape)

        if isinstance(inner_attention, (FlexAttention.Config, VarlenAttention.Config)):
            extra_kwargs["attention_masks"] = cast(Decoder, model).get_attention_masks(
                positions=positions,
            )

        extra_kwargs["positions"] = positions

    # TODO: Add CP support — call prepare_context_parallel_input here
    # to shard dummy_inputs/dummy_labels/extra_kwargs along the sequence
    # dimension, matching the trainer's post_dataloading_process.
    if parallel_dims.cp_enabled:
        raise NotImplementedError(
            "CooR precompile does not yet support context parallelism. "
            "Set --parallelism.context_parallel_degree 1."
        )

    loss_parallel_ctx = (
        # TODO(bobrenjc93): Migrate graph trainer to the manual loss-parallel
        # custom autograd function and remove this DTensor context manager.
        torch.distributed.tensor.parallel.loss_parallel()
        if parallel_dims.tp_enabled
        else contextlib.nullcontext()
    )

    maybe_register_blockmask_pytree_node()

    logger.info("Tracing fwd+loss+bwd via make_fx...")
    with loss_parallel_ctx:
        traced_result = minimal_fx_tracer(fwd_bwd_fn, module=model)(
            dummy_inputs,
            dummy_labels,
            dummy_global_valid_tokens,
            extra_kwargs,
        )
    logger.info(
        f"Traced graph has {len(list(traced_result.gm.graph.nodes))} nodes, "
        f"{len(traced_result.state_fqns)} state entries"
    )

    # Apply precompile-time graph passes (cleanup + regional_inductor)
    # so compiled Triton kernels are baked into the serialized artifact.
    # cudagraph is excluded — it runs at load time on each rank.
    from torchtitan.experiments.graph_trainer.passes import (
        apply_graph_passes,
        compile_time_passes,
    )

    passes = compile_time_passes(traced_result, config)

    traced_result.gm = apply_graph_passes(
        traced_result.gm, traced_result.example_inputs, passes
    )
    logger.info(
        f"Applied {len(passes)} precompile graph passes, "
        f"graph now has {len(list(traced_result.gm.graph.nodes))} nodes"
    )

    storage = DiskStorageAdapter(compile_config.precompile_artifact_dir)
    config_fingerprint = compute_config_fingerprint(
        model, compile_config, parallel_dims
    )

    precompile_fx_trace_save(
        traced_result,
        storage,
        config_fingerprint=config_fingerprint,
    )

    logger.info(
        f"Precompile complete. Artifact saved to "
        f"{compile_config.precompile_artifact_dir}/{_FX_TRACE_ARTIFACT_KEY}.bin"
    )


def main():
    config_manager = ConfigManager()
    config = config_manager.parse_args()

    mode = config.compile.mode
    if mode != "aot_fx_trace":
        raise ValueError(
            f"precompile_main only supports --compile.mode aot_fx_trace, "
            f"got '{mode}'."
        )

    (
        model,
        model_config,
        model_spec,
        compile_config,
        parallel_dims,
        device,
        tokenizer,
    ) = _common_setup(config)

    _precompile_aot_fx_trace(
        config,
        model,
        model_config,
        model_spec,
        compile_config,
        parallel_dims,
        device,
        tokenizer,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
