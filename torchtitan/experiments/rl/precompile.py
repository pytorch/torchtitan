# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compile-on-one-rank (CooR) precompile for the RL PolicyTrainer.

``save_rl_precompiled`` traces the RL forward+loss+backward step and applies the
compile-time graph passes (cleanup, graph-SAC, bucketing, and FlexAttention
``regional_inductor`` -- which bakes the compiled Triton kernels into the
artifact), then serializes it to disk with a config fingerprint. cudagraph is
excluded here; it is re-applied per rank at load time by ``RLTracedStep``.

At training time, set ``trainer_compile.precompile_artifact_dir`` and
``RLTracedStep`` loads the artifact instead of tracing (see
``RLTracedStep._load_precompiled``).

The single-process CooR driver (``main``) mirrors graph_trainer's
``precompile_main``: a fake process group plus ``compile_on_one_rank`` lets one
process emit a rank-agnostic artifact. NOTE: loading a CooR artifact on every
rank under Monarch still needs each trainer proc to address its GPU as
``cuda:0`` (torchrun does this via ``--virtual-local-rank``); that provisioning
change is tracked as follow-up work. The save path and same-process load are
exercised by ``tests/test_precompile_numerics.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.passes import (
    apply_graph_passes,
    compile_time_passes,
)
from torchtitan.experiments.graph_trainer.precompile import (
    compute_config_fingerprint,
    precompile_fx_trace_save,
)
from torchtitan.experiments.graph_trainer.storage import DiskStorageAdapter
from torchtitan.experiments.rl.actors.traced_step import (
    build_pass_config,
    trace_rl_fwd_bwd,
)
from torchtitan.tools.logging import logger


def save_rl_precompiled(
    model: nn.Module,
    loss_fn,
    *,
    compile_config: GraphTrainerCompileConfig,
    parallel_dims,
    parallelism,
    model_config,
    loss_config,
    train_context,
    token_ids: torch.Tensor,
    labels: torch.Tensor,
    positions: torch.Tensor,
    attention_masks: Any,
    generator_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    global_valid_tokens: int,
    device: torch.device,
) -> str:
    """Trace + compile-time passes + serialize the RL fwd+loss+bwd artifact.

    Returns the artifact path. ``compile_config.precompile_artifact_dir`` must be
    set. The dummy inputs must have the exact shapes/dtypes training will use
    (the graph bakes in fixed shapes); a real batch works well as the sample.
    """
    if not compile_config.precompile_artifact_dir:
        raise ValueError(
            "save_rl_precompiled requires compile_config.precompile_artifact_dir."
        )

    gvt = torch.tensor(float(global_valid_tokens), dtype=torch.float32, device=device)
    extra_kwargs: dict[str, Any] = {
        "attention_masks": attention_masks,
        "positions": positions,
        "generator_logprobs": generator_logprobs,
        "advantages": advantages,
        "loss_mask": loss_mask,
    }

    traced = trace_rl_fwd_bwd(
        model,
        loss_fn,
        token_ids=token_ids,
        labels=labels,
        global_valid_tokens=gvt,
        extra_kwargs=extra_kwargs,
        train_context=train_context,
    )
    logger.info(
        "RL precompile: traced graph has %d nodes, %d state entries",
        len(list(traced.gm.graph.nodes)),
        len(traced.state_fqns),
    )

    pass_config = build_pass_config(
        compile_config=compile_config,
        model_config=model_config,
        loss_config=loss_config,
        parallelism=parallelism,
        dump_folder=compile_config.precompile_artifact_dir,
    )
    # cudagraph is excluded: it is re-captured at load time on each rank.
    passes = compile_time_passes(
        traced, pass_config, use_cudagraph=False, parallel_dims=parallel_dims
    )
    traced.gm = apply_graph_passes(
        traced.gm, traced.example_inputs, passes, compile_config=compile_config
    )

    fingerprint = compute_config_fingerprint(model, compile_config, parallel_dims)
    storage = DiskStorageAdapter(compile_config.precompile_artifact_dir)
    path = precompile_fx_trace_save(traced, storage, config_fingerprint=fingerprint)
    logger.info("RL precompile artifact saved to %s", path)
    return path


def _build_dummy_batch(model, model_config, seq_len, batch, device):
    """A representative multi-document packed batch for the precompile trace.

    Shapes/dtypes must match training (the graph bakes fixed shapes). Positions
    reset to 0 per packed document, matching the RL batcher.
    """
    vocab = model_config.vocab_size
    num_docs = 4 if seq_len % 4 == 0 else 1
    doc_len = seq_len // num_docs
    token_ids = torch.randint(0, vocab, (batch, seq_len), device=device)
    labels = torch.randint(0, vocab, (batch, seq_len), device=device)
    positions = (
        torch.arange(doc_len, dtype=torch.int32, device=device)
        .repeat(num_docs)
        .expand(batch, seq_len)
        .contiguous()
    )
    generator_logprobs = torch.randn(batch, seq_len, device=device) * 0.1 - 2.0
    loss_mask = torch.zeros(batch, seq_len, dtype=torch.bool, device=device)
    loss_mask[:, doc_len // 2 :] = True
    advantages = torch.randn(batch, seq_len, device=device) * loss_mask
    gvt = int(loss_mask.sum().item())
    return token_ids, labels, positions, generator_logprobs, advantages, loss_mask, gvt


def main() -> None:
    """Single-process CooR precompile driver for the RL trainer.

    Mirrors graph_trainer's precompile_main: a fake process group plus
    ``compile_on_one_rank`` produces a rank-agnostic artifact from one GPU. Pass
    the SAME flavor / parallelism / dtypes / seq_len / batch the trainer uses so
    the fingerprint matches at load. Loading on all ranks under Monarch still
    needs the per-proc ``cuda:0`` mapping (follow-up).
    """
    import argparse
    import contextlib
    import dataclasses

    import torch.distributed as dist

    from torchtitan.components.loss import ChunkedLossWrapper
    from torchtitan.config import ParallelismConfig, TrainingConfig
    from torchtitan.distributed import ParallelDims, utils as dist_utils
    from torchtitan.experiments.graph_trainer.chunked_loss import (
        ChunkedLossWrapperWithParamGrads,
    )
    from torchtitan.experiments.rl.losses import GRPOLoss
    from torchtitan.experiments.rl.models.cast_linear import LMHeadCastConverter
    from torchtitan.experiments.rl.models.simple_fsdp_parallelize import (
        to_simple_fsdp_spec,
    )
    from torchtitan.models.qwen3 import model_registry as qwen3_model_registry
    from torchtitan.tools import utils
    from torchtitan.tools.logging import init_logger

    parser = argparse.ArgumentParser()
    parser.add_argument("--flavor", default="0.6B")
    parser.add_argument("--attn-backend", default="flex")
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--local-batch-size", type=int, required=True)
    parser.add_argument("--num-chunks", type=int, default=8)
    parser.add_argument("--dp-shard", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--mixed-precision-param", default="bfloat16")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--artifact-dir", required=True)
    args = parser.parse_args()

    init_logger()
    world_size = args.dp_shard * args.tp
    logger.info("RL CooR precompile: world_size=%d (fake PG)", world_size)
    dist.init_process_group("fake", rank=0, world_size=world_size)

    import torch.distributed.config as dist_config

    dist_config.compile_on_one_rank = True

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parallel_dims = ParallelDims(
        dp_replicate=1,
        dp_shard=args.dp_shard,
        cp=1,
        tp=args.tp,
        pp=1,
        ep=1,
        world_size=world_size,
    )
    parallel_dims.build_mesh()

    training = TrainingConfig(
        seq_len=args.seq_len,
        mixed_precision_param=args.mixed_precision_param,
        mixed_precision_reduce="float32",
        dtype=args.dtype,
    )
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=args.dp_shard, tensor_parallel_degree=args.tp
    )
    config_like = SimpleNamespace(parallelism=parallelism, training=training)

    spec = qwen3_model_registry(
        args.flavor,
        attn_backend=args.attn_backend,
        converters=[LMHeadCastConverter.Config()],
    )
    spec = to_simple_fsdp_spec(spec)
    spec.model.update_from_config(config=config_like)
    for layer_cfg in spec.model.layers:
        attn = getattr(layer_cfg, "attention", None)
        if attn is not None:
            attn.rope = dataclasses.replace(attn.rope, max_seq_len=args.seq_len)

    from torchtitan.config import TORCH_DTYPE_MAP

    with torch.device("meta"):
        with utils.set_default_dtype(TORCH_DTYPE_MAP[args.dtype]):
            model = spec.model.build()

    compile_config = GraphTrainerCompileConfig(
        enable=True, mode="aot_fx_trace", precompile_artifact_dir=args.artifact_dir
    )
    model = spec.parallelize_fn(
        model,
        parallel_dims=parallel_dims,
        training=training,
        parallelism=parallelism,
        compile_config=compile_config,
        ac_config=None,
        dump_folder=args.artifact_dir,
    )
    model.to_empty(device=device.type)
    # DTensor RNG (weight init) raises under compile_on_one_rank; disable it for
    # init only, matching graph_trainer's precompile_main.
    dist_config.compile_on_one_rank = False
    try:
        with torch.no_grad():
            model.init_weights(buffer_device=None)
    finally:
        dist_config.compile_on_one_rank = True
    model.train()

    loss_fn = ChunkedLossWrapperWithParamGrads.Config(
        num_chunks=args.num_chunks, loss_fn=GRPOLoss.Config()
    ).build()
    loss_fn.set_lm_head(model.lm_head)
    model._skip_lm_head = True

    tk, lb, pos, glp, adv, lm, gvt = _build_dummy_batch(
        model, spec.model, args.seq_len, args.local_batch_size, device
    )
    loss_parallel_ctx = (
        torch.distributed.tensor.parallel.loss_parallel()
        if parallel_dims.tp_enabled
        else contextlib.nullcontext()
    )
    train_context = dist_utils.get_spmd_context(
        parallel_dims=parallel_dims, spmd_typechecking=False
    )
    with loss_parallel_ctx:
        save_rl_precompiled(
            model,
            loss_fn,
            compile_config=compile_config,
            parallel_dims=parallel_dims,
            parallelism=parallelism,
            model_config=spec.model,
            loss_config=ChunkedLossWrapper.Config(
                num_chunks=args.num_chunks, loss_fn=GRPOLoss.Config()
            ),
            train_context=train_context,
            token_ids=tk,
            labels=lb,
            positions=pos,
            attention_masks=model.get_attention_masks(pos),
            generator_logprobs=glp,
            advantages=adv,
            loss_mask=lm,
            global_valid_tokens=gvt,
            device=device,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
