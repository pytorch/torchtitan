# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Streamed microbatches give the optimizer the same gradients as one forward_backward call.

The step has a power-of-two loss-token count N, so dividing by N is exact in any float format, and
both orders (divide each loss, or divide the summed gradients) give bitwise equal gradients.

    pytest tests/rl/unit_tests/gpu/test_stream_microbatches.py
"""

import dataclasses
import os
from functools import partial

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.elastic.utils.distributed import get_free_port

from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.utils import get_local_tensor
from torchtitan.models.qwen3 import build_model_config
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.rollout.types import RolloutTurnID
from torchtitan.rl.trainer.batcher import Batcher
from torchtitan.rl.trainer.losses.grpo import GRPOLoss
from torchtitan.rl.trainer.trainer import Trainer
from torchtitan.rl.trainer.training_sample_builder import (
    TrainingSample,
    TrainingSampleGroup,
)
from torchtitan_recipes.rl.alphabet_sort import rl_grpo_qwen3_0_6b_varlen_no_compile
from torchtitan_recipes.rl.model_configs import build_rl_model_configs

_NUM_TOKENS_PER_MICROBATCH = 64

# world size, trainer parallelism
_LAYOUTS = {
    "one_gpu": (1, ParallelismConfig()),
    # HSDP all-reduces after the step's last group; PP finalizes gradients per schedule step.
    "hsdp": (2, ParallelismConfig(data_parallel_replicate_degree=2)),
    "pp": (
        2,
        ParallelismConfig(
            pipeline_parallel_degree=2,
            pipeline_parallel_schedule="1F1B",
            num_pp_microbatches=2,
        ),
    ),
}


def _trainer_config(parallelism: ParallelismConfig) -> Trainer.Config:
    trainer_model, _ = build_rl_model_configs(
        build_model_config,
        "debugmodel",
        trainer_parallelism=parallelism,
        generator_parallelism=InferenceParallelismConfig(),
        seq_len=_NUM_TOKENS_PER_MICROBATCH,
        attn_backend="varlen",
    )
    if parallelism.pipeline_parallel_degree > 1:
        # Tied embeddings cannot be split across pipeline stages. Untied, the embedding needs its
        # own init: the tied output layer initialized it.
        trainer_model.enable_weight_tying = False
        trainer_model.tok_embeddings.param_init = {
            "weight": partial(torch.nn.init.normal_, std=0.02)
        }
    config = rl_grpo_qwen3_0_6b_varlen_no_compile(seq_len=_NUM_TOKENS_PER_MICROBATCH)
    return dataclasses.replace(
        config.trainer,
        model=trainer_model,
        parallelism=parallelism,
        training=dataclasses.replace(
            config.trainer.training,
            num_tokens_per_microbatch_per_dp_rank=_NUM_TOKENS_PER_MICROBATCH,
            max_context_length=_NUM_TOKENS_PER_MICROBATCH,
        ),
        loss=ChunkedLossWrapper.Config(
            num_chunks=2,
            loss_fn=GRPOLoss.Config(global_vocab_size=trainer_model.vocab_size),
        ),
        checkpointer=None,
    )


def _training_sample_group(rng: np.random.Generator) -> TrainingSampleGroup:
    """24 samples of 8 to 60 tokens with 512 loss tokens in total."""
    samples = []
    for rollout_id, num_tokens in enumerate(rng.integers(8, 61, size=24)):
        samples.append(
            TrainingSample(
                min_policy_version=0,
                max_policy_version=0,
                rollout_id=RolloutTurnID(group_id=0, rollout_id=rollout_id, turn_id=0),
                token_ids=rng.integers(0, 2048, size=num_tokens),
                loss_mask=np.arange(num_tokens) >= num_tokens // 3,
                logprobs=-rng.uniform(5.0, 9.0, size=num_tokens).astype(np.float32),
                advantage=np.full(num_tokens, rng.choice([-1.0, 1.0]), np.float32),
            )
        )
    # Drop loss tokens until the step has 512.
    num_extra_loss_tokens = (
        sum(int(np.count_nonzero(sample.loss_mask[1:])) for sample in samples) - 512
    )
    for sample in samples:
        while num_extra_loss_tokens > 0 and np.count_nonzero(sample.loss_mask[1:]) > 1:
            sample.loss_mask[np.flatnonzero(sample.loss_mask)[0]] = False
            num_extra_loss_tokens -= 1
    return TrainingSampleGroup(group_id=0, training_samples=samples, metrics=[])


def _run_streamed_vs_one_call(
    rank: int, world_size: int, port: int, layout: str, output_dir: str
) -> None:
    os.environ.update(
        MASTER_ADDR="localhost",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    parallelism = _LAYOUTS[layout][1]
    trainer = Trainer(
        _trainer_config(parallelism), max_num_documents=None, output_dir=output_dir
    )
    engine = trainer.engine
    batch, _ = (
        Batcher.Config()
        .build(
            num_tokens_per_microbatch_per_dp_rank=_NUM_TOKENS_PER_MICROBATCH,
            max_context_length=_NUM_TOKENS_PER_MICROBATCH,
            num_prompts_per_train_step=1,
            dp_degree=trainer.dp_size,
            num_pp_microbatches=parallelism.num_pp_microbatches,
            pad_id=0,
            temperature=1.0,
        )
        .add_training_samples(
            training_sample_group=_training_sample_group(np.random.default_rng(0))
        )
    )
    assert batch is not None
    assert batch.global_loss_token_counts.tolist() == [512]
    num_groups = len(batch.microbatch_groups)
    assert num_groups >= 2
    column = [
        [row[trainer.dp_rank] for row in group] for group in batch.microbatch_groups
    ]
    first, second = column[: num_groups // 2], column[num_groups // 2 :]

    # Record the gradients the optimizer would clip, without updating the weights.
    optimizer_gradients = []
    engine.optim.step = lambda loss, *, current_step: optimizer_gradients.append(
        [get_local_tensor(p.grad).clone() for p in engine.optim.parameters]
    )

    # 1. Today: one call; each loss is divided by the step's N loss tokens.
    engine.forward_backward(
        microbatch_groups=column,
        global_loss_token_counts=batch.global_loss_token_counts,
        global_routing_token_counts=batch.global_routing_token_counts,
    )
    engine.optim_step()
    # 2. Streamed: two calls of summed losses; the optimizer step divides by N.
    engine.forward_backward(
        microbatch_groups=first,
        global_loss_token_counts=None,
        global_routing_token_counts=None,
        ends_step=False,
    )
    engine.forward_backward(
        microbatch_groups=second,
        global_loss_token_counts=None,
        global_routing_token_counts=None,
    )
    engine.optim_step(global_loss_token_counts=batch.global_loss_token_counts)

    one_call, streamed = optimizer_gradients
    for expected, actual in zip(one_call, streamed, strict=True):
        assert torch.equal(actual, expected)
    dist.destroy_process_group()


@pytest.mark.parametrize(
    "layout",
    [
        pytest.param(layout, marks=[pytest.mark.multi_gpu] if world_size > 1 else [])
        for layout, (world_size, _) in _LAYOUTS.items()
    ],
)
def test_streamed_microbatches_match_one_forward_backward(
    layout: str, tmp_path
) -> None:
    world_size = _LAYOUTS[layout][0]
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    mp.spawn(
        _run_streamed_vs_one_call,
        args=(world_size, get_free_port(), layout, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )
