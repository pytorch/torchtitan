# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""optim_step divides the summed gradients once by the step's loss-token count N.

Losses are sums. N is 509 here, not a power of two, so dividing by a wrong count changes the
gradients. FSDP 2 with the batcher's uneven per-rank loss tokens matches 1 GPU running both
ranks' microbatches: optim_step sums the count over the DP ranks.

    pytest tests/rl/unit_tests/gpu/test_loss_token_count.py -m multi_gpu    # 2 GPUs
"""

import dataclasses
import os

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    StateDictOptions,
)
from torch.distributed.elastic.utils.distributed import get_free_port
from torch.distributed.tensor import DTensor

from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.config import PrecisionConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.utils import get_local_tensor
from torchtitan.models.common.aux_loss import aux_loss_denominators
from torchtitan.models.qwen3 import build_model_config
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.rollout.types import RolloutTurnID
from torchtitan.rl.trainer.batcher import Batcher, TrainerStepBatch
from torchtitan.rl.trainer.losses.grpo import GRPOLoss
from torchtitan.rl.trainer.trainer import Trainer
from torchtitan.rl.trainer.training_sample_builder import (
    TrainingSample,
    TrainingSampleGroup,
)
from torchtitan_recipes.rl.alphabet_sort import rl_grpo_qwen3_0_6b_varlen_no_compile
from torchtitan_recipes.rl.model_configs import build_rl_model_configs

_NUM_TOKENS_PER_MICROBATCH = 64
_NUM_LOSS_TOKENS = 509


def _trainer_config(
    parallelism: ParallelismConfig,
) -> tuple[Trainer.Config, PrecisionConfig]:
    trainer_model, _ = build_rl_model_configs(
        build_model_config,
        "debugmodel",
        trainer_parallelism=parallelism,
        generator_parallelism=InferenceParallelismConfig(),
        seq_len=_NUM_TOKENS_PER_MICROBATCH,
        attn_backend="varlen",
    )
    config = rl_grpo_qwen3_0_6b_varlen_no_compile(seq_len=_NUM_TOKENS_PER_MICROBATCH)
    trainer_config = dataclasses.replace(
        config.trainer,
        model=trainer_model,
        parallelism=parallelism,
        training=dataclasses.replace(
            config.trainer.training,
            num_tokens_per_microbatch_per_dp_rank=_NUM_TOKENS_PER_MICROBATCH,
        ),
        loss=ChunkedLossWrapper.Config(
            num_chunks=2,
            loss_fn=GRPOLoss.Config(global_vocab_size=trainer_model.vocab_size),
        ),
        checkpointer=None,
    )
    return trainer_config, config.precision


def _training_sample_group(rng: np.random.Generator) -> TrainingSampleGroup:
    """24 samples of 8 to 60 tokens with _NUM_LOSS_TOKENS loss tokens in total."""
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
    num_extra_loss_tokens = (
        sum(int(np.count_nonzero(sample.loss_mask[1:])) for sample in samples)
        - _NUM_LOSS_TOKENS
    )
    for sample in samples:
        while num_extra_loss_tokens > 0 and np.count_nonzero(sample.loss_mask[1:]) > 1:
            sample.loss_mask[np.flatnonzero(sample.loss_mask)[0]] = False
            num_extra_loss_tokens -= 1
    return TrainingSampleGroup(group_id=0, training_samples=samples, metrics=[])


def _pack_batch(*, dp_degree: int) -> TrainerStepBatch:
    batch, _ = (
        Batcher.Config()
        .build(
            num_tokens_per_microbatch_per_dp_rank=_NUM_TOKENS_PER_MICROBATCH,
            max_context_length=_NUM_TOKENS_PER_MICROBATCH,
            num_prompts_per_train_step=1,
            dp_degree=dp_degree,
            num_pp_microbatches=1,
            pad_id=0,
            temperature=1.0,
        )
        .add_training_samples(
            training_sample_group=_training_sample_group(np.random.default_rng(0))
        )
    )
    assert batch is not None
    assert batch.num_loss_tokens == _NUM_LOSS_TOKENS
    return batch


def _record_optimizer_gradients(trainer: Trainer) -> list[dict[str, torch.Tensor]]:
    """Make the optimizer record the gradients it would clip, by parameter name, instead of
    updating the weights."""
    engine = trainer.engine
    named_parameters = [
        (name, parameter)
        for model_part in engine.model_parts
        for name, parameter in model_part.named_parameters()
        if parameter.requires_grad
    ]
    gradients: list[dict[str, torch.Tensor]] = []
    engine.optim.step = lambda loss, *, current_step: gradients.append(
        {
            name: get_local_tensor(parameter.grad).clone()
            for name, parameter in named_parameters
        }
    )
    return gradients


def _run_dp_ranks_on_one_or_two_gpus(
    rank: int, world_size: int, port: int, output_dir: str
) -> None:
    """world_size 1 runs both DP ranks' microbatches; world_size 2 runs FSDP 2."""
    parallelism = (
        ParallelismConfig()
        if world_size == 1
        else ParallelismConfig(data_parallel_shard_degree=2)
    )
    os.environ.update(
        MASTER_ADDR="localhost",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    trainer_config, precision = _trainer_config(parallelism)
    trainer = Trainer(
        trainer_config,
        precision=precision,
        max_num_documents=None,
        output_dir=output_dir,
    )
    model = trainer.engine.model_parts[0]
    batch = _pack_batch(dp_degree=2)
    columns = [
        [[row[dp_rank] for row in group] for group in batch.microbatch_groups]
        for dp_rank in range(2)
    ]
    loss_tokens_by_rank = [
        sum(int(group[0].loss_token_count) for group in column) for column in columns
    ]
    # Uneven counts: dividing by a rank's own count would differ from the step's count.
    assert loss_tokens_by_rank[0] != loss_tokens_by_rank[1]
    weights_path = os.path.join(output_dir, "weights.pt")
    if world_size == 1:
        torch.save(
            get_model_state_dict(model, options=StateDictOptions(full_state_dict=True)),
            weights_path,
        )
        microbatch_groups = columns[0] + columns[1]
    else:
        set_model_state_dict(
            model,
            torch.load(weights_path, map_location="cpu", weights_only=True),
            options=StateDictOptions(full_state_dict=True),
        )
        microbatch_groups = columns[rank]
    gradients = _record_optimizer_gradients(trainer)

    trainer.engine.forward_backward(
        microbatch_groups=microbatch_groups,
        aux_loss_denominators=aux_loss_denominators(
            batch.routing_token_counts, torch.tensor(batch.num_loss_tokens)
        ),
    )
    trainer.engine.optim_step()

    assert int(trainer.engine.global_loss_token_count) == _NUM_LOSS_TOKENS
    (local_gradients,) = gradients
    full_gradients = {
        name: (
            parameter.grad.full_tensor()
            if isinstance(parameter.grad, DTensor)
            else local_gradients[name]
        )
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if rank == 0:
        torch.save(
            {name: grad.cpu() for name, grad in full_gradients.items()},
            os.path.join(output_dir, f"gradients_{world_size}.pt"),
        )
    dist.destroy_process_group()


@pytest.mark.multi_gpu
def test_fsdp_with_uneven_rank_counts_matches_one_gpu(tmp_path) -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs")
    for world_size in (1, 2):
        mp.spawn(
            _run_dp_ranks_on_one_or_two_gpus,
            args=(world_size, get_free_port(), str(tmp_path)),
            nprocs=world_size,
            join=True,
        )
    one_gpu = torch.load(tmp_path / "gradients_1.pt", weights_only=True)
    fsdp = torch.load(tmp_path / "gradients_2.pt", weights_only=True)
    assert one_gpu.keys() == fsdp.keys()
    for name, expected in one_gpu.items():
        # FSDP sums the two ranks' gradients in another order.
        torch.testing.assert_close(fsdp[name], expected, msg=name)
