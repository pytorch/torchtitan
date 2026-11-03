# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Accelerator graphs give the same results as eager with several forward_backward calls per step.

A segment is the microbatch groups between two gradient reductions: one group with plain FSDP,
the whole call with HSDP or deferred reduction. The step's first segment replays the
first-segment graph, which allocates the gradients; later segments replay a graph per segment
length, which adds into them. Each case runs the same calls with and without graphs, then
compares the gradients the optimizer sees, the loss, the loss metrics, the grad norm and the
final weights with torch.equal.

Calls per step, e.g. [3, 2] = a call with 3 groups, then a call with 2. Step 4 has 3 loss tokens
with a non-finite generator logprob, and step 5 has a microbatch without loss tokens.

    pytest tests/rl/unit_tests/gpu/test_segment_graphs.py                 # 1 GPU
    pytest tests/rl/unit_tests/gpu/test_segment_graphs.py -m multi_gpu    # 2 GPUs
"""

import dataclasses
import os
import tempfile

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
from torchtitan.observability.sdc_replayer import SDCReplayer
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.rollout.types import RolloutTurnID
from torchtitan.rl.trainer.batcher import Batcher
from torchtitan.rl.trainer.losses.dapo import DAPOLoss
from torchtitan.rl.trainer.trainer import Trainer
from torchtitan.rl.trainer.training_sample_builder import (
    TrainingSample,
    TrainingSampleGroup,
)
from torchtitan_recipes.rl.alphabet_sort import rl_grpo_qwen3_0_6b_varlen_no_compile
from torchtitan_recipes.rl.model_configs import build_rl_model_configs

_NUM_TOKENS_PER_MICROBATCH = 256
_MAX_NUM_DOCUMENTS = 16

# The step's first call has 3 groups, so its first group replays the first-segment graph and
# the other 2 replay the 1-group later-segment graph.
_FSDP_CALLS = [[3, 2], [1, 3], [2, 1], [3], [2, 2], [1, 2]]
# Under HSDP or deferral, a segment is a whole call. The engine has one first-segment graph,
# whose input is the step's first call, so every step's first call has 1 group, as the RL
# trainer sends it. Later calls of 1, 2 and 3 groups each warm up, capture and replay.
_WHOLE_CALL_CALLS = [[1, 2, 1], [1, 3, 1], [1, 2, 3], [1, 1, 2], [1, 3], [1, 2, 1, 3]]


@dataclasses.dataclass(frozen=True)
class _Case:
    world_size: int
    parallelism: ParallelismConfig
    calls_per_step: list[list[int]]
    sdc_replay: bool
    """Check every call with 2 replays. Replays capture graphs too, from the restored state."""
    preprocess_lazily: bool = False


_CASES = {
    "fsdp": _Case(1, ParallelismConfig(), _FSDP_CALLS, sdc_replay=True),
    # A checked call preprocesses up front, so lazy preprocessing runs without SDC replay.
    # RL's layout: one-group segments, each read just before its replay.
    "fsdp_lazy": _Case(
        1, ParallelismConfig(), _FSDP_CALLS, sdc_replay=False, preprocess_lazily=True
    ),
    "deferral_lazy": _Case(
        1,
        ParallelismConfig(fsdp_defer_gradient_reduction=True),
        _WHOLE_CALL_CALLS,
        sdc_replay=False,
        preprocess_lazily=True,
    ),
    # A real reduce-scatter in each segment.
    "dp_shard_2": _Case(
        2, ParallelismConfig(data_parallel_shard_degree=2), _FSDP_CALLS, sdc_replay=True
    ),
    # The replica all-reduce in each whole-call segment.
    "dp_replicate_2": _Case(
        2,
        ParallelismConfig(data_parallel_replicate_degree=2),
        _WHOLE_CALL_CALLS,
        sdc_replay=True,
    ),
}


def _trainer_config(case: _Case, *, graphs: bool) -> Trainer.Config:
    trainer_model, _ = build_rl_model_configs(
        build_model_config,
        "debugmodel",
        trainer_parallelism=case.parallelism,
        generator_parallelism=InferenceParallelismConfig(),
        seq_len=_NUM_TOKENS_PER_MICROBATCH,
        attn_backend="varlen",
    )
    config = rl_grpo_qwen3_0_6b_varlen_no_compile(seq_len=_NUM_TOKENS_PER_MICROBATCH)
    return dataclasses.replace(
        config.trainer,
        model=trainer_model,
        parallelism=case.parallelism,
        training=dataclasses.replace(
            config.trainer.training,
            num_tokens_per_microbatch_per_dp_rank=_NUM_TOKENS_PER_MICROBATCH,
            max_context_length=_NUM_TOKENS_PER_MICROBATCH,
            disable_accelerator_graphs=not graphs,
            preprocess_microbatch_groups_lazily=case.preprocess_lazily,
        ),
        loss=ChunkedLossWrapper.Config(
            num_chunks=2,
            loss_fn=DAPOLoss.Config(
                ratio_clip_low=0.2,
                ratio_clip_high=0.28,
                global_vocab_size=trainer_model.vocab_size,
            ),
        ),
        sdc_replayer=(
            SDCReplayer.Config(num_steps=-1, num_replays=2) if case.sdc_replay else None
        ),
        # The same weights in every run; SDC replay needs deterministic kernels.
        debug=dataclasses.replace(config.trainer.debug, seed=0, deterministic=True),
        checkpointer=None,
    )


def _step_microbatch_groups(step: int, rank: int, num_groups: int):
    """Pack 40 random samples with the batcher and return the step's first num_groups groups."""
    rng = np.random.default_rng(1000 + 100 * rank + step)
    samples = []
    for rollout_id, num_tokens in enumerate(rng.integers(16, 120, size=40)):
        prompt_len = int(rng.integers(4, num_tokens // 2))
        samples.append(
            TrainingSample(
                min_policy_version=0,
                max_policy_version=0,
                rollout_id=RolloutTurnID(group_id=0, rollout_id=rollout_id, turn_id=0),
                token_ids=rng.integers(0, 2048, size=num_tokens),
                loss_mask=np.arange(num_tokens) >= prompt_len,
                # Near the policy's logprobs (~ -log(2048)), so ratios hit both clip bounds.
                logprobs=-rng.uniform(7.0, 8.2, size=num_tokens).astype(np.float32),
                advantage=np.full(num_tokens, rng.choice([-1.0, 0.5, 1.0]), np.float32),
            )
        )
    batch, _ = (
        Batcher.Config(max_num_documents=_MAX_NUM_DOCUMENTS)
        .build(
            num_tokens_per_microbatch_per_dp_rank=_NUM_TOKENS_PER_MICROBATCH,
            max_context_length=_NUM_TOKENS_PER_MICROBATCH,
            num_prompts_per_train_step=1,
            dp_degree=1,
            num_pp_microbatches=1,
            pad_id=0,
            temperature=0.8,
        )
        .add_training_samples(
            training_sample_group=TrainingSampleGroup(
                group_id=0, training_samples=samples, metrics=[]
            )
        )
    )
    assert batch is not None
    groups = [[row[0] for row in group] for group in batch.microbatch_groups]
    assert len(groups) >= num_groups
    groups = groups[:num_groups]
    microbatch = groups[0][0]
    if step == 4:
        # DAPO drops loss tokens whose generator logprob is not finite.
        trained = torch.nonzero(microbatch.loss_mask).flatten()[:3]
        microbatch.generator_logprobs[trained] = float("-inf")
    if step == 5:
        # No loss tokens, like an empty bin.
        microbatch.loss_mask.zero_()
        microbatch.advantages.zero_()
    if step in (4, 5):
        microbatch.loss_token_count = (
            microbatch.loss_mask & torch.isfinite(microbatch.generator_logprobs)
        ).sum()
    return groups


def _train(rank: int, port: int, case_name: str, graphs: bool, output_dir: str) -> None:
    case = _CASES[case_name]
    os.environ.update(
        MASTER_ADDR="localhost",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(case.world_size),
    )
    trainer = Trainer(
        _trainer_config(case, graphs=graphs),
        max_num_documents=_MAX_NUM_DOCUMENTS,
        output_dir=tempfile.mkdtemp(dir=output_dir),
    )
    engine = trainer.engine
    named_parameters = [
        (name, parameter)
        for model_part in engine.model_parts
        for name, parameter in model_part.named_parameters()
        if parameter.requires_grad
    ]
    steps = []
    optimizer_step = engine.optim.step

    def recording_optimizer_step(loss, *, current_step):
        # The gradients the optimizer sees: summed over the step's calls, divided by its count.
        steps[-1]["gradients"] = {
            name: get_local_tensor(parameter.grad).clone().cpu()
            for name, parameter in named_parameters
        }
        return optimizer_step(loss, current_step=current_step)

    engine.optim.step = recording_optimizer_step
    for step, call_sizes in enumerate(case.calls_per_step, start=1):
        groups = _step_microbatch_groups(step, rank, sum(call_sizes))
        steps.append({"metrics": []})
        start = 0
        for call_size in call_sizes:
            result = engine.forward_backward(
                microbatch_groups=groups[start : start + call_size]
            )
            start += call_size
            steps[-1]["metrics"] += [
                {key: value.cpu() for key, value in metrics.items()}
                for group in result.loss_metrics
                for metrics in group
            ]
        steps[-1]["loss"] = engine.loss.cpu()
        steps[-1]["grad_norm"] = engine.optim_step().cpu()
    if graphs:
        # Every graph captured, so later steps replayed them.
        assert engine._first_segment_graph.captured
        expected_lengths = (
            {size for call_sizes in case.calls_per_step for size in call_sizes[1:]}
            if engine.graph_segment_is_whole_call
            else {1}
        )
        assert engine._later_segment_graphs.keys() == expected_lengths
        assert all(graph.captured for graph in engine._later_segment_graphs.values())
    weights = {
        name: get_local_tensor(parameter).detach().cpu()
        for name, parameter in named_parameters
    }
    torch.save(
        {"steps": steps, "weights": weights},
        os.path.join(output_dir, f"{'graphs' if graphs else 'eager'}_{rank}.pt"),
    )
    engine.close()
    dist.destroy_process_group()


@pytest.mark.parametrize(
    "case_name",
    [
        pytest.param(name, marks=[pytest.mark.multi_gpu] if case.world_size > 1 else [])
        for name, case in _CASES.items()
    ],
)
def test_segment_graphs_match_eager(case_name: str, tmp_path) -> None:
    world_size = _CASES[case_name].world_size
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    for graphs in (False, True):
        mp.spawn(
            _train,
            args=(get_free_port(), case_name, graphs, str(tmp_path)),
            nprocs=world_size,
            join=True,
        )
    for rank in range(world_size):
        eager = torch.load(tmp_path / f"eager_{rank}.pt", weights_only=True)
        graphs = torch.load(tmp_path / f"graphs_{rank}.pt", weights_only=True)
        for step, (expected, actual) in enumerate(
            zip(eager["steps"], graphs["steps"], strict=True), start=1
        ):
            for name, gradient in expected["gradients"].items():
                assert torch.equal(actual["gradients"][name], gradient), (step, name)
            assert torch.equal(actual["loss"], expected["loss"]), step
            assert torch.equal(actual["grad_norm"], expected["grad_norm"]), step
            for expected_metrics, actual_metrics in zip(
                expected["metrics"], actual["metrics"], strict=True
            ):
                assert expected_metrics.keys() == actual_metrics.keys()
                for key, value in expected_metrics.items():
                    assert torch.equal(actual_metrics[key], value), (step, key)
        for name, weight in eager["weights"].items():
            assert torch.equal(graphs["weights"][name], weight), name
