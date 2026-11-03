# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Accelerator graphs give the same results as eager for several forward_backward calls per step.

A segment is the microbatch groups that one graph replay runs. The step's first call is one
segment: it replays the first-segment graph, which allocates the gradients. A later call
splits at each gradient reduction (each group with plain FSDP, the whole call with HSDP or
deferred reduction), and each piece replays the later-segment graph of its length, which adds
into them. Each case runs the same calls with and without graphs, then compares the gradients
the optimizer sees, the loss, the grad norm and the final weights with torch.equal.

Calls per step, e.g. [2, 3] = a call with 2 groups, then a call with 3. The step's first call
has the same group count every step. Step 4 has a microbatch with half its loss tokens
ignored, and step 5 a microbatch without loss tokens.

    pytest tests/unit_tests/gpu/distributed/test_accelerator_graph_segments.py              # 1 GPU
    pytest tests/unit_tests/gpu/distributed/test_accelerator_graph_segments.py -m multi_gpu # 2 GPUs
"""

import dataclasses
import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.elastic.utils.distributed import get_free_port

from torchtitan.components.data.types import TokenizedTrainingMicrobatch
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.utils import get_local_tensor
from torchtitan.models.deepseek_v3.mtp import get_mtp_token_counts
from torchtitan.observability.sdc_replayer import SDCReplayer
from torchtitan.training_engine import TrainingEngine
from torchtitan_recipes.tests.models.llama3 import llama3_debugmodel

_NUM_TOKENS_PER_MICROBATCH = 256
_MAX_NUM_DOCUMENTS = 16
_VOCAB_SIZE = 2048

# One call per step, as the core trainer sends it: only the first-segment graph replays.
_ONE_CALL = [[3]] * 5
# The first call's 2 groups replay the first-segment graph; each later group replays the
# 1-group later-segment graph.
_FSDP_CALLS = [[2, 1], [2, 3], [2], [2, 1, 2], [2, 2], [2, 1]]
# Under HSDP or deferral, a later segment is a whole call. Without SDC replay, the
# first-segment graph captures in step 3. The 1-group later graph is called 3 times before
# that, so the capture-order rule must hold it back. Each length then captures and replays.
_WHOLE_CALL_CALLS = [
    [2, 1, 1, 2],
    [2, 1, 3],
    [2, 2, 1],
    [2, 3, 2],
    [2, 3],
    [2, 2, 1, 3],
]


@dataclasses.dataclass(frozen=True)
class _Case:
    world_size: int
    parallelism: ParallelismConfig
    calls_per_step: list[list[int]]
    sdc_replay_steps: int | None
    """SDC replay checks every call of this many first steps (-1: all), with one replay each.
    The first-segment graph can capture in a checked call; later-segment graphs don't."""
    preprocess_lazily: bool = False


_CASES = {
    "one_call": _Case(1, ParallelismConfig(), _ONE_CALL, sdc_replay_steps=-1),
    # Steps 1-2 are checked, so they preprocess up front and their later segments don't
    # capture. Steps 3-6 preprocess lazily; the 1-group later-segment graph captures in step 4.
    "fsdp": _Case(
        1, ParallelismConfig(), _FSDP_CALLS, sdc_replay_steps=2, preprocess_lazily=True
    ),
    "deferral_lazy": _Case(
        1,
        ParallelismConfig(fsdp_defer_gradient_reduction=True),
        _WHOLE_CALL_CALLS,
        sdc_replay_steps=None,
        preprocess_lazily=True,
    ),
    # A real reduce-scatter in each segment.
    "dp_shard_2": _Case(
        2,
        ParallelismConfig(data_parallel_shard_degree=2),
        _FSDP_CALLS,
        sdc_replay_steps=2,
    ),
    # The replica all-reduce in each whole-call segment.
    "dp_replicate_2": _Case(
        2,
        ParallelismConfig(data_parallel_replicate_degree=2),
        _WHOLE_CALL_CALLS,
        sdc_replay_steps=2,
    ),
}


def _engine_config(case: _Case, *, graphs: bool) -> TrainingEngine.Config:
    config = llama3_debugmodel(
        seq_len=_NUM_TOKENS_PER_MICROBATCH, parallelism=case.parallelism
    )
    return dataclasses.replace(
        config,
        training=dataclasses.replace(
            config.training,
            num_tokens_per_microbatch_per_dp_rank=_NUM_TOKENS_PER_MICROBATCH,
            max_context_length=_NUM_TOKENS_PER_MICROBATCH,
            disable_accelerator_graphs=not graphs,
            preprocess_microbatch_groups_lazily=case.preprocess_lazily,
        ),
        sdc_replayer=(
            None
            if case.sdc_replay_steps is None
            else SDCReplayer.Config(num_steps=case.sdc_replay_steps)
        ),
        # The same weights in every run; SDC replay needs deterministic kernels.
        debug=dataclasses.replace(config.debug, seed=0, deterministic=True),
    )


def _microbatch(generator: torch.Generator, step: int, index: int):
    """Random documents packed into one microbatch, as the text collator packs them."""
    size = _NUM_TOKENS_PER_MICROBATCH
    lengths = []
    while sum(lengths) < size - 32:
        lengths.append(int(torch.randint(32, 120, (1,), generator=generator)))
    lengths[-1] -= max(0, sum(lengths) - size)
    num_tokens = sum(lengths)
    positions = torch.cat(
        [torch.arange(length) for length in lengths] + [torch.arange(size - num_tokens)]
    )
    labels = torch.randint(0, _VOCAB_SIZE, (size,), generator=generator)
    labels[num_tokens:] = IGNORE_INDEX
    padding_mask = torch.arange(size) >= num_tokens
    if step == 4 and index == 0:
        labels[::2] = IGNORE_INDEX
    if step == 5 and index == 0:
        # No loss tokens, like an empty bin.
        labels[:] = IGNORE_INDEX
    loss_token_count, routing_token_counts = get_mtp_token_counts(
        target_mask=labels != IGNORE_INDEX,
        positions=positions,
        padding_mask=padding_mask,
        num_mtp_layers=0,
    )
    return TokenizedTrainingMicrobatch(
        input=torch.randint(0, _VOCAB_SIZE, (size,), generator=generator),
        labels=labels,
        positions=positions,
        padding_mask=padding_mask,
        loss_token_count=loss_token_count,
        routing_token_counts=routing_token_counts,
    )


def _train(rank: int, port: int, case_name: str, graphs: bool, output_dir: str) -> None:
    case = _CASES[case_name]
    os.environ.update(
        MASTER_ADDR="localhost",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(case.world_size),
    )
    config = _engine_config(case, graphs=graphs)
    engine = TrainingEngine(
        config,
        model_config=config.model,
        max_num_documents=_MAX_NUM_DOCUMENTS,
        output_dir=tempfile.mkdtemp(dir=output_dir),
    )
    engine.initialize(hf_assets_path=config.hf_assets_path)
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
        generator = torch.Generator().manual_seed(1000 + 100 * rank + step)
        groups = [
            [_microbatch(generator, step, index)] for index in range(sum(call_sizes))
        ]
        # The model has no auxiliary loss; every call takes the step's denominators.
        denominators = engine.step_aux_loss_denominators(groups)
        steps.append({})
        start = 0
        for call_size in call_sizes:
            engine.forward_backward(
                microbatch_groups=groups[start : start + call_size],
                aux_loss_denominators=denominators,
            )
            start += call_size
        steps[-1]["loss"] = engine.loss.cpu()
        steps[-1]["grad_norm"] = engine.optim_step().cpu()
    if graphs:
        # Every graph captured, so later steps replayed them. A one-call step builds no
        # later-segment graph: it replays only the first-segment graph.
        assert engine._first_segment_graph.captured
        later_call_sizes = [
            size for call_sizes in case.calls_per_step for size in call_sizes[1:]
        ]
        if engine._graph_segment_is_whole_call:
            expected_lengths = set(later_call_sizes)
        else:
            expected_lengths = {1} if later_call_sizes else set()
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
    if torch.accelerator.device_count() < world_size:
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
        for name, weight in eager["weights"].items():
            assert torch.equal(graphs["weights"][name], weight), name
