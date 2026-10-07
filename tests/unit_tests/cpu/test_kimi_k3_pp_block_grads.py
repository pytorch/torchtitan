# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.pipelining.schedules import ScheduleInterleaved1F1B
from torch.distributed.pipelining.stage import _PipelineStageBase
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.models.kimi_k3.pipeline_parallel.cache import PPRankLocalCache

from torchtitan.models.kimi_k3.pipeline_parallel.layout import infer_block_layout_tables
from torchtitan.models.kimi_k3.pipeline_parallel.stage import AttnResPipelineStage

NUM_LAYERS, LAYERS_PER_BLOCK = 16, 4
NUM_BLOCKS = NUM_LAYERS // LAYERS_PER_BLOCK
# Channels: one per layer's reads, the head's reads, the loss readout, the input.
HEAD, READOUT, INPUT = NUM_LAYERS, NUM_LAYERS + 1, NUM_LAYERS + 2
DIM = NUM_LAYERS + 3
TOKENS, MICROBATCHES, STEPS = 1, 4, 3
# pp4 x vpp4 with the head alone on the last stage, two layers per stage, and uneven stages
SPLITS = {
    "pp4 x vpp4, head alone": [[0], [1, 2]] + [[s + 1] for s in range(2, 15)] + [[]],
    "pp4 x vpp2": [[2 * s, 2 * s + 1] for s in range(8)],
    "pp4 x vpp2, blocks open inside stages": [
        [0, 1],
        [2, 3, 4],
        [5],
        [6, 7, 8, 9],
        [10],
        [11, 12, 13],
        [14],
        [15],
    ],
}


class _ExactStage(nn.Module):
    """A Kimi K3-shaped pipeline stage whose block gradients are small integers."""

    def __init__(
        self, layers: list[int], *, first: bool, last: bool, dtype: torch.dtype
    ):
        super().__init__()
        self.layers, self.first, self.last = layers, first, last
        self.blocks = nn.ParameterDict(
            {
                str(layer // LAYERS_PER_BLOCK): nn.Parameter(
                    torch.arange(DIM, dtype=dtype) % 3 + 1 + layer // LAYERS_PER_BLOCK
                )
                for layer in layers
                if layer % LAYERS_PER_BLOCK == 0
            }
        )

    def forward(self, hidden: torch.Tensor, stack: torch.Tensor | None = None):
        if self.first:
            hidden = F.pad(hidden, (INPUT, 0))
            stack = hidden.new_zeros(hidden.shape[0], 0, DIM)
        assert stack is not None
        for layer in self.layers:
            if layer % LAYERS_PER_BLOCK == 0:
                block = self.blocks[str(layer // LAYERS_PER_BLOCK)] * hidden[:, INPUT:]
                stack = torch.cat((stack, block.unsqueeze(1)), dim=1)
            for _ in range(2):
                read = _read(stack, layer).unsqueeze(1)
                hidden = hidden + F.pad(read, (READOUT, DIM - READOUT - 1))
        if self.last:
            return hidden[:, READOUT] + _read(stack, HEAD)
        return hidden, stack


def _read(stack: torch.Tensor, channel: int) -> torch.Tensor:
    weights = torch.arange(1, stack.shape[1] + 1, dtype=stack.dtype)
    return (stack[:, :, channel] * weights).sum(1)


def _loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (output * target).sum()


def _batch(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    weights = torch.arange(1, MICROBATCHES + 1, dtype=dtype).repeat_interleave(TOKENS)
    return weights.unsqueeze(1), weights


def _expected_block_grad(block: int, dtype: torch.dtype) -> torch.Tensor:
    total = sum(TOKENS * (mb + 1) ** 2 for mb in range(MICROBATCHES))
    reads = torch.zeros(DIM, dtype=dtype)
    reads[block * LAYERS_PER_BLOCK : NUM_LAYERS] = 2
    reads[HEAD] = 1
    return total * (block + 1) * reads


def _block_grads(modules) -> dict[int, torch.Tensor]:
    return {
        int(name.split(".")[1]): param.grad.clone()
        for module in modules
        for name, param in module.named_parameters()
    }


def _train(modules, step_fn, dtype: torch.dtype):
    params = [p for module in modules for p in module.parameters()]
    optimizer = torch.optim.SGD(params, lr=1.0) if params else None
    history = []
    for _ in range(STEPS):
        losses = step_fn(*_batch(dtype))
        history.append((_block_grads(modules), losses))
        if optimizer is not None:
            optimizer.step()
            optimizer.zero_grad()
    return history


def _run_single_device(split: list[list[int]], dtype: torch.dtype):
    last = len(split) - 1
    modules = [
        _ExactStage(layers, first=s == 0, last=s == last, dtype=dtype)
        for s, layers in enumerate(split)
    ]

    def step(inputs, targets):
        losses = []
        for x, y in zip(
            inputs.chunk(MICROBATCHES), targets.chunk(MICROBATCHES), strict=True
        ):
            out = modules[0](x)
            for module in modules[1:]:
                out = module(*out)
            loss = _loss(out, y)
            loss.backward()
            losses.append(loss.detach())
        return losses

    return _train(modules, step, dtype)


class TestKimiK3PipelineExactBlockGradients(DTensorTestBase):
    @property
    def device_type(self) -> str:
        return "cpu"

    @property
    def world_size(self) -> int:
        return 4

    def _run_pipeline(
        self,
        split: list[list[int]],
        dtype: torch.dtype,
        cache: bool,
        eval_losses: list | None = None,
    ):
        num_stages, last = len(split), len(split) - 1
        mine = range(self.rank, num_stages, self.world_size)
        modules = [
            _ExactStage(split[s], first=s == 0, last=s == last, dtype=dtype)
            for s in mine
        ]
        stages = [
            AttnResPipelineStage(module, s, num_stages, torch.device("cpu"))
            for module, s in zip(modules, mine, strict=True)
        ]
        schedule_stages: list[_PipelineStageBase] = list(stages)
        schedule = ScheduleInterleaved1F1B(
            schedule_stages,
            n_microbatches=MICROBATCHES,
            loss_fn=_loss,
            scale_grads=False,
        )
        layout = infer_block_layout_tables(
            stage_to_rank=dict(stages[0].stage_index_to_group_rank),
            n_layers=NUM_LAYERS,
            layers_per_block=LAYERS_PER_BLOCK,
            layer_to_stage={
                layer: s for s, layers in enumerate(split) for layer in layers
            },
            cache=cache,
        )
        store = PPRankLocalCache()
        for stage in stages:
            stage.set_routing(layout, store)

        def step(inputs, targets):
            losses: list[torch.Tensor] = []
            args = (inputs,) if 0 in mine else ()
            if last in mine:
                schedule.step(*args, target=targets, losses=losses)
            else:
                schedule.step(*args)
            if eval_losses is not None:
                evaluated: list[torch.Tensor] = []
                with torch.no_grad():
                    if last in mine:
                        schedule.eval(*args, target=targets, losses=evaluated)
                    else:
                        schedule.eval(*args)
                if store.blocks(0) or any(
                    store.blocks(mb) for mb in range(MICROBATCHES)
                ):
                    raise AssertionError("eval left blocks in the rank store")
                eval_losses.append(
                    (
                        [loss.detach() for loss in evaluated],
                        sum(len(s._order) + len(s._delta_in) for s in stages),
                    )
                )
            return [loss.detach() for loss in losses]

        sent = sum(len(layout.delta_to_send(s)) for s in range(num_stages))
        return _train(modules, step, dtype), sent

    @with_comms
    def test_rank_cache_matches_whole_stack_and_single_device(self):
        for dtype in (torch.bfloat16, torch.float32):
            for name, split in SPLITS.items():
                with self.subTest(dtype=dtype, split=name):
                    reference = _run_single_device(split, dtype)
                    cached, cached_sent = self._run_pipeline(split, dtype, cache=True)
                    naive, naive_sent = self._run_pipeline(split, dtype, cache=False)
                    self.assertLess(cached_sent, naive_sent)
                    for step in range(STEPS):
                        ref_grads, ref_losses = reference[step]
                        for grads, losses in (cached[step], naive[step]):
                            for block, grad in grads.items():
                                expected = _expected_block_grad(block, dtype)
                                torch.testing.assert_close(
                                    grad, ref_grads[block], rtol=0, atol=0
                                )
                                torch.testing.assert_close(
                                    grad, expected, rtol=0, atol=0
                                )
                            if losses:
                                torch.testing.assert_close(
                                    losses, ref_losses, rtol=0, atol=0
                                )

    @with_comms
    def test_forward_only_eval_between_steps(self):
        for name, split in SPLITS.items():
            with self.subTest(split=name):
                reference = _run_single_device(split, torch.float32)
                evaluated: list = []
                history, _ = self._run_pipeline(
                    split, torch.float32, cache=True, eval_losses=evaluated
                )
                for step in range(STEPS):
                    grads, losses = history[step]
                    ref_grads, ref_losses = reference[step]
                    for block, grad in grads.items():
                        torch.testing.assert_close(
                            grad, ref_grads[block], rtol=0, atol=0
                        )
                    eval_step_losses, leftover = evaluated[step]
                    if losses:
                        torch.testing.assert_close(losses, ref_losses, rtol=0, atol=0)
                        torch.testing.assert_close(
                            eval_step_losses, ref_losses, rtol=0, atol=0
                        )
                    self.assertEqual(leftover, 0)


if __name__ == "__main__":
    unittest.main()
