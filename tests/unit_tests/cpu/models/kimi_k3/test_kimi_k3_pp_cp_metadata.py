# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Numerical parity of static CP-local carrier metadata versus inference."""

import copy
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.distributed.pipelining.schedules import ScheduleInterleaved1F1B
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.config import TrainingConfig
from torchtitan.distributed.pipeline_parallel import TorchTitanPipelineStage
from torchtitan.models.kimi_k3.pipeline_parallel import (
    _build_attn_res_stage_metadata,
    _swap_in_attn_res_stages,
)
from torchtitan.models.kimi_k3.pipeline_parallel.cache import PPRankLocalCache
from torchtitan.models.kimi_k3.pipeline_parallel.layout import BlockLayoutTables


class _CarrierFirst(nn.Module):
    """Emit one trainable hidden tensor and one committed residual block."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 8, dtype=torch.bfloat16)
        self.projection = nn.Linear(8, 8, dtype=torch.bfloat16)

    def forward(self, tokens_T):
        hidden_TD = self.projection(self.embedding(tokens_T))
        return hidden_TD, hidden_TD.unsqueeze(1)


class _CarrierLast(nn.Module):
    """Read both hidden state and the previous committed residual block."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(8, 8, dtype=torch.bfloat16)

    def forward(self, hidden_TD, stack_TND):
        return self.projection(hidden_TD + stack_TND.sum(1))


def _loss(output_TD, target_TD):
    """Form a deterministic scalar while retaining both carrier gradients."""
    return (output_TD.float() - target_TD.float()).square().mean()


class TestCPLocalCarrierNumerics(DTensorTestBase):
    """Use local loop stages to isolate metadata from network and CP math."""

    @property
    def device_type(self):
        return "cpu"

    @property
    def world_size(self):
        return 1

    @with_comms
    def test_static_cp_local_metadata_preserves_exact_losses_and_gradients(self):
        torch.manual_seed(42)
        original_modules = [_CarrierFirst(), _CarrierLast()]
        layout = BlockLayoutTables(
            stage_to_rank={0: 0, 1: 0},
            n_layers=2,
            layers_per_block=1,
            layer_to_stage={0: 0, 1: 1},
            cache=False,
        )
        inputs_T = torch.arange(8, dtype=torch.int64)
        targets_TD = torch.arange(64, dtype=torch.float32).reshape(8, 8) / 64
        metadata = _build_attn_res_stage_metadata(
            training=TrainingConfig(
                num_tokens_per_microbatch_per_dp_rank=64,
                mixed_precision_param="bfloat16",
            ),
            model_config=SimpleNamespace(dim=8, vocab_size=32),
            loss_fn=ChunkedLossWrapper(ChunkedLossWrapper.Config()),
            layout=layout,
            context_parallel_degree=16,
        )
        histories = []
        original_deterministic = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            for stage_metadata in (None, metadata):
                modules = copy.deepcopy(original_modules)
                stages = [
                    TorchTitanPipelineStage(module, index, 2, torch.device("cpu"))
                    for index, module in enumerate(modules)
                ]
                schedule = ScheduleInterleaved1F1B(
                    stages, n_microbatches=2, loss_fn=_loss
                )
                rebuilt = _swap_in_attn_res_stages(schedule, stage_metadata)
                store = PPRankLocalCache()
                for stage in rebuilt:
                    stage.set_routing(layout, store)
                optimizer = torch.optim.SGD(
                    [
                        parameter
                        for module in modules
                        for parameter in module.parameters()
                    ],
                    lr=0.01,
                )
                history = []
                for _ in range(3):
                    losses = []
                    schedule.step(inputs_T, target=targets_TD, losses=losses)
                    grads = [
                        parameter.grad.detach().clone()
                        for module in modules
                        for parameter in module.parameters()
                    ]
                    norm = torch.stack([grad.float().norm() for grad in grads]).norm()
                    history.append((torch.stack(losses), grads, norm))
                    optimizer.step()
                    optimizer.zero_grad()
                histories.append(history)
        finally:
            torch.use_deterministic_algorithms(original_deterministic)
        for inferred, declared in zip(*histories, strict=True):
            assert torch.equal(inferred[0], declared[0])
            assert torch.equal(inferred[2], declared[2])
            for reference, actual in zip(inferred[1], declared[1], strict=True):
                assert torch.equal(reference, actual)
