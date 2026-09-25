# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for the TorchFT optimizer container's MoE load balancing.

The container wraps the base optimizer step to run the quorum and to
synchronize MoE load balancing across replicas, so one accepted step must
synchronize the global expert counts exactly once.
"""

import tempfile
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.unit_tests.cpu.test_optimizer_param_groups import (
    FakeMoEModel,
    FakeParallelDims,
)
from torch.distributed.device_mesh import init_device_mesh

from torchtitan.components.optimizer import (
    OptimizersContainer,
    ParamGroupConfig,
    register_moe_load_balancing_hook,
)
from torchtitan.experiments.torchft.optimizer import TorchFTOptimizersContainer


def _run_torchft_moe_load_balancing_step(rank, store_path):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        loss_mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("loss",))
        model = FakeMoEModel(load_balance_coeffs=(0.1, 0.2))
        # Rows are MoE layers; columns are experts. Summing across ranks
        # reverses rank 0's local imbalance in both layers.
        local_counts_by_rank = {
            0: [[10, 0], [0, 10]],
            1: [[0, 20], [20, 0]],
        }
        expected_global_counts = torch.tensor([[10, 20], [20, 10]])
        local_counts = torch.tensor(local_counts_by_rank[rank])
        for layer, counts in zip(model.layers.values(), local_counts):
            layer.moe.router.tokens_per_expert_E.copy_(counts)

        config = TorchFTOptimizersContainer.Config(
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=r".*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"lr": 0.1, "weight_decay": 0.0},
                ),
            ],
        )
        # Wrap the base step first to expose hook re-entry through super().step().
        OptimizersContainer(config, model_parts=[FakeMoEModel()])

        manager = Mock(spec=["start_quorum", "should_commit"])
        manager.should_commit.return_value = True
        container = config.build(
            model_parts=[model],
            ft_manager=SimpleNamespace(manager=manager, use_async_quorum=True),
        )
        register_moe_load_balancing_hook(
            container, [model], FakeParallelDims(loss_mesh=loss_mesh)
        )

        container.zero_grad()
        model.weight.grad = torch.ones_like(model.weight)
        # Keep Gloo communication real; only observe collectives during this step.
        with patch(
            "torch.distributed.all_reduce", wraps=dist.all_reduce
        ) as load_all_reduce:
            container.step()

        # This is the regression: one accepted step must synchronize load once.
        load_all_reduce.assert_called_once()
        # all_reduce updates its input tensor in place.
        global_counts = load_all_reduce.call_args.args[0]
        torch.testing.assert_close(global_counts, expected_global_counts)

        # Increase bias for the globally less-used expert in each layer.
        torch.testing.assert_close(
            model.layers["0"].moe.expert_bias_E,
            torch.tensor([0.1, -0.1]),
        )
        torch.testing.assert_close(
            model.layers["1"].moe.expert_bias_E,
            torch.tensor([-0.2, 0.2]),
        )

        # The next training step must start with empty load counters.
        torch.testing.assert_close(
            model.layers["0"].moe.router.tokens_per_expert_E,
            torch.tensor([0, 0]),
        )
        torch.testing.assert_close(
            model.layers["1"].moe.router.tokens_per_expert_E,
            torch.tensor([0, 0]),
        )

        manager.start_quorum.assert_called_once_with()
        manager.should_commit.assert_called_once_with()
        torch.testing.assert_close(model.weight, torch.tensor([0.9]))
    finally:
        dist.destroy_process_group()


class TestTorchFTMoELoadBalancing(unittest.TestCase):
    @unittest.skipUnless(
        dist.is_available() and dist.is_gloo_available(),
        "Requires Gloo for the two-rank MoE load-balancing test.",
    )
    def test_accepted_step_synchronizes_global_moe_load_once(self):
        """A wrapped base step must not repeat the FT container's MoE hook."""
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _run_torchft_moe_load_balancing_step,
                args=(f"{directory}/rendezvous",),
                nprocs=2,
                join=True,
            )
