# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import Schedule1F1B, ScheduleInterleaved1F1B

from torchtitan.models.kimi_k3.parallelize import _swap_in_attn_res_stages
from torchtitan.models.kimi_k3.pipeline_stage import AttnResPipelineStage


def _loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return output.sum()


def _get_mesh(*args, **kwargs):
    return None


class TestAttnResStageSwap(unittest.TestCase):
    """K3 rebuilds the stages core constructed as AttnResPipelineStage."""

    def setUp(self):
        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo",
                init_method="tcp://localhost:12391",
                world_size=1,
                rank=0,
            )

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    @staticmethod
    def _stage(index: int, num_stages: int) -> PipelineStage:
        return PipelineStage(
            nn.Linear(2, 2),
            index,
            num_stages,
            torch.device("cpu"),
            get_mesh=_get_mesh,
        )

    def _assert_rebuilt(self, old: PipelineStage, new: AttnResPipelineStage) -> None:
        self.assertIsInstance(new, AttnResPipelineStage)
        self.assertIsNot(new, old)
        # The same module object, so the model parts core returned stay valid.
        self.assertIs(new.submod, old.submod)
        self.assertEqual(
            (new.stage_index, new.num_stages, new.device),
            (old.stage_index, old.num_stages, old.device),
        )
        self.assertIs(new.group, old.group)
        self.assertIs(new._mesh_cache._get_mesh_cb, old._mesh_cache._get_mesh_cb)
        self.assertEqual(new.stage_index_to_group_rank, old.stage_index_to_group_rank)

    def test_single_stage_schedule(self):
        old = self._stage(0, 1)
        schedule = Schedule1F1B(old, n_microbatches=1, loss_fn=_loss)
        (new,) = _swap_in_attn_res_stages(schedule)
        self._assert_rebuilt(old, new)
        self.assertIs(schedule._stage, new)

    def test_multi_stage_schedule(self):
        old = [self._stage(index, 2) for index in range(2)]
        schedule = ScheduleInterleaved1F1B(old, n_microbatches=2, loss_fn=_loss)
        new = _swap_in_attn_res_stages(schedule)
        self.assertEqual(len(new), 2)
        for old_stage, new_stage in zip(old, new, strict=True):
            self._assert_rebuilt(old_stage, new_stage)
        self.assertEqual(schedule._stages, new)


if __name__ == "__main__":
    unittest.main()
