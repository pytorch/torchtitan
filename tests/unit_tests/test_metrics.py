# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import dataclass

from torchtitan.components.metrics import _get_metrics_rank


@dataclass
class _ParallelDimsForMetrics:
    world_size: int
    pp: int

    @property
    def pp_enabled(self) -> bool:
        return self.pp > 1


class TestPipelineMetricsRank(unittest.TestCase):
    def test_non_pipeline_logs_on_rank_zero(self):
        parallel_dims = _ParallelDimsForMetrics(world_size=8, pp=1)
        self.assertEqual(
            _get_metrics_rank(
                parallel_dims=parallel_dims,
                pp_schedule="Interleaved1F1B",
            ),
            0,
        )

    def test_loop_pipeline_logs_on_first_last_stage_rank(self):
        parallel_dims = _ParallelDimsForMetrics(world_size=8, pp=2)
        self.assertEqual(
            _get_metrics_rank(
                parallel_dims=parallel_dims,
                pp_schedule="Interleaved1F1B",
            ),
            4,
        )

    def test_v_pipeline_logs_on_rank_zero(self):
        parallel_dims = _ParallelDimsForMetrics(world_size=8, pp=2)
        for schedule in ("ZBVZeroBubble", "DualPipeV"):
            with self.subTest(schedule=schedule):
                self.assertEqual(
                    _get_metrics_rank(
                        parallel_dims=parallel_dims,
                        pp_schedule=schedule,
                    ),
                    0,
                )


if __name__ == "__main__":
    unittest.main()
