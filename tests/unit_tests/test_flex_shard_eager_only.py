# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchtitan.config import TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.flex_shard_llama3.config_registry import (
    graph_trainer_flex_shard_llama3_debugmodel,
)
from torchtitan.experiments.graph_trainer.flex_shard_llama3.parallelize import (
    _validate_flex_shard_eager_only,
)


def _parallel_dims(**overrides) -> ParallelDims:
    values = dict(
        dp_replicate=1,
        dp_shard=2,
        cp=1,
        tp=1,
        pp=1,
        ep=1,
        world_size=2,
    )
    values.update(overrides)
    return ParallelDims(**values)


class TestFlexShardEagerOnly(unittest.TestCase):
    def test_config_defaults_to_eager(self):
        config = graph_trainer_flex_shard_llama3_debugmodel()

        self.assertFalse(config.compile.enable)
        self.assertIsNone(config.compile.mode)

    def test_compile_mode_raises(self):
        with self.assertRaisesRegex(ValueError, "eager execution only"):
            _validate_flex_shard_eager_only(
                parallel_dims=_parallel_dims(),
                training=TrainingConfig(),
                compile_config=GraphTrainerCompileConfig(
                    enable=False,
                    mode="aot_fx_trace",
                ),
            )

    def test_compile_enable_raises(self):
        with self.assertRaisesRegex(ValueError, "eager execution only"):
            _validate_flex_shard_eager_only(
                parallel_dims=_parallel_dims(),
                training=TrainingConfig(),
                compile_config=GraphTrainerCompileConfig(enable=True, mode=None),
            )

    def test_tensor_parallel_raises(self):
        with self.assertRaisesRegex(ValueError, "tensor parallel"):
            _validate_flex_shard_eager_only(
                parallel_dims=_parallel_dims(tp=2, world_size=4),
                training=TrainingConfig(),
                compile_config=GraphTrainerCompileConfig(enable=False, mode=None),
            )

    def test_cpu_offload_raises(self):
        with self.assertRaisesRegex(ValueError, "CPU offload"):
            _validate_flex_shard_eager_only(
                parallel_dims=_parallel_dims(),
                training=TrainingConfig(enable_cpu_offload=True),
                compile_config=GraphTrainerCompileConfig(enable=False, mode=None),
            )


if __name__ == "__main__":
    unittest.main()
