# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GraphTrainer Dist-MoE configurations for the B200 test lane."""

from torchtitan.trainer import Trainer

from torchtitan_recipes.tests.suites.b200 import _configure_dist_moe_fsdp2_ep2


def graph_trainer_deepseek_v3_debugmodel_dist_moe_bf16_fsdp2_ep2() -> Trainer.Config:
    """Exercise BF16 Dist-MoE with non-pipeline GraphTrainer."""
    from torchtitan_recipes.tests.graph_trainer.deepseek_v3 import (
        graph_trainer_deepseek_v3_debugmodel_dist_moe_bf16,
    )

    return _configure_dist_moe_fsdp2_ep2(
        graph_trainer_deepseek_v3_debugmodel_dist_moe_bf16(seq_len=128)
    )


def graph_trainer_deepseek_v3_debugmodel_dist_moe_mxfp8_fsdp2_ep2_pp2() -> (
    Trainer.Config
):
    """Exercise MXFP8 GraphPP schedule-derived activation slots."""
    from torchtitan_recipes.tests.graph_trainer.deepseek_v3 import (
        graph_trainer_deepseek_v3_debugmodel_dist_moe_mxfp8,
    )

    config = _configure_dist_moe_fsdp2_ep2(
        graph_trainer_deepseek_v3_debugmodel_dist_moe_mxfp8(seq_len=128)
    )
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.parallelism.num_pp_microbatches = 4
    return config
