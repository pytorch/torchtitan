# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Import-safe GraphTrainer numerical-test configurations."""

from typing import Literal

from torchtitan_recipes.tests.graph_trainer.b200 import (
    graph_trainer_deepseek_v3_debugmodel_dist_moe_bf16_fsdp2_ep2,
)

from torchtitan_recipes.tests.graph_trainer.deepseek_v3 import (
    graph_trainer_deepseek_v3_debugmodel,
)
from torchtitan_recipes.tests.graph_trainer.llama3 import (
    graph_trainer_llama3_debugmodel,
    graph_trainer_llama3_debugmodel_sdpa_cross_entropy_loss,
)
from torchtitan_recipes.tests.graph_trainer.qwen3 import (
    graph_trainer_qwen3_debugmodel,
    graph_trainer_qwen3_debugmodel_moe,
)
from torchtitan_recipes.tests.models.deepseek_v3 import deepseek_v3_debugmodel
from torchtitan_recipes.tests.models.llama3 import llama3_debugmodel
from torchtitan_recipes.tests.models.qwen3 import qwen3_debugmodel, qwen3_moe_debug
from torchtitan_recipes.tests.suites.b200 import (
    deepseek_v3_debugmodel_dist_moe_bf16_fsdp2_ep2,
)


def llama3_eager_numerics():
    config = llama3_debugmodel(seq_len=2048)
    # Match GraphTrainer, which captures model and loss in one eager FX graph.
    config.model.local_compile_regions = []
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.data_parallel_shard_degree = 4
    config.training.num_tokens_per_microbatch_per_dp_rank = 16384
    return config


def llama3_graph_numerics():
    config = graph_trainer_llama3_debugmodel()
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.data_parallel_shard_degree = 4
    config.training.num_tokens_per_microbatch_per_dp_rank = 16384
    return config


def deepseek_v3_eager_numerics():
    config = deepseek_v3_debugmodel(seq_len=2048)
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 2
    config.training.num_tokens_per_microbatch_per_dp_rank = 16384
    return config


def deepseek_v3_graph_numerics():
    config = graph_trainer_deepseek_v3_debugmodel()
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 2
    config.training.num_tokens_per_microbatch_per_dp_rank = 16384
    return config


def _deepseek_v3_pp_numerics(config, *, schedule: str):
    config.training.disable_cuda_graphs = True
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.training.max_norm = float("inf")
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 8
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.expert_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = schedule
    config.metrics.save_for_all_ranks = True
    return config


def deepseek_v3_eager_pp_numerics():
    return _deepseek_v3_pp_numerics(
        deepseek_v3_debugmodel(seq_len=2048),
        schedule="Interleaved1F1B",
    )


def deepseek_v3_graph_pp_interleaved_numerics():
    config = _deepseek_v3_pp_numerics(
        graph_trainer_deepseek_v3_debugmodel(),
        schedule="Interleaved1F1B",
    )
    config.compile.inductor_compilation = "regional"
    return config


def deepseek_v3_graph_pp_zbv_numerics():
    config = _deepseek_v3_pp_numerics(
        graph_trainer_deepseek_v3_debugmodel(),
        schedule="ZBVZeroBubble",
    )
    config.compile.inductor_compilation = "regional"
    return config


def deepseek_v3_graph_pp_dual_pipe_v_numerics():
    config = _deepseek_v3_pp_numerics(
        graph_trainer_deepseek_v3_debugmodel(),
        schedule="DualPipeV",
    )
    config.compile.inductor_compilation = "regional"
    return config


def _deepseek_v3_dist_moe_pp_numerics(
    config,
    *,
    slot_policy: Literal["microbatch", "stage_microbatch"],
):
    """Configure the shared four-GPU Dist-MoE PP numerics contract."""
    from torchtitan.models.common.dist_moe import DistMoeRoutedExperts
    from torchtitan.models.common.dist_moe.runtime import DistMoeRuntime

    runtime_config = config.dist_moe
    assert isinstance(runtime_config, DistMoeRuntime.Config)
    runtime_config.activation_slot_capacity_factor = 2.0
    runtime_config.pp_activation_slot_policy = slot_policy
    for _, experts, _, _ in config.model.traverse(DistMoeRoutedExperts.Config):
        experts.inplace_wgrad_accum = False
    config.model.local_compile_regions = []
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.parallelism.num_pp_microbatches = 8
    return config


def deepseek_v3_dist_moe_eager_pp_microbatch_numerics():
    return _deepseek_v3_dist_moe_pp_numerics(
        deepseek_v3_debugmodel_dist_moe_bf16_fsdp2_ep2(),
        slot_policy="microbatch",
    )


def deepseek_v3_dist_moe_eager_pp_stage_microbatch_numerics():
    return _deepseek_v3_dist_moe_pp_numerics(
        deepseek_v3_debugmodel_dist_moe_bf16_fsdp2_ep2(),
        slot_policy="stage_microbatch",
    )


def deepseek_v3_dist_moe_graph_pp_stage_microbatch_numerics():
    config = _deepseek_v3_dist_moe_pp_numerics(
        graph_trainer_deepseek_v3_debugmodel_dist_moe_bf16_fsdp2_ep2(),
        slot_policy="stage_microbatch",
    )
    config.training.disable_cuda_graphs = True
    return config


def qwen3_eager_numerics():
    config = qwen3_debugmodel(seq_len=2048)
    # Match GraphTrainer, which captures model and loss in one eager FX graph.
    config.model.local_compile_regions = []
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.data_parallel_shard_degree = 4
    config.training.num_tokens_per_microbatch_per_dp_rank = 16384
    return config


def qwen3_graph_numerics():
    config = graph_trainer_qwen3_debugmodel()
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.data_parallel_shard_degree = 4
    config.training.num_tokens_per_microbatch_per_dp_rank = 16384
    return config


def qwen3_moe_eager_numerics():
    config = qwen3_moe_debug(seq_len=2048)
    config.training.disable_cuda_graphs = True
    config.training.num_tokens_per_microbatch_per_dp_rank = 16384
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 2
    return config


def qwen3_moe_graph_numerics():
    config = graph_trainer_qwen3_debugmodel_moe()
    config.training.disable_cuda_graphs = True
    config.training.num_tokens_per_microbatch_per_dp_rank = 16384
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 2
    return config


def llama3_sdpa_manual_numerics():
    config = graph_trainer_llama3_debugmodel_sdpa_cross_entropy_loss()
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    return config


def llama3_sdpa_autoparallel_numerics():
    config = llama3_sdpa_manual_numerics()
    config.compile.enable_autoparallel = True
    return config


def deepseek_v3_autoparallel_numerics():
    config = deepseek_v3_graph_numerics()
    config.parallelism.tensor_parallel_degree = 1
    config.compile.enable_autoparallel = True
    return config


def deepseek_v3_manual_numerics():
    config = deepseek_v3_eager_numerics()
    config.parallelism.tensor_parallel_degree = 1
    return config
