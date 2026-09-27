# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Import-safe training recipes for graph trainer numerical tests."""

from torchtitan.experiments.graph_trainer.deepseek_v3.config_registry import (
    graph_trainer_deepseek_v3_debugmodel,
)
from torchtitan.experiments.graph_trainer.llama3.config_registry import (
    graph_trainer_llama3_debugmodel,
    graph_trainer_llama3_debugmodel_sdpa_cross_entropy_loss,
)
from torchtitan.experiments.graph_trainer.qwen3.config_registry import (
    graph_trainer_qwen3_debugmodel,
    graph_trainer_qwen3_debugmodel_moe,
)
from torchtitan.models.deepseek_v3.config_registry import deepseek_v3_debugmodel
from torchtitan.models.llama3.config_registry import llama3_debugmodel
from torchtitan.models.qwen3.config_registry import qwen3_debugmodel, qwen3_moe_debug


def llama3_eager_numerics():
    config = llama3_debugmodel(seq_len=2048)
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


def _deepseek_v3_ep_overlap_numerics(*, chunk_dim: str, module_fqn: str, strategy: str):
    config = graph_trainer_deepseek_v3_debugmodel()
    config.training.disable_cuda_graphs = True
    config.training.num_tokens_per_microbatch_per_dp_rank = 16384
    config.parallelism.data_parallel_shard_degree = 8
    config.parallelism.tensor_parallel_degree = 1
    config.parallelism.expert_parallel_degree = 2
    config.compile.ep_overlap.enabled = True
    config.compile.ep_overlap.chunk_dim = chunk_dim
    config.compile.ep_overlap.module_fqn = module_fqn
    config.compile.ep_overlap.strategy = strategy
    config.compile.ep_overlap.disable_early_grad_accumulation = strategy == "graph"
    return config


def deepseek_v3_ep_overlap_transformer_eager():
    return _deepseek_v3_ep_overlap_numerics(
        chunk_dim="batch", module_fqn="layers.*", strategy="eager"
    )


def deepseek_v3_ep_overlap_transformer_graph():
    return _deepseek_v3_ep_overlap_numerics(
        chunk_dim="batch", module_fqn="layers.*", strategy="graph"
    )


def deepseek_v3_ep_overlap_moe_seq_eager():
    return _deepseek_v3_ep_overlap_numerics(
        chunk_dim="seq", module_fqn="layers.*.moe", strategy="eager"
    )


def deepseek_v3_ep_overlap_moe_seq_graph():
    return _deepseek_v3_ep_overlap_numerics(
        chunk_dim="seq", module_fqn="layers.*.moe", strategy="graph"
    )


def deepseek_v3_ep_overlap_moe_batch_eager():
    return _deepseek_v3_ep_overlap_numerics(
        chunk_dim="batch", module_fqn="layers.*.moe", strategy="eager"
    )


def deepseek_v3_ep_overlap_moe_batch_graph():
    return _deepseek_v3_ep_overlap_numerics(
        chunk_dim="batch", module_fqn="layers.*.moe", strategy="graph"
    )


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
        deepseek_v3_debugmodel(seq_len=2048), schedule="Interleaved1F1B"
    )


def deepseek_v3_graph_pp_interleaved_numerics():
    config = _deepseek_v3_pp_numerics(
        graph_trainer_deepseek_v3_debugmodel(), schedule="Interleaved1F1B"
    )
    config.compile.inductor_compilation = "regional"
    return config


def deepseek_v3_graph_pp_zbv_numerics():
    config = _deepseek_v3_pp_numerics(
        graph_trainer_deepseek_v3_debugmodel(), schedule="ZBVZeroBubble"
    )
    config.compile.inductor_compilation = "regional"
    return config


def deepseek_v3_graph_pp_dual_pipe_v_numerics():
    config = _deepseek_v3_pp_numerics(
        graph_trainer_deepseek_v3_debugmodel(), schedule="DualPipeV"
    )
    config.compile.inductor_compilation = "regional"
    return config


def qwen3_eager_numerics():
    config = qwen3_debugmodel(seq_len=2048)
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
