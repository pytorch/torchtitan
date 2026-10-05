# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from unittest import mock

import pytest

from torchtitan.components.validate import Validator
from torchtitan.config import DebugConfig, ParallelismConfig, TrainingConfig
from torchtitan.models.common.token_dispatcher import HybridEPTokenDispatcher
from torchtitan.observability.sdc_replayer import SDCReplayer
from torchtitan.training_engine import TrainingEngine
from torchtitan_recipes.tests.models.deepseek_v3 import (
    deepseek_v3_debugmodel,
    deepseek_v3_debugmodel_hybridep,
)
from torchtitan_recipes.tests.models.llama3 import (
    llama3_debugmodel,
    llama3_debugmodel_varlen_attn,
)


@contextlib.contextmanager
def _cuda_graphs_supported(value: bool):
    with mock.patch(
        "torchtitan.distributed.cuda_graph.cuda_graphs_supported", return_value=value
    ), mock.patch(
        "torchtitan.trainer.cuda_graphs_supported", return_value=value
    ), mock.patch(
        "torchtitan.training_engine.cuda_graphs_supported", return_value=value
    ):
        yield


def test_training_token_counts_are_positive() -> None:
    with pytest.raises(ValueError, match="must be greater than 0"):
        TrainingConfig(num_tokens_per_microbatch_per_dp_rank=0)
    with pytest.raises(ValueError, match="must be -1 or greater than 0"):
        TrainingConfig(num_tokens_per_train_step=0)
    with pytest.raises(ValueError, match="must be greater than 0"):
        TrainingConfig(max_context_length=0)


def test_cuda_graphs_reject_pipeline_validation() -> None:
    config = llama3_debugmodel()
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = "1F1B"
    config.validator = Validator.Config()

    with _cuda_graphs_supported(True), pytest.raises(
        ValueError, match="do not support validation"
    ):
        config.__post_init__()


def test_cuda_graphs_reject_synchronizing_ep_dispatcher() -> None:
    config = deepseek_v3_debugmodel()
    config.parallelism.expert_parallel_degree = 2

    with _cuda_graphs_supported(True), pytest.raises(
        ValueError, match="without CPU synchronization"
    ):
        config.__post_init__()


def test_cuda_graphs_allow_nonblocking_hybrid_ep() -> None:
    config = deepseek_v3_debugmodel_hybridep()
    config.parallelism.expert_parallel_degree = 2

    with _cuda_graphs_supported(True):
        config.__post_init__()


def test_cuda_graphs_reject_blocking_hybrid_ep() -> None:
    config = deepseek_v3_debugmodel_hybridep()
    for _, dispatcher, _, _ in config.model.traverse(HybridEPTokenDispatcher.Config):
        dispatcher.non_blocking_capacity_factor = None
    config.parallelism.expert_parallel_degree = 2

    with _cuda_graphs_supported(True), pytest.raises(
        ValueError, match="non_blocking_capacity_factor"
    ):
        config.__post_init__()


def test_sdc_replay_requires_strict_determinism() -> None:
    config = llama3_debugmodel()
    config.training.disable_cuda_graphs = True
    config.sdc_replayer = SDCReplayer.Config()

    with pytest.raises(ValueError, match="debug.deterministic=True"):
        TrainingEngine.Config.__post_init__(config)

    config.debug.deterministic = True
    config.debug.deterministic_warn_only = True
    with pytest.raises(ValueError, match="deterministic_warn_only=False"):
        TrainingEngine.Config.__post_init__(config)


def test_sdc_replay_count_respects_cuda_graph_support() -> None:
    config = llama3_debugmodel()
    config.debug.deterministic = True
    config.sdc_replayer = SDCReplayer.Config(num_replays=2)

    with _cuda_graphs_supported(True), pytest.raises(
        ValueError, match="at most one replay"
    ):
        TrainingEngine.Config.__post_init__(config)
    with _cuda_graphs_supported(False):
        TrainingEngine.Config.__post_init__(config)


def test_microbatch_tokens_match_activation_sharding() -> None:
    config = TrainingEngine.Config()
    config.training = TrainingConfig(num_tokens_per_microbatch_per_dp_rank=10)
    config.parallelism = ParallelismConfig(
        tensor_parallel_degree=4,
        enable_sequence_parallel=True,
    )

    with pytest.raises(ValueError, match="pipeline microbatch"):
        config.__post_init__()

    config.training.num_tokens_per_microbatch_per_dp_rank = 16
    config.__post_init__()


def test_spmd_typechecking_rejects_pipeline_parallelism() -> None:
    with pytest.raises(ValueError, match="SPMD typechecking"):
        TrainingEngine.Config(
            debug=DebugConfig(spmd_typechecking=True),
            training=TrainingConfig(disable_cuda_graphs=True),
            parallelism=ParallelismConfig(pipeline_parallel_degree=2),
        )


def test_varlen_cuda_graphs_require_document_bound() -> None:
    config = llama3_debugmodel_varlen_attn()
    config.dataloader.max_num_documents = None

    with _cuda_graphs_supported(True), pytest.raises(
        ValueError, match="max_num_documents is unset"
    ):
        config.__post_init__()
    with _cuda_graphs_supported(False):
        config.__post_init__()


def test_cpu_offload_requires_distinct_dp_shard_seeds() -> None:
    with pytest.raises(ValueError, match="distinct_seed_mesh_axes"):
        TrainingEngine.Config(
            training=TrainingConfig(enable_cpu_offload=True),
        )

    TrainingEngine.Config(
        training=TrainingConfig(enable_cpu_offload=True),
        debug=DebugConfig(distinct_seed_mesh_axes=["pp", "dp_shard"]),
    )
