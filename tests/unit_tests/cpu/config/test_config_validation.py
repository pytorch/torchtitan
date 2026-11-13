# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from unittest import mock

import pytest

from torchtitan.components.checkpointer.dcp import CheckpointManager
from torchtitan.components.validator import Validator
from torchtitan.config import DebugConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.validation import validate_model_training_config
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.models.common.moe.token_dispatcher import HybridEPTokenDispatcher
from torchtitan.observability.sdc_replayer import SDCReplayer
from torchtitan.training_engine import TrainingEngine
from torchtitan_recipes.data import c4_document_capacity
from torchtitan_recipes.tests.models.deepseek_v3 import (
    deepseek_v3_debugmodel,
    deepseek_v3_debugmodel_hybridep,
)
from torchtitan_recipes.tests.models.flux import flux_debugmodel
from torchtitan_recipes.tests.models.llama3 import (
    llama3_debugmodel,
    llama3_debugmodel_varlen_attn,
)


@contextlib.contextmanager
def _accelerator_graphs_supported(value: bool):
    with mock.patch(
        "torchtitan.distributed.accelerator_graph.accelerator_graphs_supported",
        return_value=value,
    ), mock.patch(
        "torchtitan.training_engine.accelerator_graphs_supported", return_value=value
    ):
        yield


def test_training_token_counts_are_positive() -> None:
    with pytest.raises(ValueError, match="must be greater than 0"):
        TrainingConfig(num_tokens_per_microbatch_per_dp_rank=0)
    with pytest.raises(ValueError, match="must be -1 or greater than 0"):
        TrainingConfig(num_tokens_per_train_step=0)


def test_per_group_accelerator_graph_config_restrictions() -> None:
    with pytest.raises(ValueError, match="require accelerator graphs to be enabled"):
        TrainingEngine.Config(
            training=TrainingConfig(
                accelerator_graph_per_microbatch_group=True,
                disable_accelerator_graphs=True,
            )
        )
    with pytest.raises(ValueError, match="fsdp_defer_gradient_reduction=False"):
        TrainingEngine.Config(
            training=TrainingConfig(accelerator_graph_per_microbatch_group=True),
            parallelism=ParallelismConfig(fsdp_defer_gradient_reduction=True),
        )
    with pytest.raises(ValueError, match="do not support SDC replay"):
        TrainingEngine.Config(
            training=TrainingConfig(
                accelerator_graph_per_microbatch_group=True,
                num_tokens_per_microbatch_per_dp_rank=1,
                num_tokens_per_train_step=3,
            ),
            sdc_replayer=SDCReplayer.Config(),
        )

    TrainingEngine.Config(
        training=TrainingConfig(accelerator_graph_per_microbatch_group=True),
        parallelism=ParallelismConfig(pipeline_parallel_degree=2),
    )


def test_lazy_preprocessing_config_restrictions() -> None:
    with pytest.raises(
        ValueError, match="requires training.disable_accelerator_graphs=True"
    ):
        TrainingEngine.Config(
            training=TrainingConfig(preprocess_microbatch_groups_lazily=True)
        )
    with pytest.raises(ValueError, match="does not support SDC replay"):
        TrainingEngine.Config(
            training=TrainingConfig(
                preprocess_microbatch_groups_lazily=True,
                disable_accelerator_graphs=True,
            ),
            sdc_replayer=SDCReplayer.Config(),
        )

    TrainingEngine.Config(
        training=TrainingConfig(
            preprocess_microbatch_groups_lazily=True, disable_accelerator_graphs=True
        )
    )


def test_graph_trainer_rejects_per_group_accelerator_graphs() -> None:
    config = llama3_debugmodel(attn_backend="flex")
    with pytest.raises(ValueError, match="not supported with GraphTrainer"):
        GraphTrainer.Config(
            model=config.model,
            training=TrainingConfig(accelerator_graph_per_microbatch_group=True),
        )
    with pytest.raises(ValueError, match="lazily is not supported with GraphTrainer"):
        GraphTrainer.Config(
            model=config.model,
            training=TrainingConfig(
                preprocess_microbatch_groups_lazily=True,
                disable_accelerator_graphs=True,
            ),
        )


def test_seed_checkpoint_requires_a_saving_checkpointer() -> None:
    config = llama3_debugmodel()
    config.create_seed_checkpoint = True
    config.checkpointer = None
    with pytest.raises(
        ValueError, match="create_seed_checkpoint requires a checkpointer"
    ):
        config.__post_init__()

    config.checkpointer = CheckpointManager.Config(load_only=True)
    with pytest.raises(
        ValueError, match="create_seed_checkpoint requires a checkpointer"
    ):
        config.__post_init__()

    config.checkpointer = CheckpointManager.Config()
    config.__post_init__()


def test_accelerator_graphs_allow_pipeline_validation() -> None:
    config = llama3_debugmodel()
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = "1F1B"
    config.validator = Validator.Config()

    with _accelerator_graphs_supported(True):
        config.__post_init__()


def test_accelerator_graphs_reject_synchronizing_ep_dispatcher() -> None:
    # Build with the parallelism being validated: the MoE sharding (in
    # particular the routed-expert EP placements) is derived at build time
    # and must not be patched afterwards.
    config = deepseek_v3_debugmodel(
        parallelism=ParallelismConfig(expert_parallel_degree=2)
    )
    # The builder disables accelerator graphs for EP > 1; this test needs
    # them on to exercise the dispatcher check.
    config.training.disable_accelerator_graphs = False
    config.dataloader.max_num_documents = c4_document_capacity(
        config.training.num_tokens_per_microbatch_per_dp_rank
    )

    with _accelerator_graphs_supported(True), pytest.raises(
        ValueError, match="without CPU synchronization"
    ):
        config.__post_init__()


def test_accelerator_graphs_allow_nonblocking_hybrid_ep() -> None:
    config = deepseek_v3_debugmodel_hybridep()

    with _accelerator_graphs_supported(True):
        config.__post_init__()


def test_accelerator_graphs_reject_blocking_hybrid_ep() -> None:
    config = deepseek_v3_debugmodel_hybridep()
    for _, dispatcher, _, _ in config.model.traverse(HybridEPTokenDispatcher.Config):
        dispatcher.non_blocking_capacity_factor = None

    with _accelerator_graphs_supported(True), pytest.raises(
        ValueError, match="non_blocking_capacity_factor"
    ):
        config.__post_init__()


def test_dist_moe_recipe_passes_parallel_axis_validation() -> None:
    """Dist-MoE configs must validate after MoE routed_experts replacement."""
    dist_moe = pytest.importorskip(
        "dist_moe",
        reason="Dist-MoE validation test requires the optional dist_moe package",
    )
    del dist_moe  # imported only to skip cleanly when missing
    from torchtitan.config.parallelism import ParallelismConfig

    from torchtitan_recipes.tests.models.deepseek_v3 import (
        deepseek_v3_debugmodel_dist_moe_bf16,
    )

    config = deepseek_v3_debugmodel_dist_moe_bf16(
        parallelism=ParallelismConfig(
            tensor_parallel_degree=2,
            expert_parallel_degree=2,
            data_parallel_shard_degree=2,
            context_parallel_degree=1,
        )
    )
    config.__post_init__()


def test_ep_off_built_moe_rejects_ep_gt1() -> None:
    # A MoE config built with expert_parallel_degree == 1 gets dense
    # routed-expert weight placements; validating it at EP > 1 must raise
    # instead of silently ignoring the EP degree.
    config = deepseek_v3_debugmodel()
    with pytest.raises(ValueError, match="declares no 'ep' sharding"):
        validate_model_training_config(
            config.model,
            parallelism=ParallelismConfig(expert_parallel_degree=2),
            training=config.training,
            debug=config.debug,
            activation_remat=config.activation_remat,
            max_num_documents=config.dataloader.max_num_documents,
        )


def test_flux_pipeline_parallel_degree_gt1_raises() -> None:
    config = flux_debugmodel()
    config.parallelism.pipeline_parallel_degree = 2

    with pytest.raises(ValueError, match="does not support pipeline parallelism"):
        config.__post_init__()


def test_sdc_replay_requires_strict_determinism() -> None:
    config = llama3_debugmodel()
    config.training.disable_accelerator_graphs = True
    config.sdc_replayer = SDCReplayer.Config()

    with pytest.raises(ValueError, match="debug.deterministic=True"):
        TrainingEngine.Config.__post_init__(config)

    config.debug.deterministic = True
    config.debug.deterministic_warn_only = True
    with pytest.raises(ValueError, match="deterministic_warn_only=False"):
        TrainingEngine.Config.__post_init__(config)


def test_sdc_replay_count_respects_accelerator_graph_support() -> None:
    config = llama3_debugmodel()
    config.debug.deterministic = True
    config.sdc_replayer = SDCReplayer.Config(num_replays=2)

    with _accelerator_graphs_supported(True), pytest.raises(
        ValueError, match="at most one replay"
    ):
        TrainingEngine.Config.__post_init__(config)
    with _accelerator_graphs_supported(False):
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
            training=TrainingConfig(disable_accelerator_graphs=True),
            parallelism=ParallelismConfig(pipeline_parallel_degree=2),
        )


def test_varlen_accelerator_graphs_require_document_bound() -> None:
    config = llama3_debugmodel_varlen_attn()
    config.dataloader.max_num_documents = None

    with _accelerator_graphs_supported(True), pytest.raises(
        ValueError, match="max_num_documents is unset"
    ):
        config.__post_init__()
    with _accelerator_graphs_supported(False):
        config.__post_init__()


def test_cpu_offload_requires_distinct_dp_shard_seeds() -> None:
    with pytest.raises(ValueError, match="distinct_seed_mesh_axes"):
        TrainingEngine.Config(
            parallelism=ParallelismConfig(fsdp_enable_cpu_offloading=True),
        )

    TrainingEngine.Config(
        parallelism=ParallelismConfig(fsdp_enable_cpu_offloading=True),
        debug=DebugConfig(distinct_seed_mesh_axes=["pp", "dp_shard"]),
    )
