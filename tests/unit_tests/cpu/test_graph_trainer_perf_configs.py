# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from scripts.graph_trainer_runs import perf_configs, recipes
from torchtitan.config.transform import quantization
from torchtitan.models.common.dist_moe import DistMoeRoutedExperts
from torchtitan.models.common.moe import MoE
from torchtitan.models.deepseek_v3.moe import DeepSeekV3Router


@pytest.fixture
def config_env(monkeypatch):
    monkeypatch.setattr(quantization, "has_cuda_capability", lambda *_args: True)
    monkeypatch.setattr(recipes, "_prepare_outer_cuda_graph_capture", lambda: None)
    for name in (
        "GT_RUN_MODE",
        "GT_RUN_STEPS",
        "GT_RUN_C4_SOURCE",
        "GT_RUN_TOKENIZER_PATH",
        "GT_RUN_PROFILE",
        "GT_RUN_MEMORY_PROFILE",
        "RANK",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("topology,pp_degree", [("spmd", 1), ("pp2", 2), ("pp4", 4)])
@pytest.mark.parametrize("routing", ["natural", "force_balanced"])
def test_public_pairs_preserve_workload_and_mesh(
    config_env, topology, pp_degree, routing
):
    configs = [
        getattr(perf_configs, f"{topology}_256gpu_{engine}_{routing}")()
        for engine in ("graph_trainer", "eager")
    ]
    for is_graph_trainer, config in zip((True, False), configs, strict=True):
        parallelism = config.parallelism
        dp_degree = parallelism.data_parallel_shard_degree
        assert pp_degree * dp_degree == 256
        assert parallelism.expert_parallel_degree == 64
        assert dp_degree % parallelism.expert_parallel_degree == 0
        assert parallelism.num_pp_microbatches * dp_degree == 15_360
        assert config.training.num_tokens_per_train_step == 62_914_560
        assert config.training.num_tokens_per_microbatch_per_dp_rank == 4096
        assert config.training.steps == 60
        assert not config.training.disable_cuda_graphs
        assert config.parallelism.fsdp_defer_gradient_reduction is not is_graph_trainer
        assert config.parallelism.fsdp_symm_mem_scope is None
        assert len(config.model.layers) == 61
        assert len(config.model.mtp_layers) == 1
        assert config.loss.mtp_scale == 0.1
        assert config.dataloader.num_mtp_layers == 1
        assert config.dataloader.dataset.dataset.source.path == "allenai/c4"
        assert config.checkpointer is None and config.validator is None
        assert config.activation_remat is None
        assert config.dist_moe_runtime.activation_slot_capacity_factor == 1.0
        assert config.dist_moe_runtime.scratch_capacity_factor == (
            8.0 if routing == "natural" else 1.0
        )
        for _, moe, _, _ in config.model.traverse(MoE.Config):
            assert (type(moe.router) is DeepSeekV3Router.Config) == (
                routing == "natural"
            )
            assert moe.router.aux_loss.coeff == 0.01
        assert all(
            experts.inplace_wgrad_accum is not is_graph_trainer
            for _, experts, _, _ in config.model.traverse(DistMoeRoutedExperts.Config)
        )
        stages = parallelism.pipeline_parallel_module_fqns_per_model_part
        if pp_degree == 1:
            assert stages is None
        else:
            assert len(stages) == 16
            assert sorted(
                int(fqn.split(".")[1])
                for stage in stages
                for fqn in stage
                if fqn.startswith("layers.")
            ) == list(range(61))
            assert "mtp_layers.0" in stages[-1]
        assert not config.profiler.enable_profiling
        assert not config.profiler.enable_memory_snapshot
    assert configs[0].training == configs[1].training
    assert configs[0].optim == configs[1].optim
    assert configs[0].debug == configs[1].debug


def test_public_asset_overrides_and_selected_profile_ranks(config_env, monkeypatch):
    monkeypatch.setenv("GT_RUN_TOKENIZER_PATH", "/shared/tokenizer")
    monkeypatch.setenv("GT_RUN_C4_SOURCE", "/shared/c4")
    monkeypatch.setenv("GT_RUN_PROFILE", "1")
    monkeypatch.setenv("GT_RUN_MEMORY_PROFILE", "1")
    for rank in (0, 1, 128):
        monkeypatch.setenv("RANK", str(rank))
        config = perf_configs.spmd_256gpu_graph_trainer_natural()
        assert config.hf_assets_path == "/shared/tokenizer"
        assert config.dataloader.dataset.dataset.source.path == "/shared/c4"
        assert config.profiler.enable_profiling == (rank in {0, 128})
        assert config.profiler.enable_memory_snapshot == (rank in {0, 128})
    monkeypatch.setenv("GT_RUN_MODE", "verify")
    config = perf_configs.spmd_256gpu_graph_trainer_natural()
    assert config.debug.deterministic and config.debug.seed == 42
    assert not config.debug.deterministic_warn_only
    assert not config.profiler.enable_profiling
    assert not config.profiler.enable_memory_snapshot
