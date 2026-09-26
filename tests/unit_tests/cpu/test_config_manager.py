# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import dataclasses
import io
import sys
import typing
import unittest
from unittest import mock

import pytest
import tyro
from torchtitan.components.validate import Validator
from torchtitan.config import (
    CompileConfig,
    ConfigManager,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.models.deepseek_v3.config_registry import (
    deepseek_v3_debugmodel_hybridep,
)
from torchtitan.models.llama3.config_registry import llama3_debugmodel_dist_gemm
from torchtitan.models.qwen3.config_registry import qwen3_moe_deepep
from torchtitan.observability.sdc_replayer import SDCReplayer
from torchtitan.trainer import Trainer
from torchtitan.training_engine import TrainingEngine


@contextlib.contextmanager
def cuda_graphs_supported(value: bool):
    """Pin the CUDA-graph capability predicate at every site that binds it.

    ``config/validation.py`` imports it inside the function it guards, so
    patching the defining module covers that site; ``trainer.py`` and
    ``training_engine.py`` bind it at module import. The CUDA-graph gates are
    inert on any host that cannot capture, so the tests below say which of the
    two worlds they are asserting about.
    """
    with mock.patch(
        "torchtitan.distributed.cuda_graph.cuda_graphs_supported", return_value=value
    ), mock.patch(
        "torchtitan.trainer.cuda_graphs_supported", return_value=value
    ), mock.patch(
        "torchtitan.training_engine.cuda_graphs_supported", return_value=value
    ):
        yield


class TestConfigManager(unittest.TestCase):
    def test_model_config_args(self):
        """--module and --config together load the correct config."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            ["--module", "llama3", "--config", "llama3_debugmodel"]
        )
        assert type(config.model).__qualname__ == "Llama3Model.Config"
        assert config.training.steps == 10

    def test_model_config_args_equals_form(self):
        """--module=X --config=Y form works."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            ["--module=llama3", "--config=llama3_debugmodel"]
        )
        assert type(config.model).__qualname__ == "Llama3Model.Config"

    def test_parse_args_uses_current_sys_argv(self):
        """parse_args() without args reads sys.argv at call time."""
        config_manager = ConfigManager()
        argv = ["train.py", "--module", "nonexistent", "--config", "foo"]
        with mock.patch.object(sys, "argv", argv):
            with pytest.raises(ImportError, match="Cannot import module 'nonexistent'"):
                config_manager.parse_args()

    def test_model_without_config_errors(self):
        """--module alone raises ValueError."""
        config_manager = ConfigManager()
        with pytest.raises(ValueError, match="--config is required"):
            config_manager.parse_args(["--module", "llama3"])

    def test_config_without_model_errors(self):
        """--config alone raises ValueError."""
        config_manager = ConfigManager()
        with pytest.raises(ValueError, match="--module is required"):
            config_manager.parse_args(["--config", "llama3_debugmodel"])

    def test_missing_both_errors(self):
        """No --module or --config raises ValueError."""
        config_manager = ConfigManager()
        with pytest.raises(ValueError, match="--module is required"):
            config_manager.parse_args([])

    def test_torchtitan_recipes_package_resolves(self):
        """torchtitan_recipes is importable and its configs load."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "torchtitan_recipes.tests.features",
                "--config",
                "llama3_debugmodel_fsdp2_cp2",
            ]
        )
        assert type(config.model).__qualname__ == "Llama3Model.Config"
        assert config.parallelism.context_parallel_degree == 2

    def test_invalid_model_errors(self):
        """--module with unknown module name raises ImportError."""
        config_manager = ConfigManager()
        with pytest.raises(ImportError, match="Cannot import module"):
            config_manager.parse_args(["--module", "nonexistent", "--config", "foo"])

    def test_invalid_config_errors(self):
        """--config with unknown function name lists available functions."""
        config_manager = ConfigManager()
        with pytest.raises(ValueError, match="Available config functions"):
            config_manager.parse_args(["--module", "llama3", "--config", "nonexistent"])

    def test_rl_examples_registered_as_shorthands(self):
        """RL examples are valid --module shorthands (resolved under rl/examples).

        End-to-end resolution + build is covered by the RL integration tests
        (they run ``--module alphabet_sort``); kept out of here since importing an
        example's config_registry pulls in vLLM, which isn't available on CPU.
        """
        from torchtitan.experiments import _supported_experiments

        assert "alphabet_sort" in _supported_experiments
        assert "search_r1" in _supported_experiments
        assert "verifiers.dapo_math" in _supported_experiments

    def test_cli_overrides(self):
        """CLI args override config defaults."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--training.steps",
                "5",
                "--training.num_tokens_per_microbatch_per_dp_rank",
                "4096",
                "--training.num_tokens_per_train_step",
                "8192",
                "--training.max_context_length",
                "1024",
            ]
        )
        assert config.training.steps == 5
        assert config.training.num_tokens_per_microbatch_per_dp_rank == 4096
        assert config.training.num_tokens_per_train_step == 8192
        assert config.training.max_context_length == 1024

    def test_num_tokens_per_microbatch_must_be_positive(self):
        config_manager = ConfigManager()
        with pytest.raises(SystemExit):
            config_manager.parse_args(
                [
                    "--module",
                    "llama3",
                    "--config",
                    "llama3_debugmodel",
                    "--training.num_tokens_per_microbatch_per_dp_rank",
                    "0",
                ]
            )

    def test_num_tokens_per_train_step_must_be_positive_or_unset(self):
        with pytest.raises(ValueError, match="must be -1 or greater than 0"):
            TrainingConfig(num_tokens_per_train_step=0)

    def test_max_context_length_must_be_positive(self):
        for max_context_length in (0, -1):
            with pytest.raises(ValueError, match="must be greater than 0"):
                TrainingConfig(max_context_length=max_context_length)

    def test_num_pp_microbatches_does_not_constrain_non_pp_training(self):
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--parallelism.num_pp_microbatches",
                "3",
            ]
        )
        assert config.parallelism.pipeline_parallel_degree == 1
        assert config.parallelism.num_pp_microbatches == 3

    def test_cuda_graphs_allow_single_stage_pipeline_schedule(self):
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--parallelism.pipeline_parallel_degree",
                "2",
                "--parallelism.pipeline_parallel_schedule",
                "1F1B",
            ]
        )

        assert config.parallelism.pipeline_parallel_schedule == "1F1B"

    def test_cuda_graphs_allow_looped_pipeline_schedules(self):
        for schedule in (
            "Interleaved1F1B",
            "InterleavedZeroBubble",
            "ZBVZeroBubble",
        ):
            with self.subTest(schedule=schedule), cuda_graphs_supported(True):
                config = ConfigManager().parse_args(
                    [
                        "--module",
                        "llama3",
                        "--config",
                        "llama3_debugmodel",
                        "--training.disable_cuda_graphs",
                        "--parallelism.pipeline_parallel_degree",
                        "2",
                        "--parallelism.pipeline_parallel_schedule",
                        schedule,
                    ]
                )
                config.training.disable_cuda_graphs = False
                config.__post_init__()

    def test_cuda_graphs_reject_pipeline_validation(self):
        with cuda_graphs_supported(True):
            config = ConfigManager().parse_args(
                [
                    "--module",
                    "llama3",
                    "--config",
                    "llama3_debugmodel",
                    "--training.disable_cuda_graphs",
                    "--parallelism.pipeline_parallel_degree",
                    "2",
                ]
            )
            config.training.disable_cuda_graphs = False
            config.parallelism.pipeline_parallel_schedule = "1F1B"
            config.validator = Validator.Config()
            with pytest.raises(ValueError, match="do not support validation"):
                config.__post_init__()

    def test_cuda_graphs_enabled_by_default(self):
        config = ConfigManager().parse_args(
            ["--module", "llama3", "--config", "llama3_debugmodel"]
        )
        assert not config.training.disable_cuda_graphs

    def test_cuda_graphs_reject_unsupported_expert_parallelism(self):
        with cuda_graphs_supported(True):
            config_manager = ConfigManager()
            with mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
                with pytest.raises((ValueError, SystemExit)) as exc_info:
                    config_manager.parse_args(
                        [
                            "--module",
                            "deepseek_v3",
                            "--config",
                            "deepseek_v3_debugmodel",
                            "--parallelism.expert_parallel_degree",
                            "2",
                        ]
                    )

            if isinstance(exc_info.value, SystemExit):
                assert exc_info.value.code == 2
                error = stderr.getvalue()
            else:
                error = str(exc_info.value)
            assert "without CPU synchronization" in error

    def test_cuda_graphs_allow_non_blocking_hybridep(self):
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "deepseek_v3",
                "--config",
                "deepseek_v3_debugmodel_hybridep",
                "--parallelism.expert_parallel_degree",
                "2",
            ]
        )
        assert not config.training.disable_cuda_graphs

    def test_disable_cuda_graphs_allows_pipeline_parallelism(self):
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--training.disable_cuda_graphs",
                "--parallelism.pipeline_parallel_degree",
                "2",
            ]
        )
        assert config.training.disable_cuda_graphs

    def test_sdc_replay_requires_determinism(self):
        config = ConfigManager().parse_args(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--training.disable_cuda_graphs",
            ]
        )
        config.sdc_replayer = SDCReplayer.Config()

        with pytest.raises(ValueError, match="debug.deterministic=True"):
            TrainingEngine.Config.__post_init__(config)

        config.debug.deterministic = True
        config.debug.deterministic_warn_only = True
        with pytest.raises(ValueError, match="deterministic_warn_only=False"):
            TrainingEngine.Config.__post_init__(config)

    def test_microbatch_tokens_must_match_activation_sharding(self):
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

    def test_engine_rejects_spmd_typechecking_with_pipeline_parallelism(self):
        with pytest.raises(ValueError, match="SPMD typechecking"):
            TrainingEngine.Config(
                debug=DebugConfig(spmd_typechecking=True),
                training=TrainingConfig(disable_cuda_graphs=True),
                parallelism=ParallelismConfig(pipeline_parallel_degree=2),
            )

    def test_sdc_replay_is_off_the_cli(self):
        hints = typing.get_type_hints(Trainer.Config, include_extras=True)
        assert tyro.conf.Suppress in hints["sdc_replayer"].__metadata__

        config = ConfigManager().parse_args(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--debug.deterministic",
                "--training.disable_cuda_graphs",
            ]
        )
        config.sdc_replayer = SDCReplayer.Config(num_steps=3, num_replays=2)
        TrainingEngine.Config.__post_init__(config)

    def test_sdc_replay_rejects_multiple_replays_with_cuda_graphs(self):
        with cuda_graphs_supported(True):
            config = ConfigManager().parse_args(
                [
                    "--module",
                    "llama3",
                    "--config",
                    "llama3_debugmodel",
                    "--debug.deterministic",
                ]
            )
            config.sdc_replayer = SDCReplayer.Config(num_replays=2)

            with pytest.raises(ValueError, match="at most one replay"):
                TrainingEngine.Config.__post_init__(config)

    def test_sdc_replay_allows_multiple_replays_without_cuda_graphs(self):
        config = ConfigManager().parse_args(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--debug.deterministic",
                "--training.disable_cuda_graphs",
            ]
        )
        config.sdc_replayer = SDCReplayer.Config(num_replays=2)

        TrainingEngine.Config.__post_init__(config)

    def test_sdc_replay_accepts_execution_modes(self):
        config = ConfigManager().parse_args(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--debug.deterministic",
            ]
        )
        config.sdc_replayer = SDCReplayer.Config()
        config.parallelism.fsdp_symm_mem_scope = "all"
        config.compile = CompileConfig(enable_async_tensor_parallel=True)
        configs = {
            "symm_mem_async_tp": config,
            "distributed_gemm": llama3_debugmodel_dist_gemm(seq_len=2048),
            "hybrid_ep": deepseek_v3_debugmodel_hybridep(seq_len=2048),
            "deep_ep": qwen3_moe_deepep(seq_len=512),
        }

        for name, config in configs.items():
            with self.subTest(config=name):
                config.debug.deterministic = True
                config.sdc_replayer = SDCReplayer.Config()
                TrainingEngine.Config.__post_init__(config)

    def test_cuda_graphs_reject_blocking_hybridep(self):
        with cuda_graphs_supported(True):
            from torchtitan.models.common.token_dispatcher import (
                HybridEPTokenDispatcher,
            )
            from torchtitan.models.deepseek_v3.config_registry import (
                deepseek_v3_debugmodel_hybridep,
            )

            config = deepseek_v3_debugmodel_hybridep(seq_len=2048)
            dispatcher_configs = list(
                config.model.traverse(HybridEPTokenDispatcher.Config)
            )
            assert dispatcher_configs
            for _, dispatcher_config, _, _ in dispatcher_configs:
                dispatcher_config.non_blocking_capacity_factor = None
            config.parallelism.expert_parallel_degree = 2

            with pytest.raises(ValueError, match="non_blocking_capacity_factor"):
                dataclasses.replace(config)

    def test_cuda_graphs_unsupported_allows_looped_pipeline_schedule(self):
        """Where capture cannot run, the CUDA-graph gates must not fire.

        ROCm always falls back to eager in ``wrap_with_cuda_graph``, so every
        restriction below describes a constraint that does not exist there.
        """
        with cuda_graphs_supported(False):
            config = ConfigManager().parse_args(
                [
                    "--module",
                    "llama3",
                    "--config",
                    "llama3_debugmodel",
                    "--parallelism.pipeline_parallel_degree",
                    "2",
                    "--parallelism.pipeline_parallel_schedule",
                    "Interleaved1F1B",
                ]
            )

        assert config.parallelism.pipeline_parallel_schedule == "Interleaved1F1B"
        assert not config.training.disable_cuda_graphs

    def test_cuda_graphs_unsupported_allows_pipeline_validation(self):
        config = ConfigManager().parse_args(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--training.disable_cuda_graphs",
                "--parallelism.pipeline_parallel_degree",
                "2",
            ]
        )
        config.training.disable_cuda_graphs = False
        config.parallelism.pipeline_parallel_schedule = "1F1B"
        config.validator = Validator.Config()

        with cuda_graphs_supported(False):
            config.__post_init__()

    def test_cuda_graphs_unsupported_allows_expert_parallelism(self):
        with cuda_graphs_supported(False):
            config = ConfigManager().parse_args(
                [
                    "--module",
                    "deepseek_v3",
                    "--config",
                    "deepseek_v3_debugmodel",
                    "--parallelism.expert_parallel_degree",
                    "2",
                ]
            )

        assert config.parallelism.expert_parallel_degree == 2
        assert not config.training.disable_cuda_graphs

    def test_cuda_graphs_unsupported_allows_blocking_hybridep(self):
        from torchtitan.models.common.token_dispatcher import HybridEPTokenDispatcher

        config = deepseek_v3_debugmodel_hybridep(seq_len=2048)
        for _, dispatcher_config, _, _ in config.model.traverse(
            HybridEPTokenDispatcher.Config
        ):
            dispatcher_config.non_blocking_capacity_factor = None
        config.parallelism.expert_parallel_degree = 2

        with cuda_graphs_supported(False):
            dataclasses.replace(config)

    def test_cuda_graphs_unsupported_allows_multiple_sdc_replays(self):
        config = ConfigManager().parse_args(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--debug.deterministic",
            ]
        )
        config.sdc_replayer = SDCReplayer.Config(num_replays=2)

        with cuda_graphs_supported(False):
            TrainingEngine.Config.__post_init__(config)

    def test_cuda_graphs_reject_varlen_without_max_num_documents(self):
        from torchtitan.models.llama3.config_registry import (
            llama3_debugmodel_varlen_attn,
        )

        config = llama3_debugmodel_varlen_attn()
        config.dataloader.max_num_documents = None

        with cuda_graphs_supported(True):
            with pytest.raises(ValueError, match="max_num_documents is unset"):
                config.__post_init__()

    def test_cuda_graphs_unsupported_allows_varlen_without_max_num_documents(self):
        from torchtitan.models.llama3.config_registry import (
            llama3_debugmodel_varlen_attn,
        )

        config = llama3_debugmodel_varlen_attn()
        config.dataloader.max_num_documents = None

        with cuda_graphs_supported(False):
            config.__post_init__()

    def test_cli_override_dump_folder(self):
        """CLI args override config defaults for nested fields."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--dump_folder",
                "/tmp/test_tt/",
            ]
        )
        assert config.dump_folder == "/tmp/test_tt/"

    def test_parse_pipeline_parallel_module_fqns_per_model_part(self):
        """pipeline_parallel_module_fqns_per_model_part defaults to None."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            ["--module", "llama3", "--config", "llama3_debugmodel"]
        )
        assert config.parallelism.pipeline_parallel_module_fqns_per_model_part is None

    def test_optional_component_configs_do_not_add_cli_subcommands(self):
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            ["--module", "llama3", "--config", "llama3_debugmodel"]
        )
        assert config.checkpointer is None
        assert config.compile is None
        assert config.validator is None

        hints = typing.get_type_hints(Trainer.Config, include_extras=True)
        for field_name in ("checkpointer", "compile", "validator"):
            assert tyro.conf.AvoidSubcommands in hints[field_name].__metadata__
        assert tyro.conf.Suppress in hints["create_seed_checkpoint"].__metadata__

    def test_trainer_config_quantization_default(self):
        from torchtitan.quantization.utils import has_quantization

        config_manager = ConfigManager()
        config = config_manager.parse_args(
            ["--module", "llama3", "--config", "llama3_debugmodel"]
        )
        assert not has_quantization(config.model)

    # TODO: remove this test when we remove the merge functionality
    def test_extend_trainer_config_directly(self):
        """Test that _merge_configs works to extend config types."""
        from dataclasses import dataclass, field

        from torchtitan.trainer import Trainer

        @dataclass
        class CustomCheckpoint:
            convert_path: str = "/custom/path"
            fake_model: bool = True

        @dataclass
        class CustomTrainerConfig:
            checkpointer: CustomCheckpoint = field(default_factory=CustomCheckpoint)

        MergedTrainerConfig = ConfigManager._merge_configs(
            Trainer.Config, CustomTrainerConfig
        )

        # Verify the merged type has both base and custom fields
        model = (
            ConfigManager()
            .parse_args(["--module", "llama3", "--config", "llama3_debugmodel"])
            .model
        )
        merged = MergedTrainerConfig(model=model)
        assert hasattr(merged, "checkpointer")
        assert hasattr(merged.checkpointer, "convert_path")
        assert merged.checkpointer.convert_path == "/custom/path"
        assert merged.checkpointer.fake_model is True
        assert hasattr(merged, "model")

    def test_flux_config_via_cli(self):
        """Test that --module flux --config flux_debugmodel works."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            ["--module", "flux", "--config", "flux_debugmodel"]
        )
        assert type(config.model).__qualname__ == "FluxModel.Config"
        assert hasattr(config, "encoder")
        assert config.parallelism.context_parallel_load_balancer == "headtail"

    def test_default_context_parallel_load_balancer(self):
        assert ParallelismConfig().context_parallel_load_balancer == "headtail"

    def test_deepseek_config(self):
        """Test that --module deepseek_v3 --config deepseek_v3_debugmodel works."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            ["--module", "deepseek_v3", "--config", "deepseek_v3_debugmodel"]
        )
        assert type(config.model).__qualname__ == "DeepSeekV3Model.Config"

    def test_suppressed_model_is_opaque_to_tyro(self):
        config = ConfigManager().parse_args(
            [
                "--module",
                "torchtitan_recipes.tests.transformers_modeling_backend",
                "--config",
                "transformers_backend_dense_cp_pp",
            ]
        )

        assert type(config.model).__qualname__ == "HFTransformerModel.Config"

    def test_fqn_module_with_config_registry(self):
        """--module torchtitan.models.llama3.config_registry works."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "torchtitan.models.llama3.config_registry",
                "--config",
                "llama3_debugmodel",
            ]
        )
        assert type(config.model).__qualname__ == "Llama3Model.Config"

    def test_fqn_module_without_config_registry(self):
        """--module torchtitan.models.llama3 (auto-appends .config_registry)."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "torchtitan.models.llama3",
                "--config",
                "llama3_debugmodel",
            ]
        )
        assert type(config.model).__qualname__ == "Llama3Model.Config"

    def test_fqn_module_invalid_errors(self):
        """--module with invalid FQN raises ImportError."""
        config_manager = ConfigManager()
        with pytest.raises(ImportError, match="Cannot import module"):
            config_manager.parse_args(
                [
                    "--module",
                    "torchtitan.models.nonexistent",
                    "--config",
                    "foo",
                ]
            )


if __name__ == "__main__":
    unittest.main()
