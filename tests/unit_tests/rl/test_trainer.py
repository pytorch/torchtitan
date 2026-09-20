# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import torch
from torchtitan.components.data.types import (
    TrainingMicrobatch as CoreTrainingMicrobatch,
)
from torchtitan.config import (
    CompileConfig,
    Configurable,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.observability.sdc_replayer import SDCReplayer
from torchtitan.rl.distributed.actors.trainer import TrainerActor
from torchtitan.rl.trainer import Trainer
from torchtitan.rl.types import TrainingMicrobatch
from torchtitan.trainer import Trainer as DatasetTrainer
from torchtitan.training_engine import TrainingEngine


def test_trainer_has_thin_actor_adapter() -> None:
    assert issubclass(TrainerActor, Trainer)
    assert not issubclass(Trainer, DatasetTrainer)
    assert not issubclass(DatasetTrainer, TrainingEngine)
    assert issubclass(TrainingEngine, Configurable)
    assert issubclass(DatasetTrainer.Config, TrainingEngine.Config)
    assert issubclass(Trainer.Config, TrainingEngine.Config)
    assert issubclass(TrainingMicrobatch, CoreTrainingMicrobatch)


def test_rl_trainer_uses_training_engine_config_defaults() -> None:
    assert not Trainer.Config().training.disable_cuda_graphs


def test_pipeline_parallelism_is_rejected_until_weight_sync_supports_it() -> None:
    with pytest.raises(ValueError, match="TorchStore"):
        Trainer.Config(parallelism=ParallelismConfig(pipeline_parallel_degree=2))


def test_rl_trainer_accepts_core_sdc_replay_config() -> None:
    config = Trainer.Config(
        debug=DebugConfig(deterministic=True),
        training=TrainingConfig(disable_cuda_graphs=True),
        sdc_replayer=SDCReplayer.Config(),
    )

    assert isinstance(config.sdc_replayer, SDCReplayer.Config)


def test_rl_trainer_validates_model_training_config_before_initialization() -> None:
    class ValidationReachedError(Exception):
        pass

    config = Trainer.Config(training=TrainingConfig(disable_cuda_graphs=True))
    model_config = MagicMock()
    model_spec = SimpleNamespace(model=model_config)

    with patch(
        "torchtitan.rl.trainer.validate_model_training_config",
        side_effect=ValidationReachedError,
    ) as validate:
        with pytest.raises(ValidationReachedError):
            Trainer(
                config,
                model_spec=model_spec,
                compile_config=CompileConfig(),
                max_num_documents=None,
                output_dir="",
            )

    validate.assert_called_once_with(
        model_config,
        parallelism=config.parallelism,
        training=config.training,
        debug=config.debug,
        activation_checkpoint=config.activation_checkpoint,
        compile_config=CompileConfig(),
        max_num_documents=None,
    )


def test_aux_loss_denominator_uses_global_token_count() -> None:
    trainer = SimpleNamespace(
        device=torch.device("cpu"),
        num_completed_steps=0,
        gc_handler=SimpleNamespace(run=MagicMock()),
        optimizers=SimpleNamespace(zero_grad=MagicMock()),
        config=SimpleNamespace(training=SimpleNamespace(disable_cuda_graphs=True)),
        _deferred_cuda_graph_options=None,
    )

    with patch(
        "torchtitan.training_engine.AuxLoss.set_step_denominator"
    ) as set_denominator:
        denominator = TrainingEngine.prepare_step(trainer, 17, num_accumulation_steps=3)

    torch.testing.assert_close(denominator, torch.tensor(17, dtype=torch.int64))
    trainer.gc_handler.run.assert_called_once_with(1)
    assert trainer.num_accumulation_steps == 3
    set_denominator.assert_called_once_with(denominator)


def test_close_stops_training_engine() -> None:
    async def run() -> None:
        trainer = object.__new__(Trainer)
        engine = SimpleNamespace(
            close=MagicMock(),
        )
        trainer.engine = engine

        await Trainer.close(trainer)

        engine.close.assert_called_once_with()

    asyncio.run(run())


def test_policy_version_is_restored_with_training_engine_state() -> None:
    trainer = object.__new__(Trainer)
    engine = object.__new__(TrainingEngine)
    engine.num_completed_steps = 0
    engine.ntokens_seen = 0
    engine.sdc_replayer = None
    trainer.engine = engine

    engine.load_state_dict({"step": 7, "ntokens_seen": 128})

    assert trainer.policy_version == 7


def test_forward_backward_accumulates_microbatch_metrics() -> None:
    async def run() -> None:
        trainer = object.__new__(Trainer)
        global_valid_tokens = torch.tensor(3)
        engine = SimpleNamespace(
            device=torch.device("cpu"),
            num_completed_steps=4,
            ntokens_seen=10,
            prepare_step=MagicMock(return_value=global_valid_tokens),
            sdc_replayer=None,
        )
        mean_metric = torch.tensor(0.0)
        max_metric = torch.tensor(0.0)
        metric_values = iter([(1.0, 4.0), (2.0, 3.0)])

        def forward_backward_microbatch(**kwargs):
            mean_value, max_value = next(metric_values)
            mean_metric.fill_(mean_value)
            max_metric.fill_(max_value)
            engine.loss_metrics = {
                "loss/mean": mean_metric,
                "loss/max": max_metric,
            }
            return torch.tensor(0.5)

        engine.forward_backward_microbatch = MagicMock(
            side_effect=forward_backward_microbatch
        )
        trainer.engine = engine
        trainer.config = Trainer.Config()
        trainer.dp_rank = 0
        trainer._reduce_forward_backward_metrics = MagicMock(
            side_effect=lambda *, sum_reduced_metrics, max_reduced_metrics: {
                key: float(value.item())
                for key, value in {
                    **sum_reduced_metrics,
                    **max_reduced_metrics,
                }.items()
            }
        )
        batch = TrainingMicrobatch(
            input=torch.tensor([1]),
            labels=torch.tensor([2]),
            positions=torch.tensor([0]),
            padding_mask=torch.tensor([False]),
            num_valid_tokens=1,
            generator_logprobs=torch.tensor([0.0]),
            loss_mask=torch.tensor([True]),
            advantages=torch.tensor([1.0]),
        )

        result = await Trainer.forward_backward_steps(trainer, [[batch], [batch]], 3)

        engine.prepare_step.assert_called_once_with(3, num_accumulation_steps=2)
        assert engine.forward_backward_microbatch.call_count == 2
        assert [
            call.kwargs["accumulation_index"]
            for call in engine.forward_backward_microbatch.call_args_list
        ] == [0, 1]
        assert all(
            "num_accumulation_steps" not in call.kwargs
            for call in engine.forward_backward_microbatch.call_args_list
        )
        assert all(
            call.kwargs["global_valid_tokens"] is global_valid_tokens
            for call in engine.forward_backward_microbatch.call_args_list
        )
        assert all(
            "loss_kwargs" not in call.kwargs
            for call in engine.forward_backward_microbatch.call_args_list
        )
        assert trainer._step_num_tokens_per_dp_rank == 2
        assert trainer._reduce_forward_backward_metrics.call_count == 2
        assert result == {"loss/mean": 3.0, "loss/max": 4.0}

    asyncio.run(run())


def test_optimizer_step_advances_profiler_and_reports_aux_loss_metrics() -> None:
    async def run() -> None:
        trainer = object.__new__(Trainer)
        device_mem_stats = SimpleNamespace(
            max_active_gib=3.0,
            max_active_pct=4.0,
            max_reserved_gib=5.0,
            max_reserved_pct=6.0,
            num_alloc_retries=0,
            num_ooms=0,
        )
        device_memory_monitor = SimpleNamespace(
            get_peak_stats=MagicMock(return_value=device_mem_stats),
            reset_peak_stats=MagicMock(),
        )
        engine = SimpleNamespace(
            lr_schedulers=SimpleNamespace(
                get_metrics=MagicMock(
                    return_value={"lr/AdamW/0": 0.25, "lr/AdamW/1": 0.125}
                )
            ),
            parallel_dims=SimpleNamespace(non_data_parallel_size=1),
            num_completed_steps=4,
            ntokens_seen=12,
            num_flops_per_token=200,
            has_quantization=False,
            device_memory_monitor=device_memory_monitor,
            optimizer_step=MagicMock(return_value=torch.tensor(2.0)),
            save_checkpoint=MagicMock(),
            step_profiler=MagicMock(),
        )
        engine.optimizer_step.side_effect = lambda: (
            setattr(engine, "num_completed_steps", engine.num_completed_steps + 1),
            torch.tensor(2.0),
        )[1]
        trainer.engine = engine
        trainer.gpu_peak_flops = 1000
        trainer._step_compute_start = 0.0
        trainer._step_num_tokens_per_dp_rank = 10

        with (
            patch("torchtitan.rl.trainer.time.perf_counter", return_value=2.0),
            patch(
                "torchtitan.rl.trainer.collect_aux_loss_metrics",
                return_value={"aux_loss/mean": 0.5},
            ),
            patch(
                "torchtitan.rl.trainer.compute_training_performance_metrics",
                return_value={
                    "tokens_per_second": 10.0,
                    "tflops": 2.0,
                    "mfu_percent": 50.0,
                },
            ) as compute_performance,
        ):
            result = await Trainer.optimizer_step(trainer)

        assert result.policy_version == 5
        assert result.metrics == {
            "trainer/grad_norm/mean": 2.0,
            "trainer/lr/AdamW/0": 0.25,
            "trainer/lr/AdamW/1": 0.125,
            "trainer/policy_version": 5.0,
            "trainer/tokens_per_second": 10.0,
            "trainer/tflops": 2.0,
            "trainer/memory/max_active_gib": 3.0,
            "trainer/memory/max_active_percent": 4.0,
            "trainer/memory/max_reserved_gib": 5.0,
            "trainer/memory/max_reserved_percent": 6.0,
            "trainer/memory/num_alloc_retries": 0.0,
            "trainer/memory/num_ooms": 0.0,
            "trainer/mfu_percent": 50.0,
            "aux_loss/mean": 0.5,
        }
        engine.optimizer_step.assert_called_once_with()
        engine.save_checkpoint.assert_called_once_with(last_step=False)
        engine.step_profiler.assert_called_once_with()
        compute_performance.assert_called_once_with(
            num_tokens=10,
            elapsed_time=2.0,
            non_data_parallel_size=1,
            num_flops_per_token=200,
            gpu_peak_flops=1000,
            has_quantization=False,
        )
        device_memory_monitor.get_peak_stats.assert_called_once_with()
        device_memory_monitor.reset_peak_stats.assert_called_once_with()

    asyncio.run(run())
