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
from torchtitan.training_engine import ForwardBackwardResult, TrainingEngine


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

    with patch(
        "torchtitan.rl.trainer.validate_model_training_config",
        side_effect=ValidationReachedError,
    ) as validate:
        with pytest.raises(ValidationReachedError):
            Trainer(
                config,
                model_config=model_config,
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


def test_forward_backward_uses_global_token_count() -> None:
    engine = SimpleNamespace(
        device=torch.device("cpu"),
        num_completed_steps=0,
        gc_handler=SimpleNamespace(run=MagicMock()),
        optimizers=SimpleNamespace(zero_grad=MagicMock()),
        config=SimpleNamespace(
            training=SimpleNamespace(disable_cuda_graphs=True),
            parallelism=SimpleNamespace(
                fsdp_defer_gradient_reduction=False,
                fsdp_reshard_after_forward="default",
            ),
        ),
        parallel_dims=SimpleNamespace(fsdp_enabled=False),
        _preprocess_microbatch_group=MagicMock(return_value=()),
        _run_forward_backward=MagicMock(
            return_value=ForwardBackwardResult(torch.tensor(1.0), [])
        ),
        sdc_replayer=None,
    )
    microbatch_groups = [[object()], [object()], [object()]]

    with patch(
        "torchtitan.training_engine.AuxLoss.set_step_denominator"
    ) as set_denominator:
        result = TrainingEngine.forward_backward(
            engine,
            microbatch_groups=microbatch_groups,
            global_valid_tokens=17,
        )

    global_valid_tokens = set_denominator.call_args.args[0]
    torch.testing.assert_close(global_valid_tokens, torch.tensor(17, dtype=torch.int64))
    torch.testing.assert_close(result.loss, torch.tensor(1.0))
    engine.gc_handler.run.assert_called_once_with(1)
    engine.optimizers.zero_grad.assert_called_once_with(set_to_none=True)
    assert engine.num_accumulation_steps == 3
    assert engine._preprocess_microbatch_group.call_count == 3


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
        engine = SimpleNamespace(
            num_completed_steps=4,
            ntokens_seen=10,
            sdc_replayer=None,
        )
        engine.forward_backward = MagicMock(
            return_value=ForwardBackwardResult(
                loss=torch.tensor(0.5),
                loss_metrics=[
                    {
                        "loss/mean": torch.tensor(1.0),
                        "loss/max": torch.tensor(4.0),
                    },
                    {
                        "loss/mean": torch.tensor(2.0),
                        "loss/max": torch.tensor(3.0),
                    },
                ],
            )
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

        engine.forward_backward.assert_called_once_with(
            microbatch_groups=[[batch], [batch]],
            global_valid_tokens=3,
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
