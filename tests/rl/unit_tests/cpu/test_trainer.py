# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import threading
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import torch
from torchstore import RankRole
from torchtitan.components.data.types import (
    TrainingMicrobatch as CoreTrainingMicrobatch,
)
from torchtitan.config import Configurable, DebugConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.observability.sdc_replayer import SDCReplayer
from torchtitan.rl.components.checkpointer import MirroredState
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


def test_mirrored_state_uses_a_fixed_opaque_template() -> None:
    source = MirroredState()
    assert source.state_dict().keys() == {"state"}
    source.value = {"next_index": 4, "pending_indices": [1, 3]}

    restored = MirroredState()
    restored.load_state_dict(source.state_dict())

    assert restored.loaded == source.value
    assert restored.state_dict().keys() == {"state"}


def test_rl_trainer_uses_training_engine_config_defaults() -> None:
    assert not Trainer.Config().training.disable_cuda_graphs


def test_pipeline_parallelism_is_rejected_until_weight_sync_supports_it() -> None:
    with pytest.raises(ValueError, match="TorchStore"):
        Trainer.Config(parallelism=ParallelismConfig(pipeline_parallel_degree=2))


def test_initialize_torchstore_client_uses_publisher_role() -> None:
    async def run() -> None:
        trainer = object.__new__(Trainer)
        with patch("torchtitan.rl.trainer.ts.client", new_callable=AsyncMock) as client:
            await trainer.initialize_torchstore_client()

        client.assert_awaited_once_with(role=RankRole.PUBLISHER)

    asyncio.run(run())


def test_push_model_state_dict_reuses_final_dtype_cpu_staging_buffers() -> None:
    async def run() -> None:
        trainer = object.__new__(Trainer)
        trainer.engine = SimpleNamespace(device=torch.device("cuda", 0))
        trainer.model = torch.nn.Linear(2, 2, bias=False)
        trainer.model.register_buffer("scale", torch.tensor(3.0, dtype=torch.float32))
        trainer._transfer_dtype = torch.bfloat16
        trainer._staging_model_state_dict_stream = None

        empty_like = torch.empty_like
        pin_memory_requests = []

        def allocate_without_pinning(*args, **kwargs):
            pin_memory_requests.append(kwargs.get("pin_memory", False))
            kwargs["pin_memory"] = False
            return empty_like(*args, **kwargs)

        with patch(
            "torchtitan.rl.trainer.torch.empty_like",
            side_effect=allocate_without_pinning,
        ):
            trainer._staging_state_dict = trainer._setup_staging_model_state_dict()

        staging = trainer._staging_state_dict
        weight_data_ptr = staging["weight"].data_ptr()
        scale_data_ptr = staging["scale"].data_ptr()
        assert set(staging) == {"weight", "scale"}
        assert staging["weight"].device.type == "cpu"
        assert staging["weight"].dtype == torch.bfloat16
        assert staging["scale"].dtype == torch.float32
        assert pin_memory_requests == [True, True]

        with patch(
            "torchtitan.rl.trainer.ts.put_state_dict", new_callable=AsyncMock
        ) as put_state_dict:
            await trainer.push_model_state_dict()
            trainer.model.weight.data.fill_(2.0)
            await trainer.push_model_state_dict()

        assert staging["weight"].data_ptr() == weight_data_ptr
        torch.testing.assert_close(
            staging["weight"],
            torch.full((2, 2), 2.0, dtype=torch.bfloat16),
        )
        assert put_state_dict.await_count == 2
        for call in put_state_dict.await_args_list:
            assert call.args[0] is staging
            assert call.args[1] == "model_state_dict"
            assert call.kwargs == {"direct_rdma": False}
        assert staging["scale"].data_ptr() == scale_data_ptr
        torch.testing.assert_close(staging["scale"], torch.tensor(3.0))

    asyncio.run(run())


def test_push_model_state_dict_stages_on_dedicated_cuda_stream() -> None:
    async def run() -> None:
        trainer = object.__new__(Trainer)
        trainer._staging_state_dict = {"weight": torch.empty(4)}
        trainer._staging_model_state_dict_stream = MagicMock()
        trainer._staging_device_module = MagicMock()
        source = torch.arange(4, dtype=torch.float32)
        trainer.model = SimpleNamespace(
            state_dict=MagicMock(return_value={"weight": source})
        )
        staging_complete = MagicMock()
        trainer._staging_device_module.stream.return_value = nullcontext()
        trainer._staging_device_module.Event.return_value = staging_complete
        main_thread_id = threading.get_ident()
        worker_thread_ids = []

        def synchronize() -> None:
            worker_thread_ids.append(threading.get_ident())

        async def put_state_dict(*args, **kwargs) -> None:
            worker_thread_ids.append(threading.get_ident())

        staging_complete.synchronize.side_effect = synchronize

        with patch(
            "torchtitan.rl.trainer.ts.put_state_dict",
            new_callable=AsyncMock,
            side_effect=put_state_dict,
        ) as put:
            await trainer.push_model_state_dict()

        staging_complete.record.assert_called_once_with(
            trainer._staging_model_state_dict_stream
        )
        staging_complete.synchronize.assert_called_once_with()
        put.assert_awaited_once_with(
            trainer._staging_state_dict,
            "model_state_dict",
            direct_rdma=False,
        )
        assert len(worker_thread_ids) == 2
        assert worker_thread_ids[0] == worker_thread_ids[1]
        assert worker_thread_ids[0] != main_thread_id
        torch.testing.assert_close(trainer._staging_state_dict["weight"], source)

    asyncio.run(run())


def test_push_staging_copies_parameters_already_in_transfer_dtype() -> None:
    trainer = object.__new__(Trainer)
    trainer.engine = SimpleNamespace(device=torch.device("cuda", 0))
    trainer.model = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    trainer._transfer_dtype = torch.bfloat16

    empty_like = torch.empty_like
    pin_memory_requests = []

    def allocate_without_pinning(*args, **kwargs):
        pin_memory_requests.append(kwargs.get("pin_memory", False))
        kwargs["pin_memory"] = False
        return empty_like(*args, **kwargs)

    with patch(
        "torchtitan.rl.trainer.torch.empty_like",
        side_effect=allocate_without_pinning,
    ):
        staging = trainer._setup_staging_model_state_dict()

    assert set(staging) == {"weight"}
    assert staging["weight"].dtype == torch.bfloat16
    assert pin_memory_requests == [True]


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
                max_num_documents=None,
                output_dir="",
            )

    validate.assert_called_once_with(
        model_config,
        parallelism=config.parallelism,
        training=config.training,
        debug=config.debug,
        activation_checkpoint=config.activation_checkpoint,
        max_num_documents=None,
    )


def test_forward_backward_uses_global_token_count() -> None:
    engine = SimpleNamespace(
        device=torch.device("cpu"),
        num_completed_steps=0,
        garbage_collector=SimpleNamespace(run=MagicMock()),
        optim=SimpleNamespace(zero_grad=MagicMock()),
        config=SimpleNamespace(
            training=SimpleNamespace(disable_cuda_graphs=True),
            parallelism=SimpleNamespace(
                fsdp_defer_gradient_reduction=False,
                fsdp_reshard_after_forward="default",
            ),
        ),
        parallelism_context=SimpleNamespace(
            fsdp_enabled=False, dp_enabled=False, pp_enabled=False
        ),
        model_parts=[],
        _preprocess_microbatch_groups=MagicMock(
            return_value=[("input", "labels", {}, {}) for _ in range(3)]
        ),
        _run_forward_backward=MagicMock(
            return_value=ForwardBackwardResult(torch.tensor(1.0), [])
        ),
        _cuda_graph_per_accumulation_group_enabled=False,
        sdc_replayer=None,
    )
    microbatch_groups = [[object()], [object()], [object()]]

    result = TrainingEngine.forward_backward(
        engine,
        microbatch_groups=microbatch_groups,
        global_loss_token_counts=17,
        global_routing_token_counts=torch.tensor([19]),
    )

    torch.testing.assert_close(result.loss, torch.tensor(1.0))
    engine.garbage_collector.run.assert_called_once_with(1)
    engine.optim.zero_grad.assert_called_once_with(set_to_none=True)
    assert engine.num_accumulation_steps == 3
    engine._preprocess_microbatch_groups.assert_called_once_with(microbatch_groups)


def test_close_stops_training_engine() -> None:
    async def run() -> None:
        trainer = object.__new__(Trainer)
        engine = SimpleNamespace(
            close=MagicMock(),
        )
        trainer.engine = engine
        trainer._controller_state = MirroredState()

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


def test_get_controller_state_returns_restored_state() -> None:
    async def run() -> None:
        trainer = object.__new__(Trainer)
        trainer._controller_state = MirroredState()
        trainer._controller_state.loaded = {"dataloader": {"next_index": 11}}

        assert await trainer.get_controller_state() == {
            "dataloader": {"next_index": 11}
        }

    asyncio.run(run())


def test_forward_backward_accumulates_microbatch_metrics() -> None:
    async def run() -> None:
        trainer = object.__new__(Trainer)
        engine = SimpleNamespace(
            num_completed_steps=4,
            ntokens_seen=10,
            sdc_replayer=None,
            model_config=SimpleNamespace(mtp_layers=None),
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
            loss_token_counts=torch.tensor([1]),
            routing_token_counts=torch.tensor([1]),
            generator_logprobs=torch.tensor([0.0]),
            temperature=torch.tensor([1.0]),
            loss_mask=torch.tensor([True]),
            advantages=torch.tensor([1.0]),
        )

        result = await Trainer.forward_backward(
            trainer,
            [[batch], [batch]],
            torch.tensor([3]),
            torch.tensor([2]),
        )

        engine.forward_backward.assert_called_once_with(
            microbatch_groups=[[batch], [batch]],
            global_loss_token_counts=torch.tensor([3]),
            global_routing_token_counts=torch.tensor([2]),
        )
        assert trainer._step_num_tokens_per_dp_rank == 2
        assert trainer._reduce_forward_backward_metrics.call_count == 2
        assert result == {"loss/mean": 3.0, "loss/max": 4.0}

    asyncio.run(run())


def test_optim_step_advances_profiler_and_reports_aux_loss_metrics() -> None:
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
            optim=SimpleNamespace(
                lr_schedulers=SimpleNamespace(
                    get_metrics=MagicMock(
                        return_value={"lr/AdamW/0": 0.25, "lr/AdamW/1": 0.125}
                    )
                )
            ),
            parallelism_context=SimpleNamespace(non_data_parallel_size=1),
            num_completed_steps=4,
            ntokens_seen=12,
            num_flops_per_token=200,
            has_quantization=False,
            device=torch.device("cuda", 0),
            device_memory_monitor=device_memory_monitor,
            optim_step=MagicMock(return_value=torch.tensor(2.0)),
            save_checkpoint=MagicMock(),
            step_profiler=MagicMock(),
        )
        engine.optim_step.side_effect = lambda: (
            setattr(engine, "num_completed_steps", engine.num_completed_steps + 1),
            torch.tensor(2.0),
        )[1]
        trainer.engine = engine
        trainer._controller_state = MirroredState()
        trainer.gpu_peak_flops = 1000
        trainer._step_compute_start = 0.0
        trainer._step_num_tokens_per_dp_rank = 10
        trainer._staging_model_state_dict_stream = MagicMock()
        trainer._staging_device_module = MagicMock()
        compute_stream = MagicMock()
        trainer._staging_device_module.current_stream.return_value = compute_stream

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
            result = await Trainer.optim_step(
                trainer,
                controller_state={"dataloader": {"next_index": 3}},
            )

        assert result.policy_version == 5
        trainer._staging_model_state_dict_stream.wait_stream.assert_called_once_with(
            compute_stream
        )
        assert result.metrics == {
            "trainer/grad_norm/mean": 2.0,
            "trainer/lr/AdamW/0": 0.25,
            "trainer/lr/AdamW/1": 0.125,
            "trainer/policy_version": 5.0,
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
        engine.optim_step.assert_called_once_with()
        assert trainer._controller_state.value == {"dataloader": {"next_index": 3}}
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
