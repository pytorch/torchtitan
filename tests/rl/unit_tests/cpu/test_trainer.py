# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import threading
from collections import deque
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

import torch
from torchstore import RankRole
from torchtitan.config import PrecisionConfig, TrainingConfig
from torchtitan.config.transform import LoRATransform
from torchtitan.models.base.module import Module
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.rl.trainer.batcher import TrainingMicrobatch
from torchtitan.rl.trainer.microbatch_group_reader import (
    MicrobatchGroupReader,
    ReplicaColumn,
)
from torchtitan.rl.trainer.trainer import MirroredState, Trainer
from torchtitan.training_engine import ForwardBackwardResult, TrainingEngine

from tests.rl.unit_tests.cpu.test_microbatch_group_reader import (
    fake_column,
    fake_training_sample_builder,
)


def test_mirrored_state_uses_a_fixed_opaque_template() -> None:
    source = MirroredState()
    assert source.state_dict().keys() == {"state"}
    source.value = {"next_index": 4, "pending_indices": [1, 3]}

    restored = MirroredState()
    restored.load_state_dict(source.state_dict())

    assert restored.loaded == source.value
    assert restored.state_dict().keys() == {"state"}


def test_initialize_torchstore_client_uses_publisher_role() -> None:
    async def run() -> None:
        trainer = object.__new__(Trainer)
        with patch(
            "torchtitan.rl.trainer.trainer.ts.client", new_callable=AsyncMock
        ) as client:
            await trainer.initialize_torchstore_client()

        client.assert_awaited_once_with(role=RankRole.PUBLISHER)

    asyncio.run(run())


def test_push_model_state_dict_puts_on_push_stream_in_worker() -> None:
    async def run() -> None:
        trainer = object.__new__(Trainer)
        trainer._model_state_dict_transport = None
        # Model parts are torchtitan Modules (BaseModel in the engine).
        weight_part = Module()
        weight_part.register_parameter("weight", torch.nn.Parameter(torch.ones(2, 2)))
        buffer_part = Module()
        buffer_part.norm = RMSNorm.Config(
            normalized_shape=2, preserve_dtype_params=("weight",)
        ).build()
        buffer_part.norm.register_buffer("scale", torch.tensor(3.0))
        buffer_part.register_buffer("cache", torch.ones(2), persistent=False)
        # Under PP, a rank publishes the merged state dicts of its model parts.
        trainer.engine = SimpleNamespace(model_parts=[weight_part, buffer_part])
        trainer._transfer_dtype = torch.bfloat16
        stream = MagicMock()
        trainer._push_model_state_dict_stream = stream
        main_thread_id = threading.get_ident()
        worker_thread_ids = []

        def enter_stream() -> None:
            worker_thread_ids.append(threading.get_ident())

        async def put_state_dict(*args, **kwargs) -> None:
            # The put must run while the push stream is current and idle.
            stream.__exit__.assert_not_called()
            stream.synchronize.assert_called_once()
            worker_thread_ids.append(threading.get_ident())

        stream.__enter__.side_effect = enter_stream
        with patch(
            "torchtitan.rl.trainer.trainer.ts.put_state_dict",
            new_callable=AsyncMock,
            side_effect=put_state_dict,
        ) as put:
            await trainer.push_model_state_dict()

        put.assert_awaited_once_with(
            ANY,
            "model_state_dict",
            direct_rdma=False,
            transfer_dtype=torch.bfloat16,
            preserve_dtype_keys=frozenset({"norm.weight", "norm.scale"}),
        )
        # The live model tensors are put; TorchStore owns the CPU copy.
        state_dict = put.await_args.args[0]
        assert set(state_dict) == {"weight", "norm.weight", "norm.scale"}
        assert state_dict["weight"].data_ptr() == weight_part.weight.data_ptr()
        assert state_dict["norm.scale"].data_ptr() == buffer_part.norm.scale.data_ptr()
        stream.__exit__.assert_called_once()
        assert len(worker_thread_ids) == 2
        assert worker_thread_ids[0] == worker_thread_ids[1]
        assert worker_thread_ids[0] != main_thread_id

    asyncio.run(run())


def test_push_model_state_dict_carries_the_trainable_params_and_the_buffers() -> None:
    """A push leaves out the frozen parameters, which generators load from the
    checkpoint: the LoRA base, or a router gate with freeze_gate."""

    async def push(model_part: Module) -> set[str]:
        model_part.register_buffer("expert_bias", torch.zeros(2))
        trainer = object.__new__(Trainer)
        trainer.engine = SimpleNamespace(model_parts=[model_part])
        trainer._transfer_dtype = None
        trainer._push_model_state_dict_stream = None
        trainer._model_state_dict_transport = None
        with patch(
            "torchtitan.rl.trainer.trainer.ts.put_state_dict", new_callable=AsyncMock
        ) as put:
            await trainer.push_model_state_dict()
        return set(put.await_args.args[0])

    lora_part = (
        LoRATransform(rank=1, alpha=1.0)
        .transform(Linear.Config(in_features=2, out_features=2))
        .build()
    )
    frozen_part = Module()
    frozen_part.gate = torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)

    assert asyncio.run(push(lora_part)) == {
        "lora_a.weight",
        "lora_b.weight",
        "expert_bias",
    }
    assert asyncio.run(push(frozen_part)) == {"expert_bias"}


def test_push_model_state_dict_stages_on_push_stream() -> None:
    async def run() -> None:
        trainer = object.__new__(Trainer)
        weight_part = Module()
        weight_part.register_parameter("weight", torch.nn.Parameter(torch.ones(2, 2)))
        trainer.engine = SimpleNamespace(model_parts=[weight_part])
        stream = MagicMock()
        trainer._push_model_state_dict_stream = stream
        main_thread_id = threading.get_ident()
        staged = []
        sync_thread_ids = []

        def stage(model_state_dict) -> None:
            # Staging runs on the push stream from the actor thread.
            stream.__enter__.assert_called_once()
            stream.__exit__.assert_not_called()
            assert threading.get_ident() == main_thread_id
            staged.append(model_state_dict)

        stream.synchronize.side_effect = lambda: sync_thread_ids.append(
            threading.get_ident()
        )
        trainer._model_state_dict_transport = SimpleNamespace(stage=stage)

        await trainer.push_model_state_dict()
        assert len(staged) == 1
        assert staged[0]["weight"].data_ptr() == weight_part.weight.data_ptr()
        stream.__exit__.assert_called_once()
        # The wait for staging is off the event loop.
        assert len(sync_thread_ids) == 1
        assert sync_thread_ids[0] != main_thread_id

    asyncio.run(run())


def test_rl_trainer_validates_model_training_config_before_initialization(
    tmp_path,
) -> None:
    class ValidationReachedError(Exception):
        pass

    model_config = MagicMock()
    config = Trainer.Config(
        model=model_config, training=TrainingConfig(disable_accelerator_graphs=True)
    )

    with patch(
        "torchtitan.rl.trainer.trainer.validate_model_training_config",
        side_effect=ValidationReachedError,
    ) as validate:
        with pytest.raises(ValidationReachedError):
            Trainer(
                config,
                precision=PrecisionConfig(),
                max_num_documents=None,
                output_dir=str(tmp_path),
                training_sample_builder=MagicMock(),
                microbatch_packer=MagicMock(),
                pad_id=0,
                temperature=1.0,
                vision_processor=None,
                rollout_groups_by_rdma=False,
            )

    validate.assert_called_once_with(
        model_config,
        parallelism=config.parallelism,
        training=config.training,
        debug=config.debug,
        activation_remat=config.activation_remat,
        max_num_documents=None,
    )


def test_forward_backward_uses_global_token_count() -> None:
    engine = SimpleNamespace(
        device=torch.device("cpu"),
        num_completed_steps=0,
        garbage_collector=SimpleNamespace(run=MagicMock()),
        optim=SimpleNamespace(zero_grad=MagicMock()),
        config=SimpleNamespace(
            training=SimpleNamespace(
                disable_accelerator_graphs=True,
                preprocess_microbatch_groups_lazily=False,
            ),
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
        _run_forward_backward=MagicMock(return_value=(torch.tensor(1.0), [])),
        _accelerator_graph_per_microbatch_group_enabled=False,
        sdc_replayer=None,
        flops_estimator=lambda inputs: inputs["input"].numel(),
    )
    microbatch_groups = [[_rl_microbatch(token)] for token in range(3)]

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
    engine._preprocess_microbatch_groups.assert_called_once_with(
        microbatch_groups, global_routing_token_counts=torch.tensor([19])
    )


def test_close_stops_training_engine() -> None:
    async def run() -> None:
        trainer = object.__new__(Trainer)
        engine = SimpleNamespace(
            close=MagicMock(),
        )
        trainer.engine = engine
        trainer._resume_state = MirroredState()
        trainer._started_reader = None

        await Trainer.close(trainer)

        engine.close.assert_called_once_with()

    asyncio.run(run())


def test_check_rdma_probes_reads_off_the_event_loop(monkeypatch) -> None:
    """The probe reads block; on a worker thread, one that hangs past the controller's timeout
    leaves the leader actor's event loop free for later calls."""
    trainer = object.__new__(Trainer)
    trainer._rdma_byte_buffers = "buffers"
    reads = []
    monkeypatch.setattr(
        "torchtitan.rl.trainer.trainer.check_rdma_probes",
        lambda buffers, probes, **_: reads.append(
            (buffers, threading.current_thread())
        ),
    )

    asyncio.run(trainer.check_rdma_probes(["probe"]))

    [(buffers, thread)] = reads
    assert buffers == "buffers" and thread is not threading.main_thread()


def test_policy_version_is_restored_with_training_engine_state() -> None:
    trainer = object.__new__(Trainer)
    engine = object.__new__(TrainingEngine)
    engine.num_completed_steps = 0
    engine.ntokens_seen = 0
    engine.sdc_replayer = None
    trainer.engine = engine

    engine.load_state_dict({"step": 7, "ntokens_seen": 128})

    assert trainer.policy_version == 7


def test_get_resume_state_returns_restored_state() -> None:
    async def run() -> None:
        trainer = object.__new__(Trainer)
        trainer._resume_state = MirroredState()
        trainer._resume_state.loaded = {"next_index": 11}

        assert await trainer.get_resume_state() == {"next_index": 11}

    asyncio.run(run())


def _leader(
    trainer: Trainer, packed: list[object], *, num_microbatches_per_group: int = 1
) -> None:
    """Make `trainer` its replica's only rank, whose column packs to `packed`, in order."""
    trainer._replica_group = None
    trainer._is_replica_leader = True
    trainer._replica_columns = deque()
    trainer._rollout_groups_by_rdma = False
    trainer._rdma_byte_buffers = None
    trainer._started_reader = None
    trainer._num_microbatches_per_group = num_microbatches_per_group
    trainer._training_sample_builder = fake_training_sample_builder
    microbatches = iter(packed)
    trainer._microbatch_packer = SimpleNamespace(
        pack_planned=lambda specs, samples: next(microbatches)
    )


def _column(num_groups: int, num_microbatches_per_group: int = 1) -> ReplicaColumn:
    return fake_column(
        [
            [
                [group * num_microbatches_per_group + index]
                for index in range(num_microbatches_per_group)
            ]
            for group in range(num_groups)
        ]
    )


def test_forward_backward_accumulates_microbatch_metrics() -> None:
    async def run() -> None:
        trainer = object.__new__(Trainer)
        trainer._model_state_dict_transport = None
        engine = SimpleNamespace(
            num_completed_steps=4,
            ntokens_seen=10,
            sdc_replayer=None,
            model_config=SimpleNamespace(mtp_layers=None),
            device=torch.device("cpu"),
            pp_has_last_stage=True,
        )
        trainer.engine = engine
        trainer.config = Trainer.Config(model=MagicMock())
        trainer.dp_rank = 1
        batch_0, batch_1 = _rl_microbatch(0), _rl_microbatch(1)
        _leader(trainer, [batch_0, batch_1])
        trained = []

        def forward_backward(*, microbatch_groups, **kwargs) -> ForwardBackwardResult:
            assert len(microbatch_groups) == 2
            trained.extend(microbatch_groups)
            return ForwardBackwardResult(
                loss=torch.tensor(0.5),
                loss_metrics=[
                    [
                        {
                            "loss/mean": torch.tensor(1.0),
                            "loss/max": torch.tensor(4.0),
                        }
                    ],
                    [
                        {
                            "loss/mean": torch.tensor(2.0),
                            "loss/max": torch.tensor(3.0),
                        }
                    ],
                ],
                num_flops=0,
            )

        engine.forward_backward = MagicMock(side_effect=forward_backward)
        trainer._reduce_forward_backward_metrics = MagicMock(
            side_effect=lambda *, sum_reduced_metrics, max_reduced_metrics: {
                key: float(value.item())
                for key, value in {
                    **sum_reduced_metrics,
                    **max_reduced_metrics,
                }.items()
            }
        )

        await trainer.receive_replica_column(_column(2))
        result = await Trainer.forward_backward(
            trainer,
            num_microbatch_groups=2,
            global_loss_token_counts=torch.tensor([3]),
            global_routing_token_counts=torch.tensor([2]),
        )

        engine.forward_backward.assert_called_once_with(
            microbatch_groups=ANY,
            global_loss_token_counts=torch.tensor([3]),
            global_routing_token_counts=torch.tensor([2]),
        )
        # The leader packs its column, in order, through the reader.
        groups = engine.forward_backward.call_args.kwargs["microbatch_groups"]
        assert isinstance(groups, MicrobatchGroupReader)
        assert trained == [[batch_0], [batch_1]]
        assert not trainer._replica_columns
        assert trainer._step_num_tokens_per_dp_rank == (
            2 * trainer.config.training.num_tokens_per_microbatch_per_dp_rank
        )
        assert result.pop("timing/trainer/pack_microbatches/max") > 0
        # No other rank in the replica: nothing to broadcast.
        assert result.pop("timing/trainer/broadcast_microbatches/max") == 0
        # Groups go by value (`rdma_min_group_bytes=None`): no read metrics.
        # One reduction per microbatch, then one for the timings.
        assert trainer._reduce_forward_backward_metrics.call_count == 3
        assert result == {"loss/mean": 3.0, "loss/max": 4.0}

    asyncio.run(run())


def _rl_microbatch(token: int) -> TrainingMicrobatch:
    return TrainingMicrobatch(
        input=torch.tensor([token]),
        labels=torch.tensor([token + 1]),
        positions=torch.tensor([0]),
        padding_mask=torch.tensor([False]),
        loss_token_counts=torch.tensor([1]),
        routing_token_counts=torch.tensor([1]),
        generator_logprobs=torch.tensor([0.0]),
        loss_mask=torch.tensor([True]),
        advantages=torch.tensor([1.0]),
        temperature=torch.tensor([0.7]),
    )


def test_forward_backward_preprocesses_images_on_the_reader_thread() -> None:
    async def run() -> None:
        trainer = object.__new__(Trainer)
        engine = SimpleNamespace(num_completed_steps=0, pp_has_last_stage=False)
        trained = []
        engine.forward_backward = MagicMock(
            side_effect=lambda *, microbatch_groups, **kwargs: (
                trained.extend(microbatch_groups)
                or ForwardBackwardResult(
                    loss=torch.tensor(-1.0), loss_metrics=[[{}]], num_flops=0
                )
            )
        )
        trainer.engine = engine
        trainer.config = Trainer.Config(model=MagicMock())
        trainer.dp_rank = 0
        preprocessed = _rl_microbatch(1)
        microbatch = MagicMock(spec=TrainingMicrobatch)
        threads = []
        microbatch.preprocess_images.side_effect = lambda: (
            threads.append(threading.current_thread()) or preprocessed
        )
        _leader(trainer, [microbatch])
        await trainer.receive_replica_column(_column(1))

        await Trainer.forward_backward(
            trainer,
            num_microbatch_groups=1,
            global_loss_token_counts=torch.tensor([1]),
            global_routing_token_counts=torch.tensor([1]),
        )

        assert trained == [[preprocessed]]
        assert threads and threads[0] is not threading.main_thread()

    asyncio.run(run())


def test_forward_backward_without_last_pipeline_stage_returns_no_metrics() -> None:
    async def run() -> None:
        trainer = object.__new__(Trainer)
        trainer._model_state_dict_transport = None
        engine = SimpleNamespace(num_completed_steps=0, pp_has_last_stage=False)
        trained = []
        engine.forward_backward = MagicMock(
            side_effect=lambda *, microbatch_groups, **kwargs: (
                trained.extend(microbatch_groups)
                or ForwardBackwardResult(
                    loss=torch.tensor(-1.0),
                    loss_metrics=[[{}, {}], [{}, {}]],
                    num_flops=0,
                )
            )
        )
        trainer.engine = engine
        trainer.config = Trainer.Config(model=MagicMock())
        trainer.dp_rank = 1
        trainer._reduce_forward_backward_metrics = MagicMock()
        # This rank's column: two pipeline steps of two microbatches each.
        microbatches = [_rl_microbatch(token) for token in range(4)]
        _leader(trainer, microbatches, num_microbatches_per_group=2)
        await trainer.receive_replica_column(_column(2, num_microbatches_per_group=2))

        result = await Trainer.forward_backward(
            trainer,
            num_microbatch_groups=2,
            global_loss_token_counts=torch.tensor([3]),
            global_routing_token_counts=torch.tensor([2]),
        )

        assert trained == [microbatches[:2], microbatches[2:]]
        assert trainer._step_num_tokens_per_dp_rank == (
            4 * trainer.config.training.num_tokens_per_microbatch_per_dp_rank
        )
        trainer._reduce_forward_backward_metrics.assert_not_called()
        assert result is None

    asyncio.run(run())


def test_start_reading_packs_before_forward_backward_trains_on_it() -> None:
    """`start_reading` returns at once and the leader packs the next step's first groups on the
    reader thread before `forward_backward` is called; `forward_backward` then trains on that
    reader, in order."""

    async def run() -> None:
        trainer = object.__new__(Trainer)
        engine = SimpleNamespace(num_completed_steps=0, pp_has_last_stage=False)
        trained = []
        engine.forward_backward = MagicMock(
            side_effect=lambda *, microbatch_groups, **kwargs: (
                trained.extend(microbatch_groups)
                or ForwardBackwardResult(
                    loss=torch.tensor(-1.0),
                    loss_metrics=[[{}], [{}], [{}]],
                    num_flops=0,
                )
            )
        )
        trainer.engine = engine
        trainer.config = Trainer.Config(model=MagicMock())
        trainer.dp_rank = 0
        microbatches = [_rl_microbatch(token) for token in range(3)]
        _leader(trainer, [])
        packed = []
        first_two_packed = threading.Event()

        def pack_planned(specs, samples):
            packed.append(threading.current_thread())
            if len(packed) == 2:
                first_two_packed.set()
            return microbatches[len(packed) - 1]

        trainer._microbatch_packer = SimpleNamespace(pack_planned=pack_planned)
        await trainer.receive_replica_column(_column(3))

        await trainer.start_reading(num_microbatch_groups=3)

        assert not trainer._replica_columns
        # The groups made ahead of the first one training waits on, and no more.
        assert first_two_packed.wait(timeout=10)
        assert len(packed) == 2
        await Trainer.forward_backward(
            trainer,
            num_microbatch_groups=3,
            global_loss_token_counts=torch.tensor([1]),
            global_routing_token_counts=torch.tensor([1]),
        )
        assert trained == [[microbatch] for microbatch in microbatches]
        assert all(thread is not threading.main_thread() for thread in packed)
        assert trainer._started_reader is None

    asyncio.run(run())


def test_each_step_trains_on_the_oldest_column() -> None:
    """With step N+1's column already received during step N (read-ahead), `forward_backward` N
    trains on column N and `start_reading` takes column N+1. Reversed, step N would pack N+1's
    specs and divide by plan N's token counts, and nothing else would catch it."""

    async def run() -> None:
        trainer = object.__new__(Trainer)
        engine = SimpleNamespace(num_completed_steps=0, pp_has_last_stage=False)
        trained: list[list[int]] = []

        def forward_backward(*, microbatch_groups, **kwargs) -> ForwardBackwardResult:
            trained.append([int(group[0].input) for group in microbatch_groups])
            return ForwardBackwardResult(
                loss=torch.tensor(-1.0), loss_metrics=[[{}], [{}]], num_flops=0
            )

        engine.forward_backward = MagicMock(side_effect=forward_backward)
        trainer.engine = engine
        trainer.config = Trainer.Config(model=MagicMock())
        trainer.dp_rank = 0
        _leader(trainer, [])
        # Each microbatch carries its first sample, i.e. its spec's rollout id.
        trainer._microbatch_packer = SimpleNamespace(
            pack_planned=lambda specs, samples: _rl_microbatch(samples[0])
        )

        async def forward_backward_step() -> None:
            await Trainer.forward_backward(
                trainer,
                num_microbatch_groups=2,
                global_loss_token_counts=torch.tensor([1]),
                global_routing_token_counts=torch.tensor([1]),
            )

        await trainer.receive_replica_column(fake_column([[[0]], [[1]]]))
        await trainer.receive_replica_column(fake_column([[[10]], [[11]]]))
        await forward_backward_step()
        await trainer.start_reading(num_microbatch_groups=2)
        assert not trainer._replica_columns
        await forward_backward_step()

        assert trained == [[0, 1], [10, 11]]

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
            parallelism_context=SimpleNamespace(
                non_data_parallel_size=1,
                dp_enabled=False,
                dp_cp_enabled=False,
                dp_replicate=1,
                dp_shard=1,
                get_optional_mesh=lambda _: None,
            ),
            num_completed_steps=4,
            ntokens_seen=12,
            has_quantization=False,
            device=torch.device("cpu"),
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
        trainer._resume_state = MirroredState()
        trainer.gpu_peak_flops = 1000
        trainer._step_compute_start = 0.0
        trainer._step_num_tokens_per_dp_rank = 10
        trainer._num_flops = 2000
        trainer._push_model_state_dict_stream = MagicMock()
        compute_stream = MagicMock()

        with (
            patch(
                "torchtitan.rl.trainer.trainer.torch.accelerator.current_stream",
                return_value=compute_stream,
            ),
            patch("torchtitan.rl.trainer.trainer.time.perf_counter", return_value=2.0),
            patch(
                "torchtitan.rl.trainer.trainer.collect_aux_loss_metrics",
                return_value={"aux_loss/mean": 0.5},
            ),
            patch(
                "torchtitan.rl.trainer.trainer.compute_training_performance_metrics",
                return_value={
                    "tokens_per_second": 10.0,
                    "tflops": 2.0,
                    "mfu_percent": 50.0,
                },
            ) as compute_performance,
        ):
            result = await Trainer.optim_step(
                trainer,
                resume_state={"next_index": 3},
            )

        assert result.policy_version == 5
        trainer._push_model_state_dict_stream.wait_stream.assert_called_once_with(
            compute_stream
        )
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
        engine.optim_step.assert_called_once_with()
        assert trainer._resume_state.value == {"next_index": 3}
        engine.save_checkpoint.assert_called_once_with(last_step=False)
        engine.step_profiler.assert_called_once_with()
        compute_performance.assert_called_once_with(
            num_tokens=10,
            elapsed_time=2.0,
            non_data_parallel_size=1,
            num_flops=2000.0,
            gpu_peak_flops=1000,
            has_quantization=False,
        )
        device_memory_monitor.get_peak_stats.assert_called_once_with()
        device_memory_monitor.reset_peak_stats.assert_called_once_with()

    asyncio.run(run())
