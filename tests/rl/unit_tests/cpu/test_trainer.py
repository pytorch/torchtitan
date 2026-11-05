# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

import torch
from torchstore import RankRole
from torchtitan.config import PrecisionConfig, TrainingConfig
from torchtitan.models.base.module import Module
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.rl.controller.metrics import combine_microbatch_metrics
from torchtitan.rl.trainer.batcher import TrainingMicrobatch
from torchtitan.rl.trainer.rdma_microbatch import (
    RDMAMicrobatch,
    RDMAMicrobatchPrefetcher,
)
from torchtitan.rl.trainer.trainer import MirroredState, Trainer
from torchtitan.training_engine import ForwardBackwardResult, TrainingEngine


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
            )

    validate.assert_called_once_with(
        model_config,
        parallelism=config.parallelism,
        training=config.training,
        debug=config.debug,
        activation_remat=config.activation_remat,
        max_num_documents=None,
    )


def test_forward_backward_preprocesses_every_group() -> None:
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
        _step_started=False,
        _forward_backward_captures_whole_step=False,
        sdc_replayer=None,
        flops_estimator=lambda inputs: inputs["input"].numel(),
    )
    microbatch_groups = [[_rl_microbatch(token)] for token in range(3)]

    result = TrainingEngine.forward_backward(
        engine,
        microbatch_groups=microbatch_groups,
        aux_loss_denominators=torch.tensor([1.0]),
    )

    torch.testing.assert_close(result.loss, torch.tensor(1.0))
    engine.garbage_collector.run.assert_called_once_with(1)
    engine.optim.zero_grad.assert_called_once_with(set_to_none=True)
    engine._preprocess_microbatch_groups.assert_called_once_with(microbatch_groups)


def test_close_stops_training_engine() -> None:
    async def run() -> None:
        trainer = object.__new__(Trainer)
        engine = SimpleNamespace(
            close=MagicMock(),
        )
        trainer.engine = engine
        trainer._resume_state = MirroredState()

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


def test_get_resume_state_returns_restored_state() -> None:
    async def run() -> None:
        trainer = object.__new__(Trainer)
        trainer._resume_state = MirroredState()
        trainer._resume_state.loaded = {"next_index": 11}

        assert await trainer.get_resume_state() == {"next_index": 11}

    asyncio.run(run())


@pytest.mark.parametrize("rdma", [False, True])
def test_forward_backward_accumulates_microbatch_metrics(rdma: bool) -> None:
    async def run() -> None:
        trainer = object.__new__(Trainer)
        engine = SimpleNamespace(
            num_completed_steps=4,
            ntokens_seen=10,
            sdc_replayer=None,
            model_config=SimpleNamespace(mtp_layers=None),
            device=torch.device("cpu"),
            pp_has_last_stage=True,
            uses_forward_backward_accelerator_graphs=False,
        )
        trained_groups = []

        def train(*, microbatch_groups, aux_loss_denominators):
            trained_groups.append((type(microbatch_groups), list(microbatch_groups)))
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

        engine.forward_backward = MagicMock(side_effect=train)
        trainer.engine = engine
        trainer.config = Trainer.Config(model=MagicMock())
        trainer.dp_rank = 1
        trainer._num_rdma_batches_read = 0
        trainer._forward_backward_lock = asyncio.Lock()
        trainer._step_forward_backward_s = 0.0
        trainer._step_num_tokens_per_dp_rank = 0
        trainer._step_num_flops = 0
        trainer._step_read_microbatches_s = 0.0
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
            loss_token_count=torch.tensor(1),
            routing_token_counts=torch.tensor([1]),
            generator_logprobs=torch.tensor([0.0]),
            temperature=torch.tensor([1.0]),
            loss_mask=torch.tensor([True]),
            advantages=torch.tensor([1.0]),
        )

        # This rank's column (dp_rank 1); with `rdma_min_tensor_bytes` the trainer reads it.
        microbatches = (
            [
                MagicMock(
                    spec=RDMAMicrobatch,
                    read=AsyncMock(return_value=batch),
                    batch_index=0,
                    dp_rank=1,
                    microbatch_index=microbatch_index,
                )
                for microbatch_index in range(2)
            ]
            if rdma
            else [batch, batch]
        )

        with patch("torchtitan.rl.trainer.trainer.torch.accelerator.current_stream"):
            result = await Trainer.forward_backward(
                trainer,
                [[microbatch] for microbatch in microbatches],
                torch.tensor([1.0]),
            )

        # RDMAMicrobatches are read through the prefetcher; by value, the column is passed as is.
        [(groups_type, groups)] = trained_groups
        assert (groups_type is RDMAMicrobatchPrefetcher) == rdma
        assert groups == [[batch], [batch]]
        assert trainer._step_num_tokens_per_dp_rank == (
            2 * trainer.config.training.num_tokens_per_microbatch_per_dp_rank
        )
        if rdma:
            assert trainer._num_rdma_batches_read == 1
            # The read time is reported only when there was a read.
            assert result.pop("timing/trainer/read_microbatches/max") >= 0
        assert trainer._reduce_forward_backward_metrics.call_count == (3 if rdma else 2)
        assert result == {"loss/mean": 3.0, "loss/max": 4.0}

    asyncio.run(run())


def test_streamed_forward_backward_calls_add_up_the_step_totals() -> None:
    async def run() -> None:
        trainer = object.__new__(Trainer)
        trainer.engine = SimpleNamespace(
            num_completed_steps=0,
            device=torch.device("cpu"),
            pp_has_last_stage=True,
            forward_backward=MagicMock(
                return_value=ForwardBackwardResult(
                    loss=torch.tensor(0.5),
                    loss_metrics=[[{"loss/mean": torch.tensor(1.0)}]],
                    num_flops=100,
                )
            ),
        )
        trainer.config = Trainer.Config(model=MagicMock())
        trainer.dp_rank = 0
        trainer._num_rdma_batches_read = 0
        trainer._forward_backward_lock = asyncio.Lock()
        trainer._step_forward_backward_s = 0.0
        trainer._step_num_tokens_per_dp_rank = 0
        trainer._step_num_flops = 0
        trainer._step_read_microbatches_s = 0.0
        trainer._reduce_forward_backward_metrics = MagicMock(
            side_effect=lambda *, sum_reduced_metrics, max_reduced_metrics: {
                key: float(value.item())
                for key, value in {
                    **sum_reduced_metrics,
                    **max_reduced_metrics,
                }.items()
            }
        )

        call_metrics = []
        # Each call's reads take 0.25 s.
        with patch(
            "torchtitan.rl.trainer.trainer.RDMAMicrobatchPrefetcher",
            return_value=SimpleNamespace(read_s=0.25, close=lambda: None),
        ), patch("torchtitan.rl.trainer.trainer.torch.accelerator.current_stream"):
            for batch_index in range(2):
                microbatch = MagicMock(
                    spec=RDMAMicrobatch,
                    batch_index=batch_index,
                    dp_rank=0,
                    microbatch_index=0,
                )
                call_metrics.append(
                    await Trainer.forward_backward(
                        trainer, [[microbatch]], torch.tensor([1.0])
                    )
                )

        # optim_step reports the step's tokens, FLOPs and busy time over both calls.
        assert trainer._step_num_tokens_per_dp_rank == (
            2 * trainer.config.training.num_tokens_per_microbatch_per_dp_rank
        )
        assert trainer._step_num_flops == 200
        assert trainer._step_forward_backward_s > 0.0
        assert trainer.engine.forward_backward.call_count == 2
        # The controller combines the calls' metrics; the step reports both calls' reads.
        read_s = combine_microbatch_metrics(call_metrics)[
            "timing/trainer/read_microbatches/max"
        ]
        assert read_s == 0.5

    asyncio.run(run())


def test_queued_forward_backward_reads_while_the_running_call_waits_for_the_gpu() -> None:
    # Call 0 waits for a fake GPU that finishes only when the test says so. Call 1 arrives
    # meanwhile, as the controller sends it, and must start its read before call 0 returns.
    gpu_wait_started, gpu_done = threading.Event(), threading.Event()
    read_started = [threading.Event(), threading.Event()]
    trained_groups = []

    def wait_for_gpu() -> None:
        gpu_wait_started.set()
        assert gpu_done.wait(timeout=5)

    def rdma_microbatch(batch_index: int) -> RDMAMicrobatch:
        async def read() -> int:
            read_started[batch_index].set()
            return batch_index

        return MagicMock(
            spec=RDMAMicrobatch,
            read=read,
            batch_index=batch_index,
            dp_rank=0,
            microbatch_index=0,
        )

    def train(*, microbatch_groups, aux_loss_denominators):
        trained_groups.append(list(microbatch_groups))
        return ForwardBackwardResult(
            loss=torch.tensor(0.5),
            loss_metrics=[[{"loss/mean": torch.tensor(1.0)}]],
            num_flops=0,
        )

    async def run() -> None:
        trainer = object.__new__(Trainer)
        trainer.engine = SimpleNamespace(
            num_completed_steps=0,
            device=torch.device("cpu"),
            pp_has_last_stage=True,
            forward_backward=MagicMock(side_effect=train),
        )
        trainer.config = Trainer.Config(model=MagicMock())
        trainer.dp_rank = 0
        trainer._num_rdma_batches_read = 0
        trainer._forward_backward_lock = asyncio.Lock()
        trainer._step_forward_backward_s = 0.0
        trainer._step_num_tokens_per_dp_rank = 0
        trainer._step_num_flops = 0
        trainer._step_read_microbatches_s = 0.0
        trainer._reduce_forward_backward_metrics = MagicMock(return_value={})

        with patch(
            "torchtitan.rl.trainer.trainer.torch.accelerator.current_stream",
            return_value=SimpleNamespace(synchronize=wait_for_gpu),
        ):
            call_0 = asyncio.create_task(
                Trainer.forward_backward(
                    trainer, [[rdma_microbatch(0)]], torch.tensor([1.0])
                )
            )
            assert await asyncio.to_thread(gpu_wait_started.wait, 5)
            call_1 = asyncio.create_task(
                Trainer.forward_backward(
                    trainer, [[rdma_microbatch(1)]], torch.tensor([1.0])
                )
            )
            assert await asyncio.to_thread(read_started[1].wait, 5)
            assert not call_0.done()
            gpu_done.set()
            await asyncio.gather(call_0, call_1)

        # Call 1's engine work runs after call 0's.
        assert trained_groups == [[[0]], [[1]]]

    asyncio.run(run())


@pytest.mark.parametrize(
    ("graphs_enabled", "step_started", "expected_engine_calls"),
    [
        # With graphs, the step's first group is its own engine call.
        (True, False, [[0], [1, 2]]),
        (True, True, [[0, 1, 2]]),
        (False, False, [[0, 1, 2]]),
    ],
)
@pytest.mark.parametrize("rdma", [False, True])
def test_forward_backward_sends_the_step_first_group_alone_with_graphs(
    graphs_enabled: bool,
    step_started: bool,
    expected_engine_calls: list[list[int]],
    rdma: bool,
) -> None:
    events = []

    class RecordingPrefetcher:
        def __init__(self, microbatches):
            events.append("prefetcher")
            self.microbatches = microbatches
            self.read_s = 0.25

        def close(self) -> None:
            pass

    def forward_backward(*, microbatch_groups, aux_loss_denominators):
        microbatches = (
            microbatch_groups.microbatches
            if rdma
            else [microbatch for [microbatch] in microbatch_groups]
        )
        # Both engine calls of a split call take the trainer call's denominators.
        assert aux_loss_denominators is denominators
        events.append([microbatch.microbatch_index for microbatch in microbatches])
        return ForwardBackwardResult(
            loss=torch.tensor(0.5),
            loss_metrics=[[{"loss/mean": torch.tensor(1.0)}] for _ in microbatches],
            num_flops=100,
        )

    trainer = object.__new__(Trainer)
    trainer.engine = SimpleNamespace(
        num_completed_steps=0,
        device=torch.device("cpu"),
        pp_has_last_stage=True,
        uses_forward_backward_accelerator_graphs=graphs_enabled,
        step_started=step_started,
        forward_backward=forward_backward,
    )
    trainer.config = Trainer.Config(model=MagicMock())
    trainer.dp_rank = 0
    trainer._num_rdma_batches_read = 0
    trainer._forward_backward_lock = asyncio.Lock()
    trainer._step_forward_backward_s = 0.0
    trainer._step_num_tokens_per_dp_rank = 0
    trainer._step_num_flops = 0
    trainer._step_read_microbatches_s = 0.0
    trainer._reduce_forward_backward_metrics = MagicMock(
        side_effect=lambda *, sum_reduced_metrics, max_reduced_metrics: {
            key: float(value.item())
            for key, value in {**sum_reduced_metrics, **max_reduced_metrics}.items()
        }
    )
    microbatches = [
        MagicMock(
            spec=RDMAMicrobatch if rdma else TrainingMicrobatch,
            batch_index=0,
            dp_rank=0,
            microbatch_index=index,
        )
        for index in range(3)
    ]

    denominators = torch.tensor([2.0])
    with patch(
        "torchtitan.rl.trainer.trainer.RDMAMicrobatchPrefetcher", RecordingPrefetcher
    ), patch("torchtitan.rl.trainer.trainer.torch.accelerator.current_stream"):
        metrics = asyncio.run(
            Trainer.forward_backward(
                trainer, [[microbatch] for microbatch in microbatches], denominators
            )
        )

    # Every engine call's prefetcher is made on arrival, so its reads start before any engine call.
    assert (
        events
        == (["prefetcher"] * len(expected_engine_calls) if rdma else [])
        + expected_engine_calls
    )
    # Both engine calls' loss metrics and FLOPs come back.
    assert metrics["loss/mean"] == 3.0
    assert trainer._step_num_flops == 100 * len(expected_engine_calls)
    if rdma:
        assert trainer._step_read_microbatches_s == 0.25 * len(expected_engine_calls)


@pytest.mark.parametrize(
    ("batch_index", "dp_rank", "microbatch_index"),
    [(1, 1, 0), (0, 0, 0), (0, 1, 1)],
    ids=["another_batch", "another_rank", "out_of_order"],
)
def test_forward_backward_rejects_a_misplaced_rdma_microbatch(
    batch_index: int, dp_rank: int, microbatch_index: int
) -> None:
    trainer = object.__new__(Trainer)
    trainer.engine = SimpleNamespace(num_completed_steps=0)
    trainer.config = Trainer.Config(model=MagicMock())
    trainer.dp_rank = 1
    trainer._num_rdma_batches_read = 0
    trainer._step_num_tokens_per_dp_rank = 0
    wrong = MagicMock(
        spec=RDMAMicrobatch,
        batch_index=batch_index,
        dp_rank=dp_rank,
        microbatch_index=microbatch_index,
    )

    with pytest.raises(ValueError, match=r"expected \(0, 1, 0\)"):
        asyncio.run(Trainer.forward_backward(trainer, [[wrong]], torch.tensor([1.0])))
    wrong.read.assert_not_called()


def _rl_microbatch(token: int) -> TrainingMicrobatch:
    return TrainingMicrobatch(
        input=torch.tensor([token]),
        labels=torch.tensor([token + 1]),
        positions=torch.tensor([0]),
        padding_mask=torch.tensor([False]),
        loss_token_count=torch.tensor(1),
        routing_token_counts=torch.tensor([1]),
        generator_logprobs=torch.tensor([0.0]),
        loss_mask=torch.tensor([True]),
        advantages=torch.tensor([1.0]),
        temperature=torch.tensor([0.7]),
    )


def test_forward_backward_without_last_pipeline_stage_returns_no_metrics() -> None:
    async def run() -> None:
        trainer = object.__new__(Trainer)
        engine = SimpleNamespace(
            num_completed_steps=0,
            pp_has_last_stage=False,
            uses_forward_backward_accelerator_graphs=False,
        )
        engine.forward_backward = MagicMock(
            return_value=ForwardBackwardResult(
                loss=torch.tensor(-1.0), loss_metrics=[[{}, {}], [{}, {}]], num_flops=0
            )
        )
        trainer.engine = engine
        trainer.config = Trainer.Config(model=MagicMock())
        engine.device = torch.device("cpu")
        trainer.dp_rank = 1
        trainer._forward_backward_lock = asyncio.Lock()
        trainer._step_forward_backward_s = 0.0
        trainer._step_num_tokens_per_dp_rank = 0
        trainer._step_read_microbatches_s = 0.0
        trainer._step_num_flops = 0
        trainer._reduce_forward_backward_metrics = MagicMock()
        # This rank's column: two pipeline steps of two microbatches each.
        microbatch_groups = [
            [_rl_microbatch(10 * group + index) for index in range(2)]
            for group in range(2)
        ]

        aux_loss_denominators = torch.tensor([2.0])

        with patch(
            "torchtitan.rl.trainer.trainer.torch.accelerator.current_stream"
        ) as current_stream:
            result = await Trainer.forward_backward(
                trainer, microbatch_groups, aux_loss_denominators
            )

        # Its busy time waits for the GPU, like the ranks that reduce the loss metrics.
        current_stream.assert_called_once_with(engine.device)
        current_stream.return_value.synchronize.assert_called_once_with()
        engine.forward_backward.assert_called_once_with(
            microbatch_groups=microbatch_groups,
            aux_loss_denominators=aux_loss_denominators,
        )
        assert trainer._step_num_tokens_per_dp_rank == (
            4 * trainer.config.training.num_tokens_per_microbatch_per_dp_rank
        )
        trainer._reduce_forward_backward_metrics.assert_not_called()
        assert result is None

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
            global_loss_token_count=torch.tensor(7),
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
        # forward_backward's busy time; the patched clock makes the update itself take 0 s.
        trainer._step_forward_backward_s = 2.0
        trainer._step_num_tokens_per_dp_rank = 10
        trainer._step_read_microbatches_s = 0.5
        trainer._step_num_flops = 2000
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
            ) as collect_aux_loss_metrics,
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
        collect_aux_loss_metrics.assert_called_once_with(
            engine.parallelism_context, num_loss_tokens=7
        )
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
        # The next step starts counting from zero.
        assert trainer._step_forward_backward_s == 0.0
        assert trainer._step_num_tokens_per_dp_rank == 0
        assert trainer._step_num_flops == 0
        assert trainer._step_read_microbatches_s == 0.0
        device_memory_monitor.get_peak_stats.assert_called_once_with()
        device_memory_monitor.reset_peak_stats.assert_called_once_with()

    asyncio.run(run())
