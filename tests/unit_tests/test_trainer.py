# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
import torch

from torchtitan.distributed.cudagraph import wrap_with_cuda_graph
from torchtitan.trainer import Trainer


def test_pp_forward_backward_step_returns_sentinel_without_last_stage():
    trainer = cast(
        Trainer,
        SimpleNamespace(
            pp_has_first_stage=False,
            pp_has_last_stage=False,
            pp_schedule=SimpleNamespace(step=lambda **kwargs: None),
            train_context=nullcontext,
            post_dataloading_process=lambda input_dict, labels: (
                input_dict["input"],
                labels,
                {},
            ),
            device=torch.device("cpu"),
        ),
    )

    loss = Trainer.pp_forward_backward_step(
        trainer,
        input_dict_mbs=[{"input": torch.ones(1)}],
        label_mbs=[torch.ones(1)],
        global_valid_tokens=torch.tensor(1),
    )

    torch.testing.assert_close(loss, torch.tensor([-1.0]))


def test_cuda_graph_wrapper_returns_graph_owned_output():
    class PassthroughCUDAGraphWrapper:
        def __init__(self, fn, example_inputs):
            self.fn = fn

        def __call__(self, *args):
            return self.fn(*args)

    graph_loss = torch.tensor(0.0)
    fwd_bwd = MagicMock(return_value=graph_loss)

    with (
        patch("torchtitan.distributed.cudagraph.utils.device_type", "cuda"),
        patch("torch.cuda.is_available", return_value=True),
        patch.object(torch.version, "hip", None),
        patch(
            "torchtitan.distributed.cudagraph.CUDAGraphWrapper",
            PassthroughCUDAGraphWrapper,
        ),
    ):
        runner = wrap_with_cuda_graph(fwd_bwd)
        for value in (1.0, 2.0, 3.0):
            graph_loss.fill_(value)
            loss = runner(
                torch.ones(1),
                torch.ones(1),
                torch.tensor(1),
                {"position": torch.ones(1)},
            )
            # Sanity check that the wrapper returns the same graph-owned object.
            assert loss is graph_loss

    assert fwd_bwd.call_count == 3
    _, _, global_valid_tokens, extra_kwargs = fwd_bwd.call_args.args
    torch.testing.assert_close(global_valid_tokens, torch.tensor(1))
    assert global_valid_tokens.dtype == torch.int64
    torch.testing.assert_close(extra_kwargs["position"], torch.ones(1))


def test_trainer_accumulates_reused_cuda_graph_losses():
    graph_loss = torch.tensor(0.0)
    loss_values = iter((1.0, 2.0, 3.0, 4.0, 5.0, 6.0))

    def forward_backward_step(**kwargs):
        graph_loss.fill_(next(loss_values))
        return graph_loss

    metrics_processor = SimpleNamespace(
        should_log=MagicMock(return_value=True),
        log=MagicMock(),
    )
    trainer = cast(
        Trainer,
        SimpleNamespace(
            config=SimpleNamespace(
                training=SimpleNamespace(
                    disable_cuda_graphs=False,
                    max_norm=1.0,
                ),
            ),
            optimizers=MagicMock(),
            lr_schedulers=SimpleNamespace(
                get_metrics=MagicMock(return_value={}),
                step=MagicMock(),
            ),
            parallel_dims=SimpleNamespace(
                dp_enabled=False,
                pp_enabled=False,
                dp_cp_enabled=False,
                ep_enabled=False,
                get_optional_mesh=lambda name: None,
            ),
            gradient_accumulation_steps=3,
            num_pipeline_parallel_microbatches=1,
            device=torch.device("cpu"),
            forward_backward_step=forward_backward_step,
            model_parts=[],
            checkpointer=SimpleNamespace(maybe_wait_for_staging=MagicMock()),
            metrics_processor=metrics_processor,
            tensor_logging=None,
            step=1,
            ntokens_seen=3,
        ),
    )
    data_iterator = iter(
        [({"input": torch.ones(1)}, torch.ones(1, dtype=torch.long))] * 3
    )

    with patch(
        "torchtitan.trainer.dist_utils.clip_grad_norm_",
        return_value=torch.tensor(4.0),
    ):
        Trainer.train_step(trainer, data_iterator)

    metrics_processor.log.assert_called_once_with(
        1,
        6.0,
        6.0,
        4.0,
        extra_metrics={"n_tokens_seen": 3},
    )

    metrics_processor.should_log.return_value = False
    metrics_processor.log.reset_mock()
    with patch(
        "torchtitan.trainer.dist_utils.clip_grad_norm_",
        return_value=torch.tensor(4.0),
    ):
        Trainer.train_step(
            trainer,
            data_iterator=iter(
                [({"input": torch.ones(1)}, torch.ones(1, dtype=torch.long))] * 3
            ),
        )

    metrics_processor.log.assert_not_called()


def test_train_scopes_complete_steps_with_tensor_logging_cadence() -> None:
    enabled_values = []

    @contextmanager
    def record_enabled(value: bool):
        enabled_values.append(value)
        yield

    profiler = MagicMock()
    profiler_context = MagicMock()
    profiler_context.__enter__.return_value = profiler
    config = SimpleNamespace(
        dump_folder="/tmp/tensor-logging-test",
        checkpoint=SimpleNamespace(load_step=-1),
        profiler=SimpleNamespace(build=MagicMock(return_value=profiler_context)),
        metrics=SimpleNamespace(tensor_logging=SimpleNamespace(freq=2)),
        training=SimpleNamespace(steps=2),
        validator=SimpleNamespace(enable=False),
        comm=SimpleNamespace(train_timeout_seconds=10),
    )
    trainer = cast(
        Trainer,
        SimpleNamespace(
            config=config,
            checkpointer=SimpleNamespace(
                load=MagicMock(),
                save=MagicMock(),
            ),
            step=0,
            tensor_logging=object(),
            metrics_processor=SimpleNamespace(
                should_log=MagicMock(return_value=True),
            ),
            dataloader=object(),
            batch_generator=MagicMock(return_value=iter(())),
            should_continue_training=MagicMock(side_effect=[True, True, False]),
            gc_handler=SimpleNamespace(run=MagicMock()),
            train_step=MagicMock(),
            parallel_dims=SimpleNamespace(),
        ),
    )

    with (
        patch("torchtitan.trainer.tensor_logging.set_enabled", record_enabled),
        patch("torchtitan.trainer.sl.log_trace_instant"),
        patch("torchtitan.trainer.sl.set_step"),
        patch("torchtitan.trainer.sl.log_trace_span", return_value=nullcontext()),
        patch("torchtitan.trainer.dist_utils.set_pg_timeouts"),
        patch("torch.distributed.get_rank", return_value=0),
        patch("torchtitan.trainer.time.sleep"),
    ):
        Trainer.train(trainer)

    assert enabled_values == [False, True]
    assert trainer.train_step.call_count == 2


@pytest.mark.parametrize(
    ("schedule", "schedule_csv"),
    [
        ("ZBVZeroBubble", ""),
        ("DualPipeV", ""),
        ("UnknownSchedule", ""),
        ("1F1B", "tests/assets/custom_schedule.csv"),
    ],
)
def test_tensor_logging_rejects_unsupported_pipeline_schedule_before_model_setup(
    schedule: str,
    schedule_csv: str,
) -> None:
    config = SimpleNamespace(
        model_spec=object(),
        metrics=SimpleNamespace(tensor_logging=SimpleNamespace(enabled=True)),
        parallelism=SimpleNamespace(
            pipeline_parallel_schedule=schedule,
            pipeline_parallel_schedule_csv=schedule_csv,
        ),
    )
    parallel_dims = SimpleNamespace(pp_enabled=True)

    with (
        patch.object(Trainer, "init_distributed", return_value=parallel_dims),
        patch("torchtitan.trainer.utils.get_local_device", return_value="cpu"),
        patch("torchtitan.trainer.utils.device_module.set_device"),
        pytest.raises(NotImplementedError, match="supports only the 1F1B"),
    ):
        Trainer(config)


@pytest.mark.parametrize(
    ("device_type", "cuda_available", "hip_version"),
    [
        ("cpu", False, None),
        ("cuda", False, None),
        ("cuda", True, "6.3"),
        ("xpu", False, None),
    ],
)
def test_cuda_graph_wrapper_is_noop_without_nvidia_cuda(
    device_type: str,
    cuda_available: bool,
    hip_version: str | None,
) -> None:
    fwd_bwd = MagicMock()

    with (
        patch("torchtitan.distributed.cudagraph.utils.device_type", device_type),
        patch("torch.cuda.is_available", return_value=cuda_available),
        patch.object(torch.version, "hip", hip_version),
        patch("torchtitan.distributed.cudagraph.logger.warning") as warning,
    ):
        runner = wrap_with_cuda_graph(fwd_bwd)

    assert runner is fwd_bwd
    warning.assert_called_once()
