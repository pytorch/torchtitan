# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
import torch

from torchtitan.distributed.cudagraph import wrap_with_cuda_graph
from torchtitan.distributed.sdc_replay import SDCReplay, SDCReplayMismatch
from torchtitan.trainer import (
    ForwardBackwardStep,
    ForwardBackwardStepContext,
    ForwardBackwardStepFn,
    Trainer,
)


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


def test_forward_backward_step_installs_cuda_graph_body():
    body_fn = MagicMock()
    wrapped_body_fn = MagicMock(return_value=torch.tensor(1.0))
    postprocess_fn = MagicMock(
        return_value=(torch.ones(1), torch.ones(1), {"position": torch.ones(1)})
    )
    trainer = cast(
        Trainer,
        SimpleNamespace(),
    )

    trace_decorator = MagicMock(side_effect=lambda step_fn: step_fn)
    with (
        patch(
            "torchtitan.trainer.wrap_with_cuda_graph",
            return_value=wrapped_body_fn,
        ) as wrap,
        patch(
            "torchtitan.trainer.sl.log_trace_span",
            return_value=trace_decorator,
        ) as trace_span,
    ):
        step_fn = ForwardBackwardStep(
            use_cudagraph=True,
            use_sdc=False,
            pp_enabled=False,
            postprocess_fn=postprocess_fn,
            pipeline_step_fn=MagicMock(),
            body_fn=body_fn,
            replay_state=trainer,
        )

    wrap.assert_called_once_with(body_fn)
    trace_span.assert_called_once_with("fwd_bwd")
    trace_decorator.assert_called_once()
    torch.testing.assert_close(
        step_fn(
            ForwardBackwardStepContext(
                input_dict={"input": torch.ones(1)},
                labels=torch.ones(1),
                global_valid_tokens=torch.ones(1),
            )
        ),
        torch.tensor(1.0),
    )
    wrapped_body_fn.assert_called_once()


def test_make_forward_backward_step_skips_disabled_sdc():
    trainer = cast(
        Trainer,
        SimpleNamespace(
            config=SimpleNamespace(
                should_use_cudagraph=lambda: False,
                sdc_replay=SDCReplay.Config(enabled=False),
            ),
            parallel_dims=SimpleNamespace(pp_enabled=False),
            post_dataloading_process=MagicMock(
                return_value=(torch.ones(1), torch.ones(1), {})
            ),
            pp_forward_backward_step=MagicMock(),
            _forward_backward_body=MagicMock(return_value=torch.tensor(1.0)),
            _get_sdc_replay=MagicMock(),
        ),
    )

    forward_backward_step = Trainer.make_forward_backward_step(trainer)
    forward_backward_step(
        ForwardBackwardStepContext(
            input_dict={"input": torch.ones(1)},
            labels=torch.ones(1),
            global_valid_tokens=torch.ones(1),
        )
    )

    trainer._get_sdc_replay.assert_not_called()


def test_trainer_accumulates_reused_cuda_graph_losses():
    graph_loss = torch.tensor(0.0)
    loss_values = iter((1.0, 2.0, 3.0, 4.0, 5.0, 6.0))

    def forward_backward_step(context):
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
                should_use_cudagraph=lambda: True,
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
            step=1,
            ntokens_seen=3,
            sdc_attempt_step=0,
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
    assert trainer.sdc_attempt_step == 2


def test_forward_backward_step_replays_only_first_gradient_accumulation_unit():
    replay = SimpleNamespace(
        should_run=MagicMock(return_value=True),
        run=MagicMock(side_effect=lambda execute, **kwargs: execute()),
    )
    implementation = MagicMock(return_value=torch.tensor(1.0))
    trainer = cast(
        Trainer,
        SimpleNamespace(
            _sdc_replay_unit_index=0,
            sdc_attempt_step=0,
            step=1,
            _get_sdc_replay=lambda: replay,
        ),
    )

    context = ForwardBackwardStepContext(
        input_dict={"input": torch.ones(1)},
        labels=torch.ones(1),
        global_valid_tokens=torch.ones(1),
    )

    def implementation_wrapper(
        _next_step_fn, context: ForwardBackwardStepContext
    ) -> torch.Tensor:
        return implementation(context)

    forward_backward_step = ForwardBackwardStep(
        use_cudagraph=False,
        use_sdc=True,
        pp_enabled=False,
        postprocess_fn=MagicMock(),
        pipeline_step_fn=MagicMock(),
        body_fn=MagicMock(),
        replay_state=trainer,
        step_wrappers=(implementation_wrapper,),
    )
    forward_backward_step(context)
    forward_backward_step(context)

    replay.run.assert_called_once()
    assert implementation.call_count == 2


def test_forward_backward_step_applies_step_wrapper():
    trainer = cast(Trainer, SimpleNamespace())

    def wrapper(
        _next_step_fn: ForwardBackwardStepFn, context: ForwardBackwardStepContext
    ) -> torch.Tensor:
        assert context.labels is labels
        return torch.tensor(2.0)

    labels = torch.ones(1)
    forward_backward_step = ForwardBackwardStep(
        use_cudagraph=False,
        use_sdc=False,
        pp_enabled=False,
        postprocess_fn=MagicMock(),
        pipeline_step_fn=MagicMock(),
        body_fn=MagicMock(),
        replay_state=trainer,
        step_wrappers=(wrapper,),
    )

    torch.testing.assert_close(
        forward_backward_step(
            ForwardBackwardStepContext(
                input_dict={"input": torch.ones(1)},
                labels=labels,
            )
        ),
        torch.tensor(2.0),
    )


def test_replay_failure_happens_before_optimizer():
    mismatch = SDCReplayMismatch(
        step=1,
        attempt=1,
        replay=1,
        rank=0,
        signature_mismatch="loss",
    )
    optimizers = MagicMock()
    trainer = cast(
        Trainer,
        SimpleNamespace(
            config=SimpleNamespace(
                training=SimpleNamespace(disable_cuda_graphs=True, max_norm=1.0),
                should_use_cudagraph=lambda: False,
            ),
            optimizers=optimizers,
            lr_schedulers=SimpleNamespace(get_metrics=lambda: {}, step=MagicMock()),
            parallel_dims=SimpleNamespace(
                dp_enabled=False,
                pp_enabled=False,
                dp_cp_enabled=False,
                ep_enabled=False,
                get_optional_mesh=lambda name: None,
            ),
            gradient_accumulation_steps=1,
            num_pipeline_parallel_microbatches=1,
            device=torch.device("cpu"),
            forward_backward_step=MagicMock(side_effect=mismatch),
            model_parts=[],
            checkpointer=SimpleNamespace(maybe_wait_for_staging=MagicMock()),
            metrics_processor=SimpleNamespace(should_log=MagicMock(return_value=False)),
            step=1,
            ntokens_seen=0,
            sdc_attempt_step=0,
        ),
    )

    with pytest.raises(SDCReplayMismatch):
        Trainer.train_step(
            trainer,
            iter([({"input": torch.ones(1)}, torch.ones(1, dtype=torch.long))]),
        )

    optimizers.step.assert_not_called()
    assert trainer.sdc_attempt_step == 0


def test_loading_checkpoint_rearms_attempt_local_replay():
    trainer = cast(Trainer, SimpleNamespace(sdc_attempt_step=8))

    Trainer.load_state_dict(trainer, {"step": 12, "ntokens_seen": 34})

    assert trainer.step == 12
    assert trainer.ntokens_seen == 34
    assert trainer.sdc_attempt_step == 0


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
