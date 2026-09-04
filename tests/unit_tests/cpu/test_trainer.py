# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
import torch
from torchtitan.distributed.cudagraph import wrap_with_cuda_graph
from torchtitan.observability.sdc_replayer import SDCReplayMismatch
from torchtitan.trainer import Trainer


def _batch() -> tuple[dict[str, object], torch.Tensor]:
    """One dataloader batch.

    Built fresh per call because ``train_step`` pops ``num_valid_tokens`` out of
    the dict it is handed.
    """
    return {"input": torch.ones(1), "num_valid_tokens": 1}, torch.ones(
        1, dtype=torch.long
    )


def test_pp_forward_backward_step_returns_sentinel_without_last_stage():
    trainer = cast(
        Trainer,
        SimpleNamespace(
            pp_has_first_stage=False,
            pp_has_last_stage=False,
            pp_schedule=SimpleNamespace(step=lambda **kwargs: None),
            train_context=nullcontext,
            model_parts=[
                SimpleNamespace(
                    preprocess_inputs=lambda input_dict, **kw: (
                        input_dict["input"],
                        input_dict["labels"],
                        {},
                    )
                )
            ],
            parallel_dims=SimpleNamespace(pp_enabled=True),
            config=SimpleNamespace(parallelism="PARA"),
            ntokens_seen=0,
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


def test_pp_forward_backward_step_releases_consumed_loss_graphs() -> None:
    activation_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    loss_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    loss_containers: list[list[torch.Tensor]] = []
    gradients: list[torch.Tensor] = []

    def schedule_step(**kwargs) -> None:
        loss_containers.append(kwargs["losses"])
        for value in (1.0, 2.0):
            activation = torch.tensor(value, requires_grad=True)
            loss = activation.square().view(())
            loss.backward()
            assert activation.grad is not None
            gradients.append(activation.grad.detach().clone())
            activation_refs.append(weakref.ref(activation))
            loss_refs.append(weakref.ref(loss))
            kwargs["losses"].append(loss)

    trainer = cast(
        Trainer,
        SimpleNamespace(
            pp_has_first_stage=True,
            pp_has_last_stage=True,
            pp_schedule=SimpleNamespace(step=schedule_step),
            train_context=nullcontext,
            model_parts=[
                SimpleNamespace(
                    preprocess_inputs=lambda input_dict, **kw: (
                        input_dict["input"],
                        input_dict["labels"],
                        {},
                    )
                )
            ],
            parallel_dims=SimpleNamespace(pp_enabled=True),
            config=SimpleNamespace(parallelism="PARA"),
            ntokens_seen=0,
            device=torch.device("cpu"),
        ),
    )

    reporting_loss = Trainer.pp_forward_backward_step(
        trainer,
        input_dict_mbs=[{"input": torch.ones(1)}] * 2,
        label_mbs=[torch.ones(1)] * 2,
        global_valid_tokens=torch.tensor(2),
    )

    torch.testing.assert_close(reporting_loss, torch.tensor(5.0))
    torch.testing.assert_close(torch.stack(gradients), torch.tensor([2.0, 4.0]))
    assert not reporting_loss.requires_grad
    assert reporting_loss.grad_fn is None
    assert loss_containers == [[]]
    assert all(reference() is None for reference in loss_refs)
    assert all(reference() is None for reference in activation_refs)


def test_forward_backward_step_accumulates_tokens_and_forwards_triple():
    captured = {}

    class _FakeModel:
        def preprocess_inputs(self, input_dict, **kw):
            captured["preprocess_kwargs"] = kw
            return ("INPUTS", torch.ones(7), {"positions": 1})

    def fwd_bwd_fn(inputs, labels, global_valid_tokens, extra_kwargs):
        captured["fwd_bwd_args"] = (inputs, labels, extra_kwargs)
        return torch.tensor(0.0)

    fake = SimpleNamespace(
        model_parts=[_FakeModel()],
        parallel_dims=SimpleNamespace(pp_enabled=False),
        config=SimpleNamespace(parallelism="PARA"),
        ntokens_seen=100,
        fwd_bwd_fn=fwd_bwd_fn,
    )

    Trainer.forward_backward_step(
        fake,
        input_dict={"input": 0},
        labels=torch.zeros(1),
        global_valid_tokens=torch.tensor(1),
    )

    inputs, labels, extra = captured["fwd_bwd_args"]
    assert inputs == "INPUTS"
    assert extra == {"positions": 1}
    assert labels.numel() == 7
    assert fake.ntokens_seen == 107  # labels.numel() (7) folded in
    assert captured["preprocess_kwargs"] == {
        "parallel_dims": fake.parallel_dims,
        "parallelism": "PARA",
    }


@pytest.mark.parametrize(
    ("gradient_accumulation_steps", "sdc_config", "expected"),
    [
        (1, None, 2),
        (4, None, 8),
        (1, SimpleNamespace(num_steps=1, num_replays=1), 3),
        (4, SimpleNamespace(num_steps=2, num_replays=1), 10),
        (4, SimpleNamespace(num_steps=-1, num_replays=1), 10),
    ],
)
def test_cuda_graph_warmup_covers_two_optimizer_steps(
    gradient_accumulation_steps: int,
    sdc_config: SimpleNamespace | None,
    expected: int,
) -> None:
    trainer = Trainer.__new__(Trainer)
    trainer.config = SimpleNamespace(  # pyrefly: ignore [bad-assignment]
        sdc_replayer=sdc_config
    )
    trainer.gradient_accumulation_steps = gradient_accumulation_steps

    assert Trainer._num_cuda_graph_warmup_iterations(trainer) == expected


def test_cuda_graph_wrapper_returns_graph_owned_output():
    class PassthroughCUDAGraphWrapper:
        def __init__(self, fn, example_inputs, *, num_warmup_iterations=1):
            self.fn = fn
            assert num_warmup_iterations == 2

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
        runner = wrap_with_cuda_graph(fwd_bwd, num_warmup_iterations=2)
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


def test_cuda_graph_wrapper_preserves_structured_args_and_kwargs():
    class PassthroughCUDAGraphWrapper:
        def __init__(self, fn, example_inputs, *, num_warmup_iterations=1):
            self.fn = fn
            assert num_warmup_iterations == 2

        def __call__(self, *args):
            return self.fn(*args)

    fn = MagicMock(side_effect=lambda batches, *, scale: batches[1]["x"] * scale)
    with (
        patch("torchtitan.distributed.cudagraph.utils.device_type", "cuda"),
        patch("torch.cuda.is_available", return_value=True),
        patch.object(torch.version, "hip", None),
        patch(
            "torchtitan.distributed.cudagraph.CUDAGraphWrapper",
            PassthroughCUDAGraphWrapper,
        ),
    ):
        run = wrap_with_cuda_graph(fn, num_warmup_iterations=2)
        output = run(
            [{"x": torch.tensor(1.0)}, {"x": torch.tensor(2.0)}],
            scale=torch.tensor(3.0),
        )

    torch.testing.assert_close(output, torch.tensor(6.0))
    fn.assert_called_once()
    batches = fn.call_args.args[0]
    torch.testing.assert_close(batches[0]["x"], torch.tensor(1.0))
    torch.testing.assert_close(batches[1]["x"], torch.tensor(2.0))
    torch.testing.assert_close(fn.call_args.kwargs["scale"], torch.tensor(3.0))


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
            num_pp_microbatches=1,
            device=torch.device("cpu"),
            forward_backward_step=forward_backward_step,
            sdc_replayer=None,
            model_parts=[],
            checkpointer=SimpleNamespace(maybe_wait_for_staging=MagicMock()),
            metrics_processor=metrics_processor,
            step=1,
            ntokens_seen=3,
        ),
    )
    data_iterator = iter([_batch() for _ in range(3)])

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
            data_iterator=iter([_batch() for _ in range(3)]),
        )

    metrics_processor.log.assert_not_called()


def test_train_step_replay_checks_only_first_forward_backward():
    forward_backward_step = MagicMock(return_value=torch.tensor(1.0))
    replayer = SimpleNamespace(
        run_fwd_bwd=MagicMock(side_effect=lambda execute, **kwargs: execute()),
    )
    trainer = cast(
        Trainer,
        SimpleNamespace(
            config=SimpleNamespace(
                training=SimpleNamespace(disable_cuda_graphs=True, max_norm=1.0),
            ),
            optimizers=MagicMock(),
            lr_schedulers=SimpleNamespace(get_metrics=lambda: {}, step=MagicMock()),
            parallel_dims=SimpleNamespace(
                dp_enabled=False,
                pp_enabled=False,
                dp_cp_enabled=False,
                ep_enabled=False,
                get_optional_mesh=lambda name: None,
            ),
            gradient_accumulation_steps=2,
            num_pp_microbatches=1,
            device=torch.device("cpu"),
            forward_backward_step=forward_backward_step,
            sdc_replayer=replayer,
            model_parts=[],
            checkpointer=SimpleNamespace(maybe_wait_for_staging=MagicMock()),
            metrics_processor=SimpleNamespace(should_log=MagicMock(return_value=False)),
            step=1,
            ntokens_seen=0,
        ),
    )

    with patch(
        "torchtitan.trainer.dist_utils.clip_grad_norm_",
        return_value=torch.tensor(1.0),
    ):
        Trainer.train_step(
            trainer,
            iter([_batch() for _ in range(2)]),
        )

    replayer.run_fwd_bwd.assert_called_once()
    assert replayer.run_fwd_bwd.call_args.kwargs == {"step": 1}
    assert forward_backward_step.call_count == 2


def test_replay_failure_happens_before_optimizer():
    mismatch = SDCReplayMismatch(
        step=1,
        local_step=1,
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
            num_pp_microbatches=1,
            device=torch.device("cpu"),
            forward_backward_step=MagicMock(),
            sdc_replayer=SimpleNamespace(run_fwd_bwd=MagicMock(side_effect=mismatch)),
            model_parts=[],
            checkpointer=SimpleNamespace(maybe_wait_for_staging=MagicMock()),
            metrics_processor=SimpleNamespace(should_log=MagicMock(return_value=False)),
            step=1,
            ntokens_seen=0,
        ),
    )

    with pytest.raises(SDCReplayMismatch):
        Trainer.train_step(
            trainer,
            iter([_batch()]),
        )

    optimizers.step.assert_not_called()


def test_loading_checkpoint_rearms_replay_schedule():
    replayer = SimpleNamespace(reset_schedule=MagicMock())
    trainer = cast(Trainer, SimpleNamespace(sdc_replayer=replayer))

    Trainer.load_state_dict(trainer, {"step": 12, "ntokens_seen": 34})

    assert trainer.step == 12
    assert trainer.ntokens_seen == 34
    replayer.reset_schedule.assert_called_once_with()

    disabled = cast(Trainer, SimpleNamespace(sdc_replayer=None))
    Trainer.load_state_dict(disabled, {"step": 1, "ntokens_seen": 2})
    assert disabled.step == 1


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
