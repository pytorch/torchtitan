# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import torch
from torchtitan.distributed.cudagraph import wrap_with_cuda_graph
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
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


def _make_trainer(**attributes: Any) -> Trainer:
    trainer = Trainer.__new__(Trainer)
    for name, value in attributes.items():
        setattr(trainer, name, value)
    return trainer


def test_graph_trainer_keeps_its_cuda_graph_owner() -> None:
    trainer = GraphTrainer.__new__(GraphTrainer)
    trainer.config = SimpleNamespace(
        training=SimpleNamespace(disable_cuda_graphs=False),
        sdc_replayer=None,
    )
    gradient_accumulation_body = MagicMock()
    trainer._gradient_accumulation_body = gradient_accumulation_body

    with patch("torchtitan.trainer.wrap_with_cuda_graph") as wrap:
        trainer._init_gradient_accumulation()

    assert trainer._run_gradient_accumulation is gradient_accumulation_body
    wrap.assert_not_called()


def test_graph_trainer_rejects_optimizer_cuda_graph() -> None:
    config = SimpleNamespace(training=SimpleNamespace(enable_optimizer_cuda_graph=True))

    with (
        patch.object(Trainer, "__init__") as init,
        pytest.raises(ValueError, match="not supported with GraphTrainer"),
    ):
        GraphTrainer(config)

    init.assert_not_called()


def test_pp_forward_backward_step_returns_sentinel_without_last_stage():
    sentinel = torch.full((1,), -1.0)
    trainer = _make_trainer(
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
        dataloader=SimpleNamespace(max_num_documents=None),
        config=SimpleNamespace(
            parallelism="PARA",
            training=SimpleNamespace(max_context_length=2048),
        ),
        ntokens_seen=0,
        device=torch.device("cpu"),
        _pp_loss_sentinel_on_non_last_stage=sentinel,
    )
    loss = Trainer.forward_backward_step(
        trainer,
        prepared_inputs=trainer._preprocess_accumulation_step_inputs(
            [({"input": torch.ones(1)}, torch.ones(1))]
        ),
        global_valid_tokens=torch.tensor(1),
    )

    assert loss is sentinel


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

    trainer = _make_trainer(
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
        dataloader=SimpleNamespace(max_num_documents=None),
        config=SimpleNamespace(
            parallelism="PARA",
            training=SimpleNamespace(max_context_length=2048),
        ),
        ntokens_seen=0,
        device=torch.device("cpu"),
    )
    reporting_loss = Trainer.forward_backward_step(
        trainer,
        prepared_inputs=trainer._preprocess_accumulation_step_inputs(
            [({"input": torch.ones(1)}, torch.ones(1))] * 2
        ),
        global_valid_tokens=torch.tensor(2),
    )

    torch.testing.assert_close(reporting_loss, torch.tensor(5.0))
    torch.testing.assert_close(torch.stack(gradients), torch.tensor([2.0, 4.0]))
    assert not reporting_loss.requires_grad
    assert reporting_loss.grad_fn is None
    assert loss_containers == [[]]
    assert all(reference() is None for reference in loss_refs)
    assert all(reference() is None for reference in activation_refs)


def test_pp_forward_backward_step_prepares_structured_inputs() -> None:
    fwd_bwd_fn = MagicMock(return_value=torch.tensor(0.0))

    class _FakeModel:
        def preprocess_inputs(self, input_dict, **kwargs):
            return (
                input_dict["input"] + 1,
                input_dict["labels"] + 2,
                {"positions": input_dict["positions"] + 3},
            )

    trainer = _make_trainer(
        pp_has_first_stage=True,
        pp_has_last_stage=True,
        model_parts=[_FakeModel()],
        parallel_dims=SimpleNamespace(pp_enabled=True),
        dataloader=SimpleNamespace(max_num_documents=4),
        config=SimpleNamespace(
            parallelism="PARA",
            training=SimpleNamespace(max_context_length=2048),
        ),
        ntokens_seen=0,
        device=torch.device("cpu"),
        _pp_forward_backward_body=fwd_bwd_fn,
    )
    global_valid_tokens = torch.tensor(2)

    prepared_inputs = trainer._preprocess_accumulation_step_inputs(
        [
            (
                {"input": torch.tensor(1), "positions": torch.tensor(10)},
                torch.tensor([3]),
            ),
            (
                {"input": torch.tensor(2), "positions": torch.tensor(20)},
                torch.tensor([4]),
            ),
        ]
    )
    result = Trainer.forward_backward_step(
        trainer,
        prepared_inputs=prepared_inputs,
        global_valid_tokens=global_valid_tokens,
    )

    torch.testing.assert_close(result, torch.tensor(0.0))
    arg_mbs, kwarg_mbs, target_mbs, passed_valid_tokens = fwd_bwd_fn.call_args.args
    torch.testing.assert_close(arg_mbs[0][0], torch.tensor(2))
    torch.testing.assert_close(arg_mbs[1][0], torch.tensor(3))
    torch.testing.assert_close(kwarg_mbs[0]["positions"], torch.tensor(13))
    torch.testing.assert_close(kwarg_mbs[1]["positions"], torch.tensor(23))
    torch.testing.assert_close(target_mbs[0], torch.tensor([5]))
    torch.testing.assert_close(target_mbs[1], torch.tensor([6]))
    assert passed_valid_tokens is global_valid_tokens
    assert trainer.ntokens_seen == 2


def test_forward_backward_step_counts_cp_local_tokens_and_forwards_inputs():
    captured = {}

    class _FakeModel:
        def preprocess_inputs(self, input_dict, **kw):
            captured["preprocess_kwargs"] = kw
            return ("INPUTS", torch.ones(7), {"positions": 1})

    def fwd_bwd_fn(inputs, labels, global_valid_tokens, extra_kwargs):
        captured["fwd_bwd_args"] = (inputs, labels, extra_kwargs)
        return torch.tensor(0.0)

    trainer = _make_trainer(
        model_parts=[_FakeModel()],
        dataloader=SimpleNamespace(max_num_documents=4),
        parallel_dims=SimpleNamespace(pp_enabled=False),
        config=SimpleNamespace(
            parallelism="PARA",
            training=SimpleNamespace(
                disable_cuda_graphs=True,
                max_context_length=2048,
            ),
        ),
        ntokens_seen=100,
        device=torch.device("cpu"),
        _forward_backward_body=fwd_bwd_fn,
    )

    prepared_inputs = trainer._preprocess_accumulation_step_inputs(
        [({"input": 0}, torch.zeros(14))]
    )
    Trainer.forward_backward_step(
        trainer,
        prepared_inputs=prepared_inputs,
        global_valid_tokens=torch.tensor(1),
    )

    inputs, labels, extra = captured["fwd_bwd_args"]
    assert inputs == "INPUTS"
    assert extra == {"positions": 1}
    assert labels.numel() == 7
    assert trainer.ntokens_seen == 107
    assert captured["preprocess_kwargs"] == {
        "parallel_dims": trainer.parallel_dims,
        "parallelism": "PARA",
        "max_num_documents": 4,
        "max_context_length": 2048,
    }


def test_gradient_accumulation_body_defers_fsdp_reduction():
    calls = []

    class FakeFSDP:
        def set_is_last_backward(self, value):
            calls.append(("last", value))

        def set_reshard_after_backward(self, value):
            calls.append(("reshard", value))

        def set_requires_gradient_sync(self, value):
            calls.append(("sync", value))

        def finalize_gradient_accumulation(self):
            calls.append(("finalize", True))

    fsdp_module = FakeFSDP()
    trainer = _make_trainer(
        config=SimpleNamespace(
            parallelism=SimpleNamespace(fsdp_defer_gradient_reduction=True)
        ),
        parallel_dims=SimpleNamespace(
            pp_enabled=False,
            dp_replicate_enabled=False,
        ),
        model_parts=[fsdp_module],
        _fsdp_root=fsdp_module,
        forward_backward_step=MagicMock(
            side_effect=(torch.tensor(1.0), torch.tensor(2.0))
        ),
    )

    loss = Trainer._gradient_accumulation_body(
        trainer,
        [("x0", "y0"), ("x1", "y1")],
        torch.tensor(2),
    )

    torch.testing.assert_close(loss, torch.tensor(3.0))
    assert calls == [
        ("last", False),
        ("reshard", False),
        ("sync", False),
        ("sync", True),
        ("finalize", True),
    ]


def test_gradient_accumulation_body_finalizes_only_last_schedule():
    finalize_gradients = []
    losses = iter((torch.tensor(1.0), torch.tensor(2.0)))

    def forward_backward_step(**kwargs):
        finalize_gradients.append(kwargs["finalize_gradients"])
        return next(losses)

    trainer = _make_trainer(
        config=SimpleNamespace(
            parallelism=SimpleNamespace(fsdp_defer_gradient_reduction=True)
        ),
        parallel_dims=SimpleNamespace(
            pp_enabled=True,
            dp_replicate_enabled=False,
        ),
        model_parts=[],
        _fsdp_root=None,
        forward_backward_step=forward_backward_step,
    )
    accumulation_step_inputs = [
        ([{"input": torch.tensor(1)}], [torch.tensor(1)]),
        ([{"input": torch.tensor(2)}], [torch.tensor(2)]),
    ]

    loss = Trainer._gradient_accumulation_body(
        trainer, accumulation_step_inputs, torch.tensor(2)
    )

    torch.testing.assert_close(loss, torch.tensor(3.0))
    assert finalize_gradients == [False, True]


def test_cuda_graph_passes_local_gradient_state() -> None:
    model = torch.nn.Linear(2, 2)
    trainer = _make_trainer(
        config=SimpleNamespace(
            training=SimpleNamespace(disable_cuda_graphs=False),
            sdc_replayer=None,
        ),
        parallel_dims=SimpleNamespace(pp_enabled=False),
        model_parts=[model],
    )
    with patch(
        "torchtitan.trainer.wrap_with_cuda_graph",
        side_effect=lambda fn, **kwargs: fn,
    ) as wrap:
        Trainer._init_gradient_accumulation(trainer)

    gradient_state = wrap.call_args.kwargs["gradient_state"]
    assert trainer._cudagraph_gradient_state is gradient_state
    assert gradient_state.parameters == tuple(model.parameters())


def test_optimizer_step_body_clips_before_update() -> None:
    events = []
    optimizers = MagicMock()
    optimizers.step.side_effect = lambda: events.append("step")
    trainer = _make_trainer(
        config=SimpleNamespace(
            training=SimpleNamespace(
                max_norm=1.0,
                enable_optimizer_cuda_graph=True,
            )
        ),
        parallel_dims=SimpleNamespace(
            pp_enabled=False,
            ep_enabled=False,
            get_optional_mesh=lambda name: None,
        ),
        model_parts=[SimpleNamespace(parameters=lambda: [])],
        optimizers=optimizers,
    )

    def clip_grad_norm(*args, **kwargs):
        events.append("clip")
        return torch.tensor(2.0)

    with patch(
        "torchtitan.trainer.dist_utils.clip_grad_norm_",
        side_effect=clip_grad_norm,
    ):
        grad_norm = Trainer._optimizer_step_body(
            trainer, torch.ones((), dtype=torch.int32)
        )

    torch.testing.assert_close(grad_norm, torch.tensor(2.0))
    assert events == ["clip", "step"]


def test_optimizer_cuda_graph_waits_for_forward_backward_capture() -> None:
    model = torch.nn.Linear(2, 2)
    trainer = _make_trainer(
        config=SimpleNamespace(
            training=SimpleNamespace(disable_cuda_graphs=False),
            sdc_replayer=None,
        ),
        parallel_dims=SimpleNamespace(pp_enabled=False),
        model_parts=[model],
        optimizers=MagicMock(),
    )
    with patch(
        "torchtitan.trainer.wrap_with_cuda_graph",
        side_effect=lambda fn, **kwargs: fn,
    ):
        Trainer._init_gradient_accumulation(trainer)

    with pytest.raises(AssertionError, match="forward-backward CUDA graph"):
        Trainer._prepare_optimizer_cuda_graph(trainer)

    assert trainer._cudagraph_gradient_state is not None
    trainer._cudagraph_gradient_state.record()
    Trainer._prepare_optimizer_cuda_graph(trainer)
    trainer.optimizers.prepare_for_cuda_graph.assert_called_once_with()


def test_optimizer_step_is_wrapped_separately() -> None:
    trainer = _make_trainer(
        config=SimpleNamespace(
            training=SimpleNamespace(enable_optimizer_cuda_graph=True)
        )
    )
    wrapped = MagicMock()
    with patch("torchtitan.trainer.wrap_with_cuda_graph", return_value=wrapped) as wrap:
        Trainer._init_optimizer_step_function(trainer)

    assert trainer.optimizer_step_fn is wrapped
    assert wrap.call_args.args == (trainer._optimizer_step_body,)
    assert wrap.call_args.kwargs["sdc_num_steps"] == 0
    assert wrap.call_args.kwargs["sdc_num_replays"] == 0
    assert (
        wrap.call_args.kwargs["capture_setup"] == trainer._prepare_optimizer_cuda_graph
    )


def test_cuda_graph_wrapper_returns_graph_owned_output():
    class PassthroughCUDAGraphWrapper:
        def __init__(
            self,
            fn,
            example_inputs,
            *,
            num_warmup_iterations=1,
            gradient_state=None,
            capture_setup=None,
        ):
            self.fn = fn
            assert num_warmup_iterations == 2
            assert gradient_state is None
            assert capture_setup is None

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
        runner = wrap_with_cuda_graph(
            fwd_bwd,
            sdc_num_steps=0,
            sdc_num_replays=0,
        )
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
        def __init__(
            self,
            fn,
            example_inputs,
            *,
            num_warmup_iterations=1,
            gradient_state=None,
            capture_setup=None,
        ):
            self.fn = fn
            assert num_warmup_iterations == 2
            assert gradient_state is None
            assert capture_setup is None

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
        run = wrap_with_cuda_graph(
            fn,
            sdc_num_steps=0,
            sdc_num_replays=0,
        )
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
    prepared_positions = []

    def run_gradient_accumulation(accumulation_step_inputs, global_valid_tokens):
        prepared_positions.append(
            [step[2]["positions"].item() for step in accumulation_step_inputs]
        )
        graph_loss.fill_(sum(next(loss_values) for _ in accumulation_step_inputs))
        return graph_loss

    model = SimpleNamespace(
        preprocess_inputs=lambda input_dict, **kwargs: (
            input_dict["input"],
            input_dict["labels"],
            {"positions": input_dict["positions"]},
        ),
        parameters=lambda: [],
    )

    metrics_processor = SimpleNamespace(
        should_log=MagicMock(return_value=True),
        log=MagicMock(),
    )
    trainer = _make_trainer(
        config=SimpleNamespace(
            training=SimpleNamespace(
                disable_cuda_graphs=False,
                max_norm=1.0,
                max_context_length=2048,
                enable_optimizer_cuda_graph=True,
            ),
            parallelism="PARA",
        ),
        dataloader=SimpleNamespace(max_num_documents=None),
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
        _run_gradient_accumulation=run_gradient_accumulation,
        optimizer_step_fn=MagicMock(return_value=torch.tensor(4.0)),
        sdc_replayer=None,
        model_parts=[model],
        checkpointer=SimpleNamespace(maybe_wait_for_staging=MagicMock()),
        metrics_processor=metrics_processor,
        step=1,
        ntokens_seen=0,
    )

    def batch(position):
        input_dict, labels = _batch()
        input_dict["positions"] = torch.tensor(position)
        return input_dict, labels

    data_iterator = iter([batch(position) for position in range(1, 4)])

    with patch(
        "torchtitan.trainer.dist_utils.clip_grad_norm_",
        return_value=torch.tensor(4.0),
    ):
        Trainer.train_step(trainer, data_iterator)

    trainer.optimizers.zero_grad.assert_called_once_with(set_to_none=True)
    metrics_processor.log.assert_called_once_with(
        1,
        6.0,
        6.0,
        4.0,
        extra_metrics={"n_tokens_seen": 3},
    )

    metrics_processor.should_log.return_value = False
    metrics_processor.log.reset_mock()
    trainer.optimizers.zero_grad.reset_mock()
    with patch(
        "torchtitan.trainer.dist_utils.clip_grad_norm_",
        return_value=torch.tensor(4.0),
    ):
        Trainer.train_step(
            trainer,
            data_iterator=iter([batch(position) for position in range(4, 7)]),
        )

    trainer.optimizers.zero_grad.assert_called_once_with(set_to_none=True)
    metrics_processor.log.assert_not_called()
    assert prepared_positions == [[1, 2, 3], [4, 5, 6]]


def test_train_step_replay_checks_whole_accumulation():
    run_gradient_accumulation = MagicMock(return_value=torch.tensor(1.0))
    replayer = SimpleNamespace(
        run_fwd_bwd=MagicMock(side_effect=lambda execute, **kwargs: execute()),
    )
    trainer = _make_trainer(
        config=SimpleNamespace(
            training=SimpleNamespace(
                disable_cuda_graphs=True,
                max_norm=1.0,
                max_context_length=2048,
                enable_optimizer_cuda_graph=False,
            ),
            parallelism="PARA",
        ),
        dataloader=SimpleNamespace(max_num_documents=None),
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
        _run_gradient_accumulation=run_gradient_accumulation,
        sdc_replayer=replayer,
        model_parts=[
            SimpleNamespace(
                preprocess_inputs=lambda input_dict, **kwargs: (
                    input_dict["input"],
                    input_dict["labels"],
                    {},
                ),
                parameters=lambda: [],
            )
        ],
        checkpointer=SimpleNamespace(maybe_wait_for_staging=MagicMock()),
        metrics_processor=SimpleNamespace(should_log=MagicMock(return_value=False)),
        step=1,
        ntokens_seen=0,
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
    run_gradient_accumulation.assert_called_once()


def test_replay_failure_happens_before_optimizer():
    mismatch = SDCReplayMismatch(
        step=1,
        local_step=1,
        replay=1,
        rank=0,
        signature_mismatch="loss",
    )
    optimizers = MagicMock()
    trainer = _make_trainer(
        config=SimpleNamespace(
            training=SimpleNamespace(
                disable_cuda_graphs=True,
                max_norm=1.0,
                max_context_length=2048,
                enable_optimizer_cuda_graph=False,
            ),
            parallelism="PARA",
        ),
        dataloader=SimpleNamespace(max_num_documents=None),
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
        _run_gradient_accumulation=MagicMock(),
        sdc_replayer=SimpleNamespace(run_fwd_bwd=MagicMock(side_effect=mismatch)),
        model_parts=[
            SimpleNamespace(
                preprocess_inputs=lambda input_dict, **kwargs: (
                    input_dict["input"],
                    input_dict["labels"],
                    {},
                ),
                parameters=lambda: [],
            )
        ],
        checkpointer=SimpleNamespace(maybe_wait_for_staging=MagicMock()),
        metrics_processor=SimpleNamespace(should_log=MagicMock(return_value=False)),
        step=1,
        ntokens_seen=0,
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
        runner = wrap_with_cuda_graph(
            fwd_bwd,
            sdc_num_steps=0,
            sdc_num_replays=0,
        )

    assert runner is fwd_bwd
    warning.assert_called_once()


class _RecordingFSDPPart:
    def __init__(self) -> None:
        self.requires_all_reduce_calls: list[bool] = []

    def set_requires_all_reduce(self, flag: bool, *, recurse: bool = True) -> None:
        assert recurse is True
        self.requires_all_reduce_calls.append(flag)

    def parameters(self):
        return iter(())


def _run_gradient_accumulation_recording_all_reduce(
    *,
    dp_replicate_enabled: bool,
    gradient_accumulation_steps: int,
) -> list[bool]:
    part = _RecordingFSDPPart()
    trainer = _make_trainer(
        config=SimpleNamespace(
            parallelism=SimpleNamespace(fsdp_defer_gradient_reduction=False),
        ),
        parallel_dims=SimpleNamespace(
            pp_enabled=False,
            dp_replicate_enabled=dp_replicate_enabled,
        ),
        forward_backward_step=MagicMock(return_value=torch.tensor(1.0)),
        model_parts=[part],
        _fsdp_root=None,
    )
    Trainer._gradient_accumulation_body(
        trainer,
        [("input", "labels")] * gradient_accumulation_steps,
        torch.tensor(1),
    )
    return part.requires_all_reduce_calls


def test_hsdp_skips_replicate_all_reduce_until_last_accum_group():
    flags = _run_gradient_accumulation_recording_all_reduce(
        dp_replicate_enabled=True,
        gradient_accumulation_steps=3,
    )
    assert flags == [False, False, True]


def test_hsdp_keeps_all_reduce_on_single_accum_group():
    flags = _run_gradient_accumulation_recording_all_reduce(
        dp_replicate_enabled=True,
        gradient_accumulation_steps=1,
    )
    assert flags == [True]


def test_pure_fsdp_does_not_toggle_requires_all_reduce():
    flags = _run_gradient_accumulation_recording_all_reduce(
        dp_replicate_enabled=False,
        gradient_accumulation_steps=3,
    )
    assert flags == []
