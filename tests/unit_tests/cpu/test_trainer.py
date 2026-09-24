# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import weakref
from functools import partial
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import torch
from torchtitan.components.data.types import (
    TokenizedTrainingMicrobatch,
    TrainingMicrobatch,
)
from torchtitan.distributed.cuda_graph import wrap_with_cuda_graph
from torchtitan.observability.metrics import compute_training_performance_metrics
from torchtitan.observability.sdc_replayer import SDCReplayMismatch
from torchtitan.trainer import Trainer
from torchtitan.training_engine import ForwardBackwardResult, TrainingEngine


def _batch() -> TokenizedTrainingMicrobatch:
    """One microbatch produced by ``Trainer.microbatch_generator``.

    Built fresh per call because tests may mutate its model-facing dictionary.
    """
    return TokenizedTrainingMicrobatch(
        input=torch.ones(1),
        labels=torch.ones(1, dtype=torch.long),
        positions=torch.zeros(1, dtype=torch.long),
        padding_mask=torch.zeros(1, dtype=torch.bool),
        num_valid_tokens=1,
    )


class _DictTrainingMicrobatch(TrainingMicrobatch):
    def __init__(
        self,
        input_dict: dict[str, Any],
        loss_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._input_dict = input_dict
        self._loss_kwargs = loss_kwargs or {}
        self.labels = input_dict["labels"]
        self.num_valid_tokens = self.labels.numel()
        self.to_input_dict_calls: list[tuple[torch.device | str, bool]] = []
        self.to_loss_kwargs_calls: list[tuple[torch.device | str, bool]] = []

    def as_input_dict(self) -> dict[str, Any]:
        return dict(self._input_dict)

    def to_input_dict(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> dict[str, Any]:
        self.to_input_dict_calls.append((device, non_blocking))
        return super().to_input_dict(device, non_blocking=non_blocking)

    def loss_kwargs(self) -> dict[str, Any]:
        return dict(self._loss_kwargs)

    def to_loss_kwargs(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> dict[str, Any]:
        self.to_loss_kwargs_calls.append((device, non_blocking))
        return super().to_loss_kwargs(device, non_blocking=non_blocking)


def _dict_microbatch(
    input_dict: dict[str, Any],
    loss_kwargs: dict[str, Any] | None = None,
) -> TrainingMicrobatch:
    return _DictTrainingMicrobatch(input_dict, loss_kwargs)


def _training_loop(trainer: TrainingEngine) -> SimpleNamespace:
    if not hasattr(trainer, "_num_optimizer_steps_since_cuda_graph_init"):
        trainer._num_optimizer_steps_since_cuda_graph_init = 0
    if not hasattr(trainer, "loss_is_finite"):
        trainer.loss_is_finite = torch.ones((), dtype=torch.int32)
    if not hasattr(trainer, "optimizer_step"):
        trainer.optimizer_step = lambda: TrainingEngine.optimizer_step(trainer)

    return SimpleNamespace(
        engine=trainer,
        config=trainer.config,
        gradient_accumulation_steps=trainer.gradient_accumulation_steps,
        num_pp_microbatches=trainer.num_pp_microbatches,
        metrics_processor=trainer.metrics_processor,
    )


def test_microbatch_generator_preserves_labels() -> None:
    labels = torch.ones(1, dtype=torch.long)
    microbatch = TokenizedTrainingMicrobatch(
        input=torch.ones(1),
        labels=labels,
        positions=torch.zeros(1, dtype=torch.long),
        padding_mask=torch.zeros(1, dtype=torch.bool),
        num_valid_tokens=1,
    )
    trainer = cast(
        Trainer,
        SimpleNamespace(
            config=SimpleNamespace(
                training=SimpleNamespace(
                    num_tokens_per_microbatch_per_dp_rank=1,
                )
            ),
            metrics_processor=SimpleNamespace(
                ntokens_since_last_log=0,
                data_loading_times=[],
            ),
        ),
    )

    output = next(Trainer.microbatch_generator(trainer, [microbatch]))

    assert output is microbatch
    assert output.labels is labels
    assert trainer.metrics_processor.ntokens_since_last_log == 1


def test_pp_forward_backward_microbatch_group_returns_sentinel_without_last_stage(
    monkeypatch,
) -> None:
    sentinel = torch.full((1,), -1.0)
    trainer = cast(
        TrainingEngine,
        SimpleNamespace(
            pp_has_last_stage=False,
            pp_schedule=SimpleNamespace(step=lambda **kwargs: None),
            parallel_dims=SimpleNamespace(),
            config=SimpleNamespace(
                debug=SimpleNamespace(spmd_typechecking=False),
            ),
            device=torch.device("cpu"),
            _pp_loss_sentinel_on_non_last_stage=sentinel,
        ),
    )
    monkeypatch.setattr(
        "torchtitan.training_engine.dist_utils.get_spmd_context",
        lambda **kwargs: contextlib.nullcontext(),
    )

    loss = TrainingEngine._pp_forward_backward_microbatch_group(
        trainer,
        inputs=None,
        labels=None,
        model_kwargs=[{}],
        loss_kwargs={"global_valid_tokens": torch.tensor(1)},
    )

    assert loss is sentinel


def test_pp_forward_backward_microbatch_group_releases_consumed_loss_graphs(
    monkeypatch,
) -> None:
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
        TrainingEngine,
        SimpleNamespace(
            pp_has_last_stage=True,
            pp_schedule=SimpleNamespace(step=schedule_step),
            parallel_dims=SimpleNamespace(),
            config=SimpleNamespace(
                debug=SimpleNamespace(spmd_typechecking=False),
            ),
            device=torch.device("cpu"),
        ),
    )
    monkeypatch.setattr(
        "torchtitan.training_engine.dist_utils.get_spmd_context",
        lambda **kwargs: contextlib.nullcontext(),
    )

    reporting_loss = TrainingEngine._pp_forward_backward_microbatch_group(
        trainer,
        inputs=[(torch.ones(1),), (torch.ones(1),)],
        labels=[torch.ones(1), torch.ones(1)],
        model_kwargs=[{}, {}],
        loss_kwargs={"global_valid_tokens": torch.tensor(2)},
    )

    torch.testing.assert_close(reporting_loss, torch.tensor(5.0))
    torch.testing.assert_close(torch.stack(gradients), torch.tensor([2.0, 4.0]))
    assert not reporting_loss.requires_grad
    assert reporting_loss.grad_fn is None
    assert loss_containers == [[]]
    assert all(reference() is None for reference in loss_refs)
    assert all(reference() is None for reference in activation_refs)


def test_preprocess_microbatch_groups_prepares_structured_pp_inputs() -> None:
    class _FakeModel:
        def preprocess_inputs(self, input_dict, **kwargs):
            return (
                input_dict["input"] + 1,
                input_dict["labels"] + 2,
                {"positions": input_dict["positions"] + 3},
            )

    trainer = cast(
        TrainingEngine,
        SimpleNamespace(
            pp_has_first_stage=True,
            pp_has_last_stage=True,
            model_parts=[_FakeModel()],
            parallel_dims=SimpleNamespace(pp_enabled=True, cp=1),
            max_num_documents=4,
            preprocess_inputs_kwargs={},
            config=SimpleNamespace(
                parallelism="PARA",
                training=SimpleNamespace(
                    max_context_length=2048,
                    num_tokens_per_microbatch_per_dp_rank=1,
                ),
            ),
            ntokens_seen=0,
            device=torch.device("cpu"),
        ),
    )
    microbatches = [
        _dict_microbatch(
            {
                "input": torch.tensor(1),
                "positions": torch.tensor(10),
                "labels": torch.tensor([3]),
            }
        ),
        _dict_microbatch(
            {
                "input": torch.tensor(2),
                "positions": torch.tensor(20),
                "labels": torch.tensor([4]),
            }
        ),
    ]

    [(arg_mbs, kwarg_mbs, target_mbs)] = TrainingEngine._preprocess_microbatch_groups(
        trainer, [microbatches]
    )

    assert arg_mbs is not None
    assert target_mbs is not None
    torch.testing.assert_close(arg_mbs[0][0], torch.tensor(2))
    torch.testing.assert_close(arg_mbs[1][0], torch.tensor(3))
    torch.testing.assert_close(kwarg_mbs[0]["positions"], torch.tensor(13))
    torch.testing.assert_close(kwarg_mbs[1]["positions"], torch.tensor(23))
    torch.testing.assert_close(target_mbs[0], torch.tensor([5]))
    torch.testing.assert_close(target_mbs[1], torch.tensor([6]))
    assert trainer.ntokens_seen == 2
    for microbatch in microbatches:
        assert isinstance(microbatch, _DictTrainingMicrobatch)
        assert microbatch.to_input_dict_calls == [(trainer.device, True)]
        assert microbatch.to_loss_kwargs_calls == [(trainer.device, True)]


def test_preprocess_microbatch_groups_rejects_pp_loss_kwargs() -> None:
    trainer = cast(
        TrainingEngine,
        SimpleNamespace(
            pp_has_first_stage=True,
            pp_has_last_stage=True,
            model_parts=[
                SimpleNamespace(
                    preprocess_inputs=lambda input_dict, **kwargs: (
                        input_dict["input"],
                        input_dict["labels"],
                        {},
                    )
                )
            ],
            parallel_dims=SimpleNamespace(pp_enabled=True, cp=1),
            max_num_documents=None,
            preprocess_inputs_kwargs={},
            config=SimpleNamespace(
                parallelism="PARA",
                training=SimpleNamespace(
                    max_context_length=1,
                    num_tokens_per_microbatch_per_dp_rank=1,
                ),
            ),
            device=torch.device("cpu"),
            ntokens_seen=0,
        ),
    )
    microbatch = _dict_microbatch(
        {"input": torch.tensor([1]), "labels": torch.tensor([1])},
        {"advantages": torch.tensor([0.1])},
    )

    with pytest.raises(ValueError, match="pipeline parallelism"):
        TrainingEngine._preprocess_microbatch_groups(
            trainer,
            [[microbatch]],
        )


def test_forward_backward_runs_whole_accumulation() -> None:
    captured: dict[str, Any] = {}

    class _FakeModel:
        def preprocess_inputs(self, input_dict, **kw):
            captured["preprocess_kwargs"] = kw
            return ("INPUTS", torch.ones(7), {"positions": 1})

    losses = iter((torch.tensor(1.0), torch.tensor(2.0)))

    def forward_backward_body(*, inputs, labels, model_kwargs, loss_kwargs):
        captured.setdefault("fwd_bwd_args", []).append((inputs, labels, model_kwargs))
        torch.testing.assert_close(loss_kwargs["global_valid_tokens"], torch.tensor(2))
        torch.testing.assert_close(loss_kwargs["advantages"], torch.tensor([0.1]))
        assert loss_kwargs["reduction"] == "sum"
        engine.loss_metrics = {"loss/mean": next(losses)}
        return engine.loss_metrics["loss/mean"]

    engine = object.__new__(TrainingEngine)
    engine.model_parts = [_FakeModel()]
    engine.max_num_documents = 4
    engine.parallel_dims = SimpleNamespace(
        pp_enabled=False,
        cp=1,
        fsdp_enabled=False,
        dp_replicate_enabled=False,
    )
    engine.config = SimpleNamespace(
        parallelism=SimpleNamespace(
            fsdp_defer_gradient_reduction=False,
            fsdp_reshard_after_forward="default",
        ),
        training=SimpleNamespace(
            disable_cuda_graphs=True,
            max_context_length=2048,
            num_tokens_per_microbatch_per_dp_rank=7,
        ),
    )
    engine.preprocess_inputs_kwargs = {"processor": "VALUE"}
    engine.ntokens_seen = 100
    engine.num_completed_steps = 0
    engine.device = torch.device("cpu")
    engine.gc_handler = SimpleNamespace(run=MagicMock())
    engine.optimizers = SimpleNamespace(zero_grad=MagicMock())
    engine.sdc_replayer = None
    engine._non_pp_forward_backward_microbatch = forward_backward_body
    engine._run_forward_backward = partial(
        TrainingEngine._forward_backward_body,
        engine,
        defer_fsdp_gradient_reduction=False,
    )
    microbatches = [
        _dict_microbatch(
            {"input": index, "labels": torch.zeros(1)},
            {"advantages": torch.tensor([0.1]), "reduction": "sum"},
        )
        for index in range(2)
    ]

    result = TrainingEngine.forward_backward(
        engine,
        microbatch_groups=[[microbatch] for microbatch in microbatches],
        global_valid_tokens=2,
    )

    torch.testing.assert_close(result.loss, torch.tensor(3.0))
    assert [metrics["loss/mean"].item() for metrics in result.loss_metrics] == [
        1.0,
        2.0,
    ]
    assert engine.num_accumulation_steps == 2
    assert engine.ntokens_seen == 114
    assert engine.loss_is_finite.item() == 1
    engine.gc_handler.run.assert_called_once_with(1)
    engine.optimizers.zero_grad.assert_called_once_with(set_to_none=True)
    for microbatch in microbatches:
        assert isinstance(microbatch, _DictTrainingMicrobatch)
        assert microbatch.to_input_dict_calls == [(engine.device, True)]
        assert microbatch.to_loss_kwargs_calls == [(engine.device, True)]
    for inputs, labels, model_kwargs in captured["fwd_bwd_args"]:
        assert inputs == "INPUTS"
        assert model_kwargs == {"positions": 1}
        assert labels.numel() == 7
    assert captured["preprocess_kwargs"] == {
        "parallel_dims": engine.parallel_dims,
        "parallelism": engine.config.parallelism,
        "max_num_documents": 4,
        "max_context_length": 2048,
        "processor": "VALUE",
    }


def test_cuda_graph_wrapper_returns_graph_owned_output():
    class PassthroughCUDAGraphWrapper:
        def __init__(
            self,
            fn,
            example_inputs,
            *,
            num_warmup_iterations,
        ):
            self.fn = fn
            assert num_warmup_iterations == 0

        def __call__(self, *args):
            return self.fn(*args)

    graph_loss = torch.tensor(0.0)
    fwd_bwd = MagicMock(return_value=graph_loss)

    with (
        patch("torchtitan.distributed.cuda_graph.utils.device_type", "cuda"),
        patch("torch.cuda.is_available", return_value=True),
        patch.object(torch.version, "hip", None),
        patch(
            "torchtitan.distributed.cuda_graph.CUDAGraphWrapper",
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


def test_cuda_graph_wrapper_preserves_structured_args_and_kwargs():
    class PassthroughCUDAGraphWrapper:
        def __init__(
            self,
            fn,
            example_inputs,
            *,
            num_warmup_iterations,
        ):
            self.fn = fn
            assert num_warmup_iterations == 0

        def __call__(self, *args):
            return self.fn(*args)

    fn = MagicMock(side_effect=lambda batches, *, scale: batches[1]["x"] * scale)
    with (
        patch("torchtitan.distributed.cuda_graph.utils.device_type", "cuda"),
        patch("torch.cuda.is_available", return_value=True),
        patch.object(torch.version, "hip", None),
        patch(
            "torchtitan.distributed.cuda_graph.CUDAGraphWrapper",
            PassthroughCUDAGraphWrapper,
        ),
    ):
        run = wrap_with_cuda_graph(fn)
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


def test_training_engine_owns_gradient_accumulation_cuda_graph_warmup() -> None:
    eager_forward_backward = MagicMock(
        return_value=ForwardBackwardResult(torch.tensor(1.0), [])
    )
    cuda_graph_forward_backward = MagicMock(
        return_value=ForwardBackwardResult(torch.tensor(2.0), [])
    )
    engine = cast(
        TrainingEngine,
        SimpleNamespace(
            config=SimpleNamespace(
                sdc_replayer=None,
                debug=SimpleNamespace(spmd_typechecking=False),
                training=SimpleNamespace(disable_cuda_graphs=False),
                parallelism=SimpleNamespace(
                    enable_sequence_parallel=False,
                    fsdp_defer_gradient_reduction=True,
                    fsdp_reshard_after_forward="never",
                ),
            ),
            parallel_dims=SimpleNamespace(pp_enabled=False, fsdp_enabled=True),
            _forward_backward_body=eager_forward_backward,
        ),
    )

    with (
        patch(
            "torchtitan.training_engine.wrap_with_cuda_graph",
            return_value=cuda_graph_forward_backward,
        ) as wrap,
        patch(
            "torchtitan.training_engine.run_eager_on_cuda_graph_stream",
            side_effect=lambda fn, *args: fn(*args),
        ) as run_eager,
        patch("torchtitan.training_engine.cuda_graphs_supported", return_value=True),
    ):
        TrainingEngine._initialize_forward_backward(engine)

        # Calls remain eager until two complete optimizer steps have finished.
        for _ in range(3):
            torch.testing.assert_close(
                engine._run_forward_backward([(), ()], torch.tensor(0)).loss,
                torch.tensor(1.0),
            )
        engine._num_optimizer_steps_since_cuda_graph_init = 1
        for _ in range(2):
            torch.testing.assert_close(
                engine._run_forward_backward([(), ()], torch.tensor(0)).loss,
                torch.tensor(1.0),
            )

        engine._num_optimizer_steps_since_cuda_graph_init = 2
        torch.testing.assert_close(
            engine._run_forward_backward([(), ()], torch.tensor(0)).loss,
            torch.tensor(2.0),
        )

    wrap.assert_called_once()
    wrapped_forward_backward = wrap.call_args.args[0]
    assert isinstance(wrapped_forward_backward, partial)
    assert wrapped_forward_backward.func is eager_forward_backward
    assert wrapped_forward_backward.keywords == {"defer_fsdp_gradient_reduction": True}
    assert run_eager.call_count == 5
    assert eager_forward_backward.call_count == 5
    assert all(
        call.kwargs["defer_fsdp_gradient_reduction"] is True
        for call in eager_forward_backward.call_args_list
    )
    cuda_graph_forward_backward.assert_called_once()


def test_training_engine_skips_gradient_accumulation_graph_when_unsupported() -> None:
    eager_forward_backward = MagicMock(
        return_value=ForwardBackwardResult(torch.tensor(1.0), [])
    )
    engine = cast(
        TrainingEngine,
        SimpleNamespace(
            config=SimpleNamespace(
                sdc_replayer=None,
                debug=SimpleNamespace(spmd_typechecking=False),
                training=SimpleNamespace(disable_cuda_graphs=False),
                parallelism=SimpleNamespace(
                    enable_sequence_parallel=False,
                    fsdp_defer_gradient_reduction=False,
                ),
            ),
            parallel_dims=SimpleNamespace(pp_enabled=False),
            _forward_backward_body=eager_forward_backward,
        ),
    )

    with (
        patch("torchtitan.training_engine.wrap_with_cuda_graph") as wrap,
        patch("torchtitan.training_engine.cuda_graphs_supported", return_value=False),
        patch("torchtitan.training_engine.run_eager_on_cuda_graph_stream") as run_eager,
    ):
        TrainingEngine._initialize_forward_backward(engine)
        torch.testing.assert_close(
            engine._run_forward_backward([], torch.tensor(0)).loss,
            torch.tensor(1.0),
        )

    eager_forward_backward.assert_called_once()
    wrap.assert_not_called()
    run_eager.assert_not_called()


def test_trainer_accumulates_reused_cuda_graph_losses():
    graph_loss = torch.tensor(0.0)
    loss_values = iter((1.0, 2.0, 3.0, 4.0, 5.0, 6.0))

    def forward_backward(*, microbatch_groups, global_valid_tokens):
        assert len(microbatch_groups) == 3
        torch.testing.assert_close(global_valid_tokens, torch.tensor(3))
        graph_loss.fill_(sum(next(loss_values) for _ in microbatch_groups))
        return ForwardBackwardResult(graph_loss, [])

    metrics_processor = SimpleNamespace(
        should_log=MagicMock(return_value=True),
        log=MagicMock(),
    )
    trainer = cast(
        TrainingEngine,
        SimpleNamespace(
            config=SimpleNamespace(
                training=SimpleNamespace(
                    disable_cuda_graphs=False,
                    max_norm=1.0,
                ),
            ),
            optimizers=MagicMock(),
            ema=None,
            lr_schedulers=SimpleNamespace(
                get_metrics=MagicMock(return_value={}),
                step=MagicMock(),
            ),
            parallel_dims=SimpleNamespace(
                dp_enabled=False,
                pp_enabled=False,
                dp_cp_enabled=False,
                ep_enabled=False,
                dp_replicate_enabled=False,
                get_optional_mesh=lambda name: None,
            ),
            gradient_accumulation_steps=3,
            num_pp_microbatches=1,
            device=torch.device("cpu"),
            forward_backward=forward_backward,
            sdc_replayer=None,
            model_parts=[],
            checkpointer=SimpleNamespace(maybe_wait_for_staging=MagicMock()),
            metrics_processor=metrics_processor,
            num_completed_steps=0,
            ntokens_seen=3,
        ),
    )
    data_iterator = iter([_batch() for _ in range(3)])

    with patch(
        "torchtitan.training_engine.dist_utils.clip_grad_norm_",
        return_value=torch.tensor(4.0),
    ):
        Trainer.train_step(_training_loop(trainer), data_iterator)

    metrics_processor.log.assert_called_once_with(
        1,
        6.0,
        6.0,
        4.0,
        extra_metrics={"n_tokens_seen": 3},
    )
    assert trainer.num_completed_steps == 1

    metrics_processor.should_log.return_value = False
    metrics_processor.log.reset_mock()
    with patch(
        "torchtitan.training_engine.dist_utils.clip_grad_norm_",
        return_value=torch.tensor(4.0),
    ):
        Trainer.train_step(
            _training_loop(trainer),
            data_iterator=iter([_batch() for _ in range(3)]),
        )

    metrics_processor.log.assert_not_called()
    assert trainer.num_completed_steps == 2


def test_engine_replay_checks_whole_accumulation() -> None:
    run_forward_backward = MagicMock(
        return_value=ForwardBackwardResult(torch.tensor(1.0), [])
    )
    replayer = SimpleNamespace(
        run_fwd_bwd=MagicMock(side_effect=lambda fn, **kwargs: fn()),
    )
    engine = object.__new__(TrainingEngine)
    engine.config = SimpleNamespace(
        training=SimpleNamespace(disable_cuda_graphs=True),
        parallelism=SimpleNamespace(
            fsdp_defer_gradient_reduction=False,
            fsdp_reshard_after_forward="default",
        ),
    )
    engine.parallel_dims = SimpleNamespace(fsdp_enabled=False)
    engine.device = torch.device("cpu")
    engine.gc_handler = SimpleNamespace(run=MagicMock())
    engine.optimizers = SimpleNamespace(zero_grad=MagicMock())
    engine.sdc_replayer = replayer
    engine.num_completed_steps = 0
    engine._preprocess_microbatch_groups = MagicMock(
        side_effect=lambda groups: [(group[0].labels,) for group in groups]
    )
    engine._run_forward_backward = run_forward_backward

    TrainingEngine.forward_backward(
        engine,
        microbatch_groups=[[_batch()], [_batch()]],
        global_valid_tokens=2,
    )

    replayer.run_fwd_bwd.assert_called_once()
    assert replayer.run_fwd_bwd.call_args.kwargs["step"] == 1
    assert callable(replayer.run_fwd_bwd.call_args.kwargs["get_loss"])
    run_forward_backward.assert_called_once()


def test_replay_failure_propagates_from_engine():
    mismatch = SDCReplayMismatch(
        step=1,
        local_step=1,
        replay=1,
        rank=0,
        signature_mismatch="loss",
    )
    engine = object.__new__(TrainingEngine)
    engine.config = SimpleNamespace(
        training=SimpleNamespace(disable_cuda_graphs=True),
        parallelism=SimpleNamespace(
            fsdp_defer_gradient_reduction=False,
            fsdp_reshard_after_forward="default",
        ),
    )
    engine.parallel_dims = SimpleNamespace(fsdp_enabled=False)
    engine.device = torch.device("cpu")
    engine.gc_handler = SimpleNamespace(run=MagicMock())
    engine.optimizers = SimpleNamespace(zero_grad=MagicMock())
    engine.sdc_replayer = SimpleNamespace(run_fwd_bwd=MagicMock(side_effect=mismatch))
    engine.num_completed_steps = 0
    engine._preprocess_microbatch_groups = MagicMock(return_value=[("input",)])
    engine._run_forward_backward = MagicMock()

    with pytest.raises(SDCReplayMismatch):
        TrainingEngine.forward_backward(
            engine,
            microbatch_groups=[[_batch()]],
            global_valid_tokens=torch.tensor(1),
        )


def test_loading_checkpoint_rearms_replay_schedule():
    replayer = SimpleNamespace(reset_schedule=MagicMock())
    trainer = cast(TrainingEngine, SimpleNamespace(sdc_replayer=replayer))

    TrainingEngine.load_state_dict(trainer, {"step": 12, "ntokens_seen": 34})

    assert trainer.num_completed_steps == 12
    assert trainer.ntokens_seen == 34
    replayer.reset_schedule.assert_called_once_with()

    disabled = cast(TrainingEngine, SimpleNamespace(sdc_replayer=None))
    TrainingEngine.load_state_dict(disabled, {"step": 1, "ntokens_seen": 2})
    assert disabled.num_completed_steps == 1


def test_initialize_preserves_phase_order():
    events = []
    model_mem_stats = object()
    engine = cast(
        TrainingEngine,
        SimpleNamespace(
            device_memory_monitor=SimpleNamespace(
                get_peak_stats=lambda: (
                    events.append("model_memory"),
                    model_mem_stats,
                )[1]
            ),
            _initialize_model=MagicMock(
                side_effect=lambda *args, **kwargs: events.append("model")
            ),
            _initialize_optimizer=MagicMock(
                side_effect=lambda *args, **kwargs: events.append("optimizer")
            ),
            _initialize_checkpointer=MagicMock(
                side_effect=lambda *args, **kwargs: events.append("checkpointer")
            ),
            _initialize_forward_backward=MagicMock(
                side_effect=lambda: events.append("forward_backward")
            ),
            state_dict_adapter=None,
        ),
    )
    TrainingEngine.initialize(
        engine,
        compile_config=None,
        hf_assets_path="",
        create_seed_checkpoint=True,
    )

    assert events == [
        "model",
        "model_memory",
        "optimizer",
        "checkpointer",
        "forward_backward",
    ]
    assert engine.model_device_mem_stats is model_mem_stats
    engine._initialize_model.assert_called_once_with(
        compile_config=None,
        hf_assets_path="",
        create_seed_checkpoint=True,
    )
    engine._initialize_optimizer.assert_called_once_with()
    engine._initialize_checkpointer.assert_called_once_with(
        dataloader=None,
        sd_adapter=engine.state_dict_adapter,
    )
    engine._initialize_forward_backward.assert_called_once_with()


def test_compute_training_performance_metrics():
    metrics = compute_training_performance_metrics(
        num_tokens=20,
        elapsed_time=2.0,
        non_data_parallel_size=2,
        num_flops_per_token=200,
        gpu_peak_flops=1000,
        has_quantization=False,
    )

    assert metrics == {
        "tokens_per_second": 5.0,
        "tflops": 1e-9,
        "mfu_percent": 100.0,
    }


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
        patch("torchtitan.distributed.cuda_graph.utils.device_type", device_type),
        patch("torch.cuda.is_available", return_value=cuda_available),
        patch.object(torch.version, "hip", hip_version),
        patch("torchtitan.distributed.cuda_graph.logger.warning") as warning,
    ):
        runner = wrap_with_cuda_graph(fwd_bwd)

    assert runner is fwd_bwd
    warning.assert_called_once()


def test_cuda_graph_accumulation_requires_deferred_gradient_reduction() -> None:
    cuda_graph_forward_backward = MagicMock()
    engine = cast(
        TrainingEngine,
        SimpleNamespace(
            parallel_dims=SimpleNamespace(pp_enabled=False),
            config=SimpleNamespace(
                training=SimpleNamespace(disable_cuda_graphs=False),
                sdc_replayer=None,
                parallelism=SimpleNamespace(
                    fsdp_defer_gradient_reduction=False,
                    fsdp_reshard_after_forward="never",
                ),
            ),
            _forward_backward_body=MagicMock(),
        ),
    )

    with (
        patch(
            "torchtitan.training_engine.wrap_with_cuda_graph",
            return_value=cuda_graph_forward_backward,
        ),
        patch("torchtitan.training_engine.cuda_graphs_supported", return_value=True),
    ):
        TrainingEngine._initialize_forward_backward(engine)
        with pytest.raises(ValueError, match="fsdp_defer_gradient_reduction=True"):
            engine._run_forward_backward(
                [(), ()],
                torch.tensor(2),
            )
        engine._num_optimizer_steps_since_cuda_graph_init = 2
        engine._run_forward_backward([()], torch.tensor(1))

    cuda_graph_forward_backward.assert_called_once_with([()], torch.tensor(1))


@pytest.mark.parametrize("configured_defer", [False, True])
def test_initialize_forward_backward_uses_eager_fsdp_reduction_config(
    configured_defer: bool,
) -> None:
    forward_backward_body = MagicMock(
        return_value=ForwardBackwardResult(torch.tensor(1.0), [])
    )
    engine = cast(
        TrainingEngine,
        SimpleNamespace(
            config=SimpleNamespace(
                training=SimpleNamespace(disable_cuda_graphs=True),
                parallelism=SimpleNamespace(
                    fsdp_defer_gradient_reduction=configured_defer,
                    fsdp_reshard_after_forward="default",
                ),
                sdc_replayer=None,
            ),
            parallel_dims=SimpleNamespace(
                fsdp_enabled=True,
                pp_enabled=False,
            ),
            _forward_backward_body=forward_backward_body,
        ),
    )

    TrainingEngine._initialize_forward_backward(engine)
    engine._run_forward_backward(
        [(), ()],
        torch.tensor(2),
    )

    assert (
        forward_backward_body.call_args.kwargs["defer_fsdp_gradient_reduction"]
        is configured_defer
    )


class _RecordingFSDPPart:
    def __init__(self) -> None:
        self.requires_all_reduce_calls: list[bool] = []
        self.is_last_backward_calls: list[bool] = []
        self.reshard_after_backward_calls: list[bool] = []
        self.requires_gradient_sync_calls: list[bool] = []

    def set_requires_all_reduce(self, flag: bool, *, recurse: bool = True) -> None:
        assert recurse is True
        self.requires_all_reduce_calls.append(flag)

    def set_is_last_backward(self, flag: bool) -> None:
        self.is_last_backward_calls.append(flag)

    def set_reshard_after_backward(self, flag: bool) -> None:
        self.reshard_after_backward_calls.append(flag)

    def set_requires_gradient_sync(self, flag: bool, *, recurse: bool = True) -> None:
        assert recurse is True
        self.requires_gradient_sync_calls.append(flag)

    def parameters(self):
        return iter(())

    def preprocess_inputs(self, input_dict, **kwargs):
        return input_dict["input"], input_dict["labels"], {}


def _run_forward_backward_recording_all_reduce(
    *,
    dp_replicate_enabled: bool,
    gradient_accumulation_steps: int,
    defer_fsdp_gradient_reduction: bool = False,
) -> list[bool]:
    part = _RecordingFSDPPart()
    engine = cast(
        TrainingEngine,
        SimpleNamespace(
            parallel_dims=SimpleNamespace(
                pp_enabled=False,
                dp_replicate_enabled=dp_replicate_enabled,
                fsdp_enabled=True,
            ),
            model_parts=[part],
            _non_pp_forward_backward_microbatch=MagicMock(
                return_value=torch.tensor(1.0)
            ),
        ),
    )
    TrainingEngine._forward_backward_body(
        engine,
        [("input", "labels", {}, {})] * gradient_accumulation_steps,
        torch.tensor(gradient_accumulation_steps),
        defer_fsdp_gradient_reduction=defer_fsdp_gradient_reduction,
    )
    return part.requires_all_reduce_calls


@pytest.mark.parametrize("defer_fsdp_gradient_reduction", [False, True])
def test_hsdp_skips_replicate_all_reduce_until_last_accum_group(
    defer_fsdp_gradient_reduction: bool,
) -> None:
    flags = _run_forward_backward_recording_all_reduce(
        dp_replicate_enabled=True,
        gradient_accumulation_steps=3,
        defer_fsdp_gradient_reduction=defer_fsdp_gradient_reduction,
    )
    assert flags == [False, False, True]


def test_hsdp_keeps_all_reduce_on_single_accum_group() -> None:
    flags = _run_forward_backward_recording_all_reduce(
        dp_replicate_enabled=True,
        gradient_accumulation_steps=1,
    )
    assert flags == [True]


@pytest.mark.parametrize("defer_fsdp_gradient_reduction", [False, True])
def test_pp_hsdp_skips_replicate_all_reduce_until_last_accum_group(
    defer_fsdp_gradient_reduction: bool,
) -> None:
    model_parts = [_RecordingFSDPPart(), _RecordingFSDPPart()]
    engine = cast(
        TrainingEngine,
        SimpleNamespace(
            parallel_dims=SimpleNamespace(
                pp_enabled=True,
                dp_replicate_enabled=True,
            ),
            model_parts=model_parts,
            _pp_forward_backward_microbatch_group=MagicMock(
                side_effect=(torch.tensor(1.0), torch.tensor(2.0))
            ),
        ),
    )

    TrainingEngine._forward_backward_body(
        engine,
        [(None, [{}], None)] * 2,
        torch.tensor(2),
        defer_fsdp_gradient_reduction=defer_fsdp_gradient_reduction,
    )

    for model_part in model_parts:
        assert model_part.requires_all_reduce_calls == [False, True]


def test_pure_fsdp_does_not_toggle_requires_all_reduce():
    flags = _run_forward_backward_recording_all_reduce(
        dp_replicate_enabled=False,
        gradient_accumulation_steps=3,
    )
    assert flags == []


@pytest.mark.parametrize("defer_fsdp_gradient_reduction", [False, True])
def test_fsdp_gradient_accumulation_reduction_policy(
    defer_fsdp_gradient_reduction: bool,
) -> None:
    fsdp_root = _RecordingFSDPPart()
    engine = cast(
        TrainingEngine,
        SimpleNamespace(
            parallel_dims=SimpleNamespace(
                pp_enabled=False,
                dp_replicate_enabled=False,
                fsdp_enabled=True,
            ),
            model_parts=[fsdp_root],
            _non_pp_forward_backward_microbatch=MagicMock(
                side_effect=(torch.tensor(1.0), torch.tensor(2.0))
            ),
        ),
    )

    result = TrainingEngine._forward_backward_body(
        engine,
        [("input", "labels", {}, {})] * 2,
        torch.tensor(2),
        defer_fsdp_gradient_reduction=defer_fsdp_gradient_reduction,
    )

    torch.testing.assert_close(result.loss, torch.tensor(3.0))
    assert result.loss_metrics == [{}, {}]
    if defer_fsdp_gradient_reduction:
        assert fsdp_root.is_last_backward_calls == [False, True]
        assert fsdp_root.reshard_after_backward_calls == [False, True]
        assert fsdp_root.requires_gradient_sync_calls == [False, True]
    else:
        assert fsdp_root.is_last_backward_calls == []
        assert fsdp_root.reshard_after_backward_calls == []
        assert fsdp_root.requires_gradient_sync_calls == []


@pytest.mark.parametrize(
    ("defer_fsdp_gradient_reduction", "expected_finalize_gradients"),
    [(False, [True, True]), (True, [False, True])],
)
def test_pp_gradient_accumulation_finalization_policy(
    defer_fsdp_gradient_reduction: bool,
    expected_finalize_gradients: list[bool],
) -> None:
    pp_forward_backward = MagicMock(side_effect=(torch.tensor(1.0), torch.tensor(2.0)))
    engine = cast(
        TrainingEngine,
        SimpleNamespace(
            parallel_dims=SimpleNamespace(
                pp_enabled=True,
                dp_replicate_enabled=False,
                fsdp_enabled=True,
            ),
            model_parts=[],
            _pp_forward_backward_microbatch_group=pp_forward_backward,
        ),
    )

    result = TrainingEngine._forward_backward_body(
        engine,
        [(None, [{}], None)] * 2,
        torch.tensor(2),
        defer_fsdp_gradient_reduction=defer_fsdp_gradient_reduction,
    )

    torch.testing.assert_close(result.loss, torch.tensor(3.0))
    assert [
        call.kwargs["finalize_gradients"] for call in pp_forward_backward.call_args_list
    ] == expected_finalize_gradients
