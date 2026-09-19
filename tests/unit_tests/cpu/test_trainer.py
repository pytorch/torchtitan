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
from torchtitan.components.data.types import (
    TokenizedTrainingMicrobatch,
    TrainingMicrobatch,
)
from torchtitan.distributed.cuda_graph import wrap_with_cuda_graph
from torchtitan.observability.metrics import compute_training_performance_metrics
from torchtitan.observability.sdc_replayer import SDCReplayMismatch
from torchtitan.trainer import Trainer
from torchtitan.training_engine import TrainingEngine


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


def _bind_pp_forward_backward_body(trainer: TrainingEngine) -> None:
    trainer.forward_backward_body_fn = (
        lambda **kwargs: TrainingEngine._pp_forward_backward_body(trainer, **kwargs)
    )


def _training_loop(trainer: TrainingEngine) -> SimpleNamespace:
    if not hasattr(trainer, "_num_optimizer_steps_since_cuda_graph_init"):
        trainer._num_optimizer_steps_since_cuda_graph_init = 0
    if not hasattr(trainer, "loss_is_finite"):
        trainer.loss_is_finite = torch.ones((), dtype=torch.int32)
    if not hasattr(trainer, "prepare_step"):
        trainer.prepare_step = lambda global_valid_tokens, **kwargs: (
            TrainingEngine.prepare_step(trainer, global_valid_tokens, **kwargs)
        )
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


def test_pp_forward_backward_microbatch_returns_sentinel_without_last_stage():
    sentinel = torch.full((1,), -1.0)
    trainer = cast(
        TrainingEngine,
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
            parallel_dims=SimpleNamespace(
                pp_enabled=True, cp=1, dp_replicate_enabled=False
            ),
            max_num_documents=None,
            preprocess_inputs_kwargs={},
            config=SimpleNamespace(
                parallelism="PARA",
                dataloader=SimpleNamespace(max_num_documents=None),
                training=SimpleNamespace(
                    disable_cuda_graphs=True,
                    max_context_length=2048,
                    num_tokens_per_microbatch_per_dp_rank=1,
                ),
            ),
            ntokens_seen=0,
            device=torch.device("cpu"),
            sdc_replayer=None,
            num_accumulation_steps=1,
            num_completed_steps=0,
            _pp_loss_sentinel_on_non_last_stage=sentinel,
        ),
    )
    _bind_pp_forward_backward_body(trainer)

    loss = TrainingEngine.forward_backward_microbatch(
        trainer,
        microbatch_group=[
            _dict_microbatch({"input": torch.ones(1), "labels": torch.ones(1)})
        ],
        global_valid_tokens=torch.tensor(1),
    )

    torch.testing.assert_close(loss, sentinel)


def test_pp_forward_backward_microbatch_releases_consumed_loss_graphs() -> None:
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
            parallel_dims=SimpleNamespace(
                pp_enabled=True, cp=1, dp_replicate_enabled=False
            ),
            max_num_documents=None,
            preprocess_inputs_kwargs={},
            config=SimpleNamespace(
                parallelism="PARA",
                training=SimpleNamespace(
                    disable_cuda_graphs=True,
                    max_context_length=2048,
                    num_tokens_per_microbatch_per_dp_rank=1,
                ),
            ),
            ntokens_seen=0,
            device=torch.device("cpu"),
            sdc_replayer=None,
            num_accumulation_steps=1,
            num_completed_steps=0,
        ),
    )
    _bind_pp_forward_backward_body(trainer)

    reporting_loss = TrainingEngine.forward_backward_microbatch(
        trainer,
        microbatch_group=[
            _dict_microbatch({"input": torch.ones(1), "labels": torch.ones(1)}),
            _dict_microbatch({"input": torch.ones(1), "labels": torch.ones(1)}),
        ],
        global_valid_tokens=torch.tensor(2),
    )

    torch.testing.assert_close(reporting_loss, torch.tensor(5.0))
    torch.testing.assert_close(torch.stack(gradients), torch.tensor([2.0, 4.0]))
    assert not reporting_loss.requires_grad
    assert reporting_loss.grad_fn is None
    assert loss_containers == [[]]
    assert all(reference() is None for reference in loss_refs)
    assert all(reference() is None for reference in activation_refs)


def test_pp_forward_backward_microbatch_prepares_structured_inputs() -> None:
    forward_backward_body_fn = MagicMock(return_value=torch.tensor(0.0))

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
            parallel_dims=SimpleNamespace(
                pp_enabled=True, cp=1, dp_replicate_enabled=False
            ),
            max_num_documents=4,
            preprocess_inputs_kwargs={},
            config=SimpleNamespace(
                parallelism="PARA",
                training=SimpleNamespace(
                    disable_cuda_graphs=True,
                    max_context_length=2048,
                    num_tokens_per_microbatch_per_dp_rank=1,
                ),
            ),
            ntokens_seen=0,
            device=torch.device("cpu"),
            forward_backward_body_fn=forward_backward_body_fn,
            sdc_replayer=None,
            num_accumulation_steps=1,
            num_completed_steps=0,
        ),
    )
    global_valid_tokens = torch.tensor(2)

    result = TrainingEngine.forward_backward_microbatch(
        trainer,
        microbatch_group=[
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
        ],
        global_valid_tokens=global_valid_tokens,
    )

    torch.testing.assert_close(result, torch.tensor(0.0))
    passed_kwargs = forward_backward_body_fn.call_args.kwargs
    arg_mbs = passed_kwargs["inputs"]
    kwarg_mbs = passed_kwargs["model_kwargs"]
    target_mbs = passed_kwargs["labels"]
    passed_loss_kwargs = passed_kwargs["loss_kwargs"]
    torch.testing.assert_close(arg_mbs[0][0], torch.tensor(2))
    torch.testing.assert_close(arg_mbs[1][0], torch.tensor(3))
    torch.testing.assert_close(kwarg_mbs[0]["positions"], torch.tensor(13))
    torch.testing.assert_close(kwarg_mbs[1]["positions"], torch.tensor(23))
    torch.testing.assert_close(target_mbs[0], torch.tensor([5]))
    torch.testing.assert_close(target_mbs[1], torch.tensor([6]))
    assert set(passed_loss_kwargs) == {"global_valid_tokens"}
    assert passed_loss_kwargs["global_valid_tokens"] is global_valid_tokens
    assert trainer.ntokens_seen == 2


def test_pp_forward_backward_microbatch_rejects_batch_loss_kwargs() -> None:
    trainer = cast(
        TrainingEngine,
        SimpleNamespace(
            parallel_dims=SimpleNamespace(pp_enabled=True, dp_replicate_enabled=False),
            device=torch.device("cpu"),
            sdc_replayer=None,
            num_accumulation_steps=1,
            num_completed_steps=0,
        ),
    )
    microbatch = _dict_microbatch(
        {"labels": torch.tensor([1])},
        {"advantages": torch.tensor([0.1])},
    )

    with pytest.raises(ValueError, match="pipeline parallelism"):
        TrainingEngine.forward_backward_microbatch(
            trainer,
            microbatch_group=[microbatch],
            global_valid_tokens=torch.tensor(1),
        )


def test_forward_backward_microbatch_accumulates_tokens_and_forwards_triple():
    captured: dict[str, Any] = {}

    class _FakeModel:
        def preprocess_inputs(self, input_dict, **kw):
            captured["preprocess_kwargs"] = kw
            return ("INPUTS", torch.ones(7), {"positions": 1})

    def forward_backward_body_fn(*, inputs, labels, model_kwargs, loss_kwargs):
        captured["fwd_bwd_args"] = (inputs, labels, model_kwargs)
        torch.testing.assert_close(loss_kwargs["global_valid_tokens"], torch.tensor(1))
        torch.testing.assert_close(loss_kwargs["advantages"], torch.tensor([0.1]))
        assert loss_kwargs["reduction"] == "sum"
        return torch.tensor(0.0, requires_grad=True)

    fake = SimpleNamespace(
        model_parts=[_FakeModel()],
        max_num_documents=4,
        parallel_dims=SimpleNamespace(
            pp_enabled=False, cp=1, dp_replicate_enabled=False
        ),
        config=SimpleNamespace(
            parallelism="PARA",
            dataloader=SimpleNamespace(max_num_documents=4),
            training=SimpleNamespace(
                disable_cuda_graphs=True,
                max_context_length=2048,
                num_tokens_per_microbatch_per_dp_rank=7,
            ),
        ),
        preprocess_inputs_kwargs={"processor": "VALUE"},
        ntokens_seen=100,
        device=torch.device("cpu"),
        forward_backward_body_fn=forward_backward_body_fn,
        sdc_replayer=None,
        num_accumulation_steps=1,
        num_completed_steps=0,
    )

    microbatch = _dict_microbatch(
        {"input": 0, "labels": torch.zeros(1)},
        {"advantages": torch.tensor([0.1]), "reduction": "sum"},
    )
    detached_loss = TrainingEngine.forward_backward_microbatch(
        fake,  # pyrefly: ignore[bad-argument-type]
        microbatch_group=[microbatch],
        global_valid_tokens=torch.tensor(1),
    )

    inputs, labels, extra = captured["fwd_bwd_args"]
    assert inputs == "INPUTS"
    assert extra == {"positions": 1}
    assert labels.numel() == 7
    assert fake.ntokens_seen == 107
    assert not detached_loss.requires_grad
    assert fake.loss_is_finite.item() == 1
    assert microbatch.to_input_dict_calls == [(fake.device, True)]
    assert microbatch.to_loss_kwargs_calls == [(fake.device, True)]
    assert captured["preprocess_kwargs"] == {
        "parallel_dims": fake.parallel_dims,
        "parallelism": "PARA",
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


def test_training_engine_owns_cuda_graph_warmup() -> None:
    eager_forward_backward_body = MagicMock(return_value=torch.tensor(1.0))
    cuda_graph_forward_backward_body = MagicMock(return_value=torch.tensor(2.0))
    engine = cast(
        TrainingEngine,
        SimpleNamespace(
            config=SimpleNamespace(
                sdc_replayer=None,
                debug=SimpleNamespace(spmd_typechecking=False),
                training=SimpleNamespace(disable_cuda_graphs=False),
            ),
            parallel_dims=SimpleNamespace(pp_enabled=False),
            _non_pp_forward_backward_body=eager_forward_backward_body,
            _num_optimizer_steps_since_cuda_graph_init=100,
        ),
    )

    with (
        patch(
            "torchtitan.training_engine.dist_utils.get_spmd_context",
            return_value=MagicMock(),
        ),
        patch(
            "torchtitan.training_engine.wrap_with_cuda_graph",
            return_value=cuda_graph_forward_backward_body,
        ) as wrap,
        patch(
            "torchtitan.training_engine.run_eager_on_cuda_graph_stream",
            side_effect=lambda fn, **kwargs: fn(**kwargs),
        ) as run_eager,
    ):
        TrainingEngine._initialize_forward_backward(engine)
        assert engine._num_optimizer_steps_since_cuda_graph_init == 0

        # Any number of forward/backward calls remains eager until two complete
        # optimizer steps have finished.
        for _ in range(3):
            torch.testing.assert_close(
                engine.forward_backward_body_fn(value=torch.tensor(0)),
                torch.tensor(1.0),
            )
        engine._num_optimizer_steps_since_cuda_graph_init = 1
        for _ in range(2):
            torch.testing.assert_close(
                engine.forward_backward_body_fn(value=torch.tensor(0)),
                torch.tensor(1.0),
            )

        engine._num_optimizer_steps_since_cuda_graph_init = 2
        torch.testing.assert_close(
            engine.forward_backward_body_fn(value=torch.tensor(0)), torch.tensor(2.0)
        )

    wrap.assert_called_once_with(eager_forward_backward_body)
    assert run_eager.call_count == 5
    assert eager_forward_backward_body.call_count == 5
    cuda_graph_forward_backward_body.assert_called_once()


def test_training_engine_skips_cuda_graph_warmup_when_unsupported() -> None:
    eager_forward_backward_body = MagicMock(return_value=torch.tensor(1.0))
    engine = cast(
        TrainingEngine,
        SimpleNamespace(
            config=SimpleNamespace(
                sdc_replayer=None,
                debug=SimpleNamespace(spmd_typechecking=False),
                training=SimpleNamespace(disable_cuda_graphs=False),
            ),
            parallel_dims=SimpleNamespace(pp_enabled=False),
            _non_pp_forward_backward_body=eager_forward_backward_body,
        ),
    )

    with (
        patch(
            "torchtitan.training_engine.dist_utils.get_spmd_context",
            return_value=MagicMock(),
        ),
        patch(
            "torchtitan.training_engine.wrap_with_cuda_graph",
            side_effect=lambda fn: fn,
        ),
        patch("torchtitan.training_engine.run_eager_on_cuda_graph_stream") as run_eager,
    ):
        TrainingEngine._initialize_forward_backward(engine)
        torch.testing.assert_close(
            engine.forward_backward_body_fn(value=torch.tensor(0)), torch.tensor(1.0)
        )

    eager_forward_backward_body.assert_called_once()
    run_eager.assert_not_called()


def test_trainer_accumulates_reused_cuda_graph_losses():
    graph_loss = torch.tensor(0.0)
    loss_values = iter((1.0, 2.0, 3.0, 4.0, 5.0, 6.0))

    def forward_backward_microbatch(**kwargs):
        graph_loss.fill_(next(loss_values))
        return graph_loss

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
            forward_backward_microbatch=forward_backward_microbatch,
            sdc_replayer=None,
            model_parts=[],
            checkpointer=SimpleNamespace(maybe_wait_for_staging=MagicMock()),
            metrics_processor=metrics_processor,
            num_completed_steps=0,
            ntokens_seen=3,
            gc_handler=SimpleNamespace(run=MagicMock()),
            _deferred_cuda_graph_options=None,
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


def test_engine_replay_checks_only_first_forward_backward():
    forward_backward_body_fn = MagicMock(return_value=torch.tensor(1.0))
    replayer = SimpleNamespace(
        run_fwd_bwd=MagicMock(side_effect=lambda fn, **kwargs: fn()),
    )
    engine = cast(
        TrainingEngine,
        SimpleNamespace(
            config=SimpleNamespace(
                training=SimpleNamespace(
                    disable_cuda_graphs=True,
                    max_context_length=1,
                    num_tokens_per_microbatch_per_dp_rank=1,
                ),
                parallelism="PARA",
            ),
            parallel_dims=SimpleNamespace(
                pp_enabled=False,
                dp_replicate_enabled=False,
                cp=1,
            ),
            device=torch.device("cpu"),
            model_parts=[
                SimpleNamespace(
                    preprocess_inputs=lambda input_dict, **kwargs: (
                        input_dict["input"],
                        input_dict["labels"],
                        {},
                    )
                )
            ],
            max_num_documents=None,
            preprocess_inputs_kwargs={},
            forward_backward_body_fn=forward_backward_body_fn,
            sdc_replayer=replayer,
            num_accumulation_steps=2,
            num_completed_steps=0,
            ntokens_seen=0,
        ),
    )

    for accumulation_index in range(2):
        TrainingEngine.forward_backward_microbatch(
            engine,
            microbatch_group=[_batch()],
            global_valid_tokens=torch.tensor(2),
            accumulation_index=accumulation_index,
        )

    replayer.run_fwd_bwd.assert_called_once()
    assert replayer.run_fwd_bwd.call_args.kwargs == {"step": 1}
    assert forward_backward_body_fn.call_count == 2


def test_replay_failure_propagates_from_engine():
    mismatch = SDCReplayMismatch(
        step=1,
        local_step=1,
        replay=1,
        rank=0,
        signature_mismatch="loss",
    )
    engine = cast(
        TrainingEngine,
        SimpleNamespace(
            config=SimpleNamespace(
                training=SimpleNamespace(
                    disable_cuda_graphs=True,
                    num_tokens_per_microbatch_per_dp_rank=1,
                )
            ),
            parallel_dims=SimpleNamespace(
                pp_enabled=False,
                dp_replicate_enabled=False,
            ),
            device=torch.device("cpu"),
            model_parts=[],
            sdc_replayer=SimpleNamespace(run_fwd_bwd=MagicMock(side_effect=mismatch)),
            num_accumulation_steps=1,
            num_completed_steps=0,
        ),
    )

    with pytest.raises(SDCReplayMismatch):
        TrainingEngine.forward_backward_microbatch(
            engine,
            microbatch_group=[_batch()],
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
        ),
    )
    model_spec = MagicMock()
    sd_adapter = MagicMock()

    TrainingEngine.initialize(
        engine,
        model_spec,
        compile_config=None,
        sd_adapter=sd_adapter,
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
        model_spec,
        compile_config=None,
        create_seed_checkpoint=True,
    )
    engine._initialize_optimizer.assert_called_once_with(model_spec)
    engine._initialize_checkpointer.assert_called_once_with(
        dataloader=None,
        sd_adapter=sd_adapter,
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


class _RecordingFSDPPart:
    def __init__(self) -> None:
        self.requires_all_reduce_calls: list[bool] = []

    def set_requires_all_reduce(self, flag: bool, *, recurse: bool = True) -> None:
        assert recurse is True
        self.requires_all_reduce_calls.append(flag)

    def parameters(self):
        return iter(())

    def preprocess_inputs(self, input_dict, **kwargs):
        return input_dict["input"], input_dict["labels"], {}


def _run_forward_backward_recording_all_reduce(
    *,
    dp_replicate_enabled: bool,
    gradient_accumulation_steps: int,
    disable_cuda_graphs: bool,
) -> list[bool]:
    part = _RecordingFSDPPart()
    trainer = cast(
        TrainingEngine,
        SimpleNamespace(
            config=SimpleNamespace(
                parallelism="PARA",
                training=SimpleNamespace(
                    disable_cuda_graphs=disable_cuda_graphs,
                    max_context_length=1,
                    num_tokens_per_microbatch_per_dp_rank=1,
                ),
            ),
            parallel_dims=SimpleNamespace(
                pp_enabled=False,
                dp_replicate_enabled=dp_replicate_enabled,
                cp=1,
            ),
            device=torch.device("cpu"),
            forward_backward_body_fn=MagicMock(return_value=torch.tensor(1.0)),
            sdc_replayer=None,
            model_parts=[part],
            max_num_documents=None,
            preprocess_inputs_kwargs={},
            num_accumulation_steps=gradient_accumulation_steps,
            num_completed_steps=1,
            ntokens_seen=0,
        ),
    )
    for accumulation_index in range(gradient_accumulation_steps):
        TrainingEngine.forward_backward_microbatch(
            trainer,
            microbatch_group=[_batch()],
            global_valid_tokens=torch.tensor(gradient_accumulation_steps),
            accumulation_index=accumulation_index,
        )
    return part.requires_all_reduce_calls


def test_hsdp_skips_replicate_all_reduce_until_last_accum_group():
    flags = _run_forward_backward_recording_all_reduce(
        dp_replicate_enabled=True,
        gradient_accumulation_steps=3,
        disable_cuda_graphs=True,
    )
    assert flags == [False, False, True]


@pytest.mark.parametrize("disable_cuda_graphs", [True, False])
def test_hsdp_keeps_all_reduce_on_single_accum_group(disable_cuda_graphs: bool):
    flags = _run_forward_backward_recording_all_reduce(
        dp_replicate_enabled=True,
        gradient_accumulation_steps=1,
        disable_cuda_graphs=disable_cuda_graphs,
    )
    assert flags == [True]


def test_pure_fsdp_does_not_toggle_requires_all_reduce():
    flags = _run_forward_backward_recording_all_reduce(
        dp_replicate_enabled=False,
        gradient_accumulation_steps=3,
        disable_cuda_graphs=True,
    )
    assert flags == []


def test_hsdp_does_not_toggle_requires_all_reduce_under_cuda_graphs():
    flags = _run_forward_backward_recording_all_reduce(
        dp_replicate_enabled=True,
        gradient_accumulation_steps=3,
        disable_cuda_graphs=False,
    )
    assert flags == []
