# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
import torch
from torchtitan.distributed.accelerator_graph import (
    _ForwardBackwardAcceleratorGraphWrapper,
    _run_eager_on_accelerator_graph_stream,
    accelerator_graph_manager,
    accelerator_graph_teardown,
    AcceleratorGraphWrapper,
    get_accelerator_graph_annotations,
    wrap_with_accelerator_graph,
)


@pytest.fixture(autouse=True)
def reset_accelerator_graph_manager():
    accelerator_graph_teardown()
    yield
    accelerator_graph_teardown()


def test_accelerator_graph_wrapper_uses_configured_warmup_iterations() -> None:
    with (
        patch.object(accelerator_graph_manager, "maybe_initialize"),
        patch.object(accelerator_graph_manager, "register"),
    ):
        wrapper = AcceleratorGraphWrapper(
            lambda value: value,
            (torch.tensor(1),),
            num_warmup_iterations=2,
        )

    assert wrapper._warmup_remaining == 2


def test_accelerator_graph_wrapper_rejects_negative_warmup_iterations() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        AcceleratorGraphWrapper(
            lambda value: value,
            (torch.tensor(1),),
            num_warmup_iterations=-1,
        )


def test_warmup_completion_tracks_all_registered_wrappers() -> None:
    assert accelerator_graph_manager.all_wrappers_warmed_up
    accelerator_graph_manager.capture_allowed = False
    with (
        patch.object(accelerator_graph_manager, "maybe_initialize"),
        patch(
            "torchtitan.distributed.accelerator_graph._run_eager_on_accelerator_graph_stream",
            side_effect=lambda fn, *args: fn(*args),
        ),
    ):
        value = torch.tensor(1)
        train = AcceleratorGraphWrapper(lambda x: x, (value,), num_warmup_iterations=1)
        optimizer = AcceleratorGraphWrapper(
            lambda x: x, (value,), num_warmup_iterations=2
        )
        assert not accelerator_graph_manager.all_wrappers_warmed_up
        train(value)
        optimizer(value)
        assert not accelerator_graph_manager.all_wrappers_warmed_up
        optimizer(value)
        assert accelerator_graph_manager.all_wrappers_warmed_up
        assert not accelerator_graph_manager.capture_allowed

        validation = AcceleratorGraphWrapper(
            lambda x: x, (value,), num_warmup_iterations=1, name="validation"
        )
        assert not accelerator_graph_manager.all_wrappers_warmed_up
        validation(value)
        assert accelerator_graph_manager.all_wrappers_warmed_up


def test_eager_execution_on_accelerator_graph_stream_synchronizes_streams() -> None:
    current_stream = MagicMock()
    graph_stream = MagicMock()
    fn = MagicMock(return_value="output")

    with (
        patch.object(accelerator_graph_manager, "maybe_initialize") as maybe_initialize,
        patch.object(accelerator_graph_manager, "_stream", graph_stream),
        patch("torch.accelerator.current_stream", return_value=current_stream),
    ):
        output = _run_eager_on_accelerator_graph_stream(fn, "arg", keyword="value")

    assert output == "output"
    maybe_initialize.assert_called_once_with()
    graph_stream.wait_stream.assert_called_once_with(current_stream)
    graph_stream.__enter__.assert_called_once_with()
    fn.assert_called_once_with("arg", keyword="value")
    current_stream.wait_stream.assert_called_once_with(graph_stream)


def test_wrap_with_accelerator_graph_captures_first_invocation() -> None:
    graph = MagicMock()
    fn = MagicMock(side_effect=lambda value, *, scale: value * scale)
    prepare = MagicMock()
    with (
        patch("torchtitan.distributed.accelerator_graph.utils.device_type", "cuda"),
        patch(
            "torch.accelerator.current_accelerator",
            return_value=torch.device("cuda"),
        ),
        patch.object(torch.version, "hip", None),
        patch.object(accelerator_graph_manager, "maybe_initialize"),
        patch.object(accelerator_graph_manager, "register"),
        patch.object(accelerator_graph_manager, "_graph_pool", object()),
        patch.object(accelerator_graph_manager, "_stream", MagicMock()),
        patch("torch.accelerator.current_stream", return_value=MagicMock()),
        patch("torch.cuda.CUDAGraph", return_value=graph) as graph_constructor,
        patch("torch.cuda.graph", return_value=nullcontext()),
        patch(
            "torchtitan.distributed.accelerator_graph.cuda_graph.get_kernel_annotations",
            return_value={},
        ),
    ):
        run = wrap_with_accelerator_graph(fn, prepare_for_capture=prepare)
        value = torch.tensor(1.0)
        run(value, scale=2)
        prepare.assert_called_once_with(value, scale=2)
        graph_constructor.assert_called_once()
        graph.replay.assert_called_once()
        run(value, scale=2)
        prepare.assert_called_once()
        assert fn.call_count == 1
        assert graph.replay.call_count == 2


def test_tensor_input_indices_control_replay_copies() -> None:
    static_input = torch.tensor(1)
    excluded_input = torch.tensor(2)
    copied_input = torch.tensor(3)

    with (
        patch.object(accelerator_graph_manager, "maybe_initialize"),
        patch.object(accelerator_graph_manager, "register"),
    ):
        wrapper = AcceleratorGraphWrapper(
            lambda *args: args,
            (static_input, excluded_input, copied_input),
            static_input_indices=(0,),
            tensor_input_indices=[0, 2],
            num_warmup_iterations=0,
        )

    wrapper._args = (static_input, excluded_input, copied_input)
    graph = cast(torch.cuda.CUDAGraph, MagicMock())
    wrapper._graph = graph
    wrapper._output = "output"

    result = wrapper(torch.tensor(4), torch.tensor(5), torch.tensor(6))

    assert result == "output"
    assert static_input.item() == 1
    assert excluded_input.item() == 2
    assert copied_input.item() == 6
    cast(MagicMock, graph.replay).assert_called_once_with()


def test_accelerator_graph_wrapper_collects_annotations() -> None:
    graph = cast(torch.cuda.CUDAGraph, MagicMock())
    annotations = {42: [{"module_fqn": "layers.0"}]}
    graph_pool = object()
    stream = MagicMock()

    with (
        patch("torchtitan.distributed.accelerator_graph.utils.device_type", "cuda"),
        patch.object(accelerator_graph_manager, "maybe_initialize"),
        patch.object(accelerator_graph_manager, "register"),
        patch.object(accelerator_graph_manager, "_graph_pool", graph_pool),
        patch.object(accelerator_graph_manager, "_stream", stream),
        patch.object(accelerator_graph_manager, "all_annotations", {}),
        patch("torch.cuda.CUDAGraph", return_value=graph),
        patch("torch.cuda.graph", return_value=nullcontext()) as cuda_graph,
        patch(
            "torchtitan.distributed.accelerator_graph.cuda_graph.get_kernel_annotations",
            return_value=annotations,
        ),
    ):
        wrapper = AcceleratorGraphWrapper(
            lambda x: x,
            (torch.tensor(1),),
            num_warmup_iterations=0,
        )

        output = wrapper(torch.tensor(2))

        assert output.item() == 2
        assert get_accelerator_graph_annotations() == annotations
        cuda_graph.assert_called_once_with(
            graph,
            pool=graph_pool,
            stream=stream,
            enable_annotations=True,
            annotation_config=None,
            capture_error_mode="thread_local",
        )


@pytest.mark.parametrize("initialized", [False, True])
def test_manager_teardown_resets_capture_state(initialized) -> None:
    wrappers = (MagicMock(), MagicMock()) if initialized else ()
    with (
        patch.object(accelerator_graph_manager, "_initialized", initialized),
        patch.object(accelerator_graph_manager, "_wrappers", list(wrappers)),
        patch.object(accelerator_graph_manager, "all_annotations", {1: ["capture"]}),
        patch("torch.accelerator.synchronize"),
    ):
        accelerator_graph_manager.capture_allowed = False
        accelerator_graph_teardown()
        assert accelerator_graph_manager.capture_allowed
        assert not accelerator_graph_manager._initialized
        assert not get_accelerator_graph_annotations()
        assert not accelerator_graph_manager._wrappers
        for wrapper in wrappers:
            wrapper.teardown.assert_called_once_with()


def test_preparation_and_cache_release_run_only_before_first_capture() -> None:
    events = []

    def capture(*args, **kwargs):
        events.append("capture")
        return nullcontext()

    with (
        patch("torchtitan.distributed.accelerator_graph.utils.device_type", "cuda"),
        patch.object(accelerator_graph_manager, "maybe_initialize"),
        patch.object(accelerator_graph_manager, "register"),
        patch.object(accelerator_graph_manager, "_graph_pool", object()),
        patch.object(accelerator_graph_manager, "_stream", MagicMock()),
        patch.object(accelerator_graph_manager, "log_memory"),
        patch("torch.cuda.CUDAGraph"),
        patch("torch.cuda.graph", side_effect=capture),
        patch(
            "torch.accelerator.empty_cache", side_effect=lambda: events.append("cache")
        ),
        patch(
            "torchtitan.distributed.accelerator_graph._run_eager_on_accelerator_graph_stream",
            side_effect=lambda fn, *args: fn(*args),
        ),
        patch(
            "torchtitan.distributed.accelerator_graph.cuda_graph.get_kernel_annotations",
            return_value={},
        ),
    ):
        wrapper = AcceleratorGraphWrapper(
            lambda x: x,
            (torch.tensor(1),),
            num_warmup_iterations=2,
            prepare_for_capture=lambda x: events.append("prepare"),
            name="eval",
        )
        for _ in range(4):
            wrapper(torch.tensor(1))
        assert events == ["prepare", "cache", "capture"]


def test_capture_gate_does_not_record_gradients() -> None:
    accelerator_graph_manager.capture_allowed = False
    parameter = torch.nn.Parameter(torch.tensor(2.0))

    def body(value):
        loss = parameter * value
        loss.backward()
        return loss.detach()

    with (
        patch("torchtitan.distributed.accelerator_graph.utils.device_type", "cuda"),
        patch.object(accelerator_graph_manager, "maybe_initialize"),
        patch.object(accelerator_graph_manager, "_stream", MagicMock()),
        patch("torch.accelerator.current_stream", return_value=MagicMock()),
        patch("torch.cuda.CUDAGraph") as graph,
    ):
        wrapper = _ForwardBackwardAcceleratorGraphWrapper(
            body,
            (torch.tensor(3.0),),
            parameters=[parameter],
            num_warmup_iterations=2,
            name="body",
        )
        for _ in range(4):
            parameter.grad = None
            assert wrapper(torch.tensor(3.0)).item() == 6.0
        assert wrapper._warmup_remaining == 0
        assert wrapper._gradient_state._captured_parameter_gradients is None
        graph.assert_not_called()
    accelerator_graph_teardown()


def test_structured_wrapper_validates_and_copies_replay_inputs() -> None:
    graph = cast(torch.cuda.CUDAGraph, MagicMock())
    graph_stream = MagicMock()
    current_stream = MagicMock()
    fn = MagicMock(side_effect=lambda batches, *, scale: batches[1]["x"] * scale)

    with (
        patch("torchtitan.distributed.accelerator_graph.utils.device_type", "cuda"),
        patch(
            "torch.accelerator.current_accelerator",
            return_value=torch.device("cuda"),
        ),
        patch.object(torch.version, "hip", None),
        patch.object(accelerator_graph_manager, "maybe_initialize"),
        patch.object(accelerator_graph_manager, "register"),
        patch.object(accelerator_graph_manager, "_graph_pool", object()),
        patch.object(accelerator_graph_manager, "_stream", graph_stream),
        patch("torch.accelerator.current_stream", return_value=current_stream),
        patch("torch.cuda.CUDAGraph", return_value=graph),
        patch("torch.cuda.graph", return_value=nullcontext()),
        patch(
            "torchtitan.distributed.accelerator_graph.cuda_graph.get_kernel_annotations",
            return_value={},
        ),
    ):
        run = wrap_with_accelerator_graph(fn)
        torch.testing.assert_close(
            run(
                [{"x": torch.tensor(4.0)}, {"x": torch.tensor(5.0)}],
                scale=torch.tensor(4.0),
            ),
            torch.tensor(20.0),
        )
        torch.testing.assert_close(
            run(
                [{"x": torch.tensor(6.0)}, {"x": torch.tensor(7.0)}],
                scale=torch.tensor(5.0),
            ),
            torch.tensor(20.0),
        )

        with pytest.raises(ValueError, match="structure must remain constant"):
            run([{"x": torch.tensor(1.0)}], scale=torch.tensor(1.0))
        with pytest.raises(ValueError, match="same shape, dtype, and device"):
            run(
                [{"x": torch.ones(2)}, {"x": torch.tensor(1.0)}],
                scale=torch.tensor(1.0),
            )

    assert fn.call_count == 1
    captured_batches = fn.call_args.args[0]
    torch.testing.assert_close(captured_batches[0]["x"], torch.tensor(6.0))
    torch.testing.assert_close(captured_batches[1]["x"], torch.tensor(7.0))
    torch.testing.assert_close(fn.call_args.kwargs["scale"], torch.tensor(5.0))
    assert cast(MagicMock, graph.replay).call_count == 2


def test_accelerator_graph_wrapper_restores_capture_allocated_gradients() -> None:
    parameter = torch.nn.Parameter(torch.ones(2))
    frozen_parameter = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    captured_gradient = torch.full_like(parameter, 3.0)
    graph = cast(torch.cuda.CUDAGraph, MagicMock())

    def forward_backward(value):
        parameter.grad = captured_gradient
        return value

    with (
        patch("torchtitan.distributed.accelerator_graph.utils.device_type", "cuda"),
        patch.object(accelerator_graph_manager, "maybe_initialize"),
        patch.object(accelerator_graph_manager, "register"),
        patch.object(accelerator_graph_manager, "_graph_pool", object()),
        patch.object(accelerator_graph_manager, "_stream", MagicMock()),
        patch("torch.cuda.CUDAGraph", return_value=graph),
        patch("torch.cuda.graph", return_value=nullcontext()),
        patch(
            "torchtitan.distributed.accelerator_graph.cuda_graph.get_kernel_annotations",
            return_value={},
        ),
    ):
        wrapper = _ForwardBackwardAcceleratorGraphWrapper(
            forward_backward,
            (torch.tensor(1.0),),
            num_warmup_iterations=0,
            parameters=(parameter, parameter, frozen_parameter),
        )

        wrapper(torch.tensor(2.0))
        assert parameter.grad is captured_gradient
        assert frozen_parameter.grad is None

        parameter.grad = None
        wrapper(torch.tensor(3.0))
        assert parameter.grad is captured_gradient

        wrapper.teardown()
        assert parameter.grad is None

    assert cast(MagicMock, graph.replay).call_count == 2


@pytest.mark.parametrize("capture_first", [False, True])
def test_accelerator_graph_wrapper_requires_cleared_owned_gradients(
    capture_first: bool,
) -> None:
    parameter = torch.nn.Parameter(torch.ones(2))
    graph = cast(torch.cuda.CUDAGraph, MagicMock())

    def forward_backward(value):
        parameter.grad = torch.ones_like(parameter)
        return value

    with (
        patch("torchtitan.distributed.accelerator_graph.utils.device_type", "cuda"),
        patch.object(accelerator_graph_manager, "maybe_initialize"),
        patch.object(accelerator_graph_manager, "register"),
        patch.object(accelerator_graph_manager, "_graph_pool", object()),
        patch.object(accelerator_graph_manager, "_stream", MagicMock()),
        patch("torch.cuda.CUDAGraph", return_value=graph),
        patch("torch.cuda.graph", return_value=nullcontext()),
        patch(
            "torchtitan.distributed.accelerator_graph.cuda_graph.get_kernel_annotations",
            return_value={},
        ),
    ):
        wrapper = _ForwardBackwardAcceleratorGraphWrapper(
            forward_backward,
            (torch.tensor(1.0),),
            num_warmup_iterations=0,
            parameters=(parameter,),
        )

        if capture_first:
            wrapper(torch.tensor(2.0))
        parameter.grad = torch.ones_like(parameter)
        with pytest.raises(RuntimeError, match="must be None"):
            wrapper(torch.tensor(3.0))
