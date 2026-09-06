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

from torchtitan.distributed.cudagraph import (
    _manager,
    CUDAGraphWrapper,
    get_cudagraph_annotations,
    wrap_with_cuda_graph,
)


def test_cudagraph_wrapper_uses_configured_warmup_iterations() -> None:
    with (
        patch.object(_manager, "maybe_initialize"),
        patch.object(_manager, "register"),
    ):
        wrapper = CUDAGraphWrapper(
            lambda value: value,
            (torch.tensor(1),),
            num_warmup_iterations=2,
        )

    assert wrapper._warmup_remaining == 2


def test_cudagraph_wrapper_rejects_negative_warmup_iterations() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        CUDAGraphWrapper(
            lambda value: value,
            (torch.tensor(1),),
            num_warmup_iterations=-1,
        )


@pytest.mark.parametrize(
    ("gradient_accumulation_steps", "sdc_num_steps", "sdc_num_replays", "expected"),
    [
        (1, 0, 0, 2),
        (4, 0, 0, 8),
        (1, 1, 1, 3),
        (4, 2, 1, 10),
        (4, -1, 1, 10),
        (4, 1, 3, 11),
        (4, 5, 3, 14),
    ],
)
def test_cuda_graph_warmup_covers_two_optimizer_steps(
    gradient_accumulation_steps: int,
    sdc_num_steps: int,
    sdc_num_replays: int,
    expected: int,
) -> None:
    graph = MagicMock()
    fn = MagicMock(side_effect=lambda value: value)
    with (
        patch("torchtitan.distributed.cudagraph.utils.device_type", "cuda"),
        patch("torch.cuda.is_available", return_value=True),
        patch.object(torch.version, "hip", None),
        patch.object(_manager, "maybe_initialize"),
        patch.object(_manager, "register"),
        patch.object(_manager, "_graph_pool", object()),
        patch.object(_manager, "_stream", MagicMock()),
        patch("torch.cuda.current_stream", return_value=MagicMock()),
        patch("torch.cuda.stream", return_value=nullcontext()),
        patch("torch.cuda.CUDAGraph", return_value=graph) as graph_constructor,
        patch("torch.cuda.graph", return_value=nullcontext()),
        patch(
            "torchtitan.distributed.cudagraph.get_kernel_annotations",
            return_value={},
        ),
    ):
        run = wrap_with_cuda_graph(
            fn,
            gradient_accumulation_steps=gradient_accumulation_steps,
            sdc_num_steps=sdc_num_steps,
            sdc_num_replays=sdc_num_replays,
        )
        value = torch.tensor(1.0)
        for _ in range(expected):
            run(value)
        graph_constructor.assert_not_called()
        assert fn.call_count == expected

        run(value)
        graph_constructor.assert_called_once()
        graph.replay.assert_called_once()
        run(value)
        assert fn.call_count == expected + 1
        assert graph.replay.call_count == 2


def test_tensor_input_indices_control_replay_copies() -> None:
    static_input = torch.tensor(1)
    excluded_input = torch.tensor(2)
    copied_input = torch.tensor(3)

    with (
        patch.object(_manager, "maybe_initialize"),
        patch.object(_manager, "register"),
    ):
        wrapper = CUDAGraphWrapper(
            lambda *args: args,
            (static_input, excluded_input, copied_input),
            static_input_indices=(0,),
            tensor_input_indices=[0, 2],
        )

    wrapper._warmup_remaining = 0
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


def test_cudagraph_wrapper_collects_annotations() -> None:
    graph = cast(torch.cuda.CUDAGraph, MagicMock())
    annotations = {42: [{"module_fqn": "layers.0"}]}
    graph_pool = object()
    stream = MagicMock()

    with (
        patch.object(_manager, "maybe_initialize"),
        patch.object(_manager, "register"),
        patch.object(_manager, "_graph_pool", graph_pool),
        patch.object(_manager, "_stream", stream),
        patch.object(_manager, "all_annotations", {}),
        patch("torch.cuda.CUDAGraph", return_value=graph),
        patch("torch.cuda.graph", return_value=nullcontext()) as cuda_graph,
        patch(
            "torchtitan.distributed.cudagraph.get_kernel_annotations",
            return_value=annotations,
        ),
    ):
        wrapper = CUDAGraphWrapper(lambda x: x, (torch.tensor(1),))
        wrapper._warmup_remaining = 0

        output = wrapper(torch.tensor(2))

        assert output.item() == 2
        assert get_cudagraph_annotations() == annotations
        cuda_graph.assert_called_once_with(
            graph,
            pool=graph_pool,
            stream=stream,
            enable_annotations=True,
            capture_error_mode="thread_local",
        )


def test_structured_wrapper_validates_and_copies_replay_inputs() -> None:
    graph = cast(torch.cuda.CUDAGraph, MagicMock())
    graph_stream = MagicMock()
    current_stream = MagicMock()
    fn = MagicMock(side_effect=lambda batches, *, scale: batches[1]["x"] * scale)

    with (
        patch("torchtitan.distributed.cudagraph.utils.device_type", "cuda"),
        patch("torch.cuda.is_available", return_value=True),
        patch.object(torch.version, "hip", None),
        patch.object(_manager, "maybe_initialize"),
        patch.object(_manager, "register"),
        patch.object(_manager, "_graph_pool", object()),
        patch.object(_manager, "_stream", graph_stream),
        patch("torch.cuda.current_stream", return_value=current_stream),
        patch("torch.cuda.stream", return_value=nullcontext()),
        patch("torch.cuda.CUDAGraph", return_value=graph),
        patch("torch.cuda.graph", return_value=nullcontext()),
        patch(
            "torchtitan.distributed.cudagraph.get_kernel_annotations",
            return_value={},
        ),
    ):
        run = wrap_with_cuda_graph(
            fn,
            gradient_accumulation_steps=1,
            sdc_num_steps=0,
            sdc_num_replays=0,
            num_warmup_steps=1,
        )
        torch.testing.assert_close(
            run(
                [{"x": torch.tensor(1.0)}, {"x": torch.tensor(2.0)}],
                scale=torch.tensor(3.0),
            ),
            torch.tensor(6.0),
        )
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

    assert fn.call_count == 2
    captured_batches = fn.call_args.args[0]
    torch.testing.assert_close(captured_batches[0]["x"], torch.tensor(6.0))
    torch.testing.assert_close(captured_batches[1]["x"], torch.tensor(7.0))
    torch.testing.assert_close(fn.call_args.kwargs["scale"], torch.tensor(5.0))
    assert cast(MagicMock, graph.replay).call_count == 2
