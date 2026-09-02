# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext
from typing import cast
from unittest.mock import MagicMock, patch

import torch

from torchtitan.distributed.cudagraph import (
    _manager,
    CUDAGraphWrapper,
    get_cudagraph_annotations,
)


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
