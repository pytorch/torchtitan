# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast
from unittest.mock import MagicMock, patch

import torch

from torchtitan.distributed.cudagraph import CUDAGraphWrapper, _manager


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
