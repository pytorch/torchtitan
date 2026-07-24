# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any
from unittest.mock import MagicMock, patch

import torch
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.distributed.context_parallel import prepare_context_parallel_input


def test_prepare_context_parallel_input_shards_additional_sequence_inputs() -> None:
    inputs = torch.tensor([[1, 2]])
    labels = torch.tensor([[2, 3]])
    positions = torch.tensor([[0, 1]])
    generator_logprobs = torch.tensor([[0.1, 0.2]])
    loss_mask = torch.tensor([[True, False]])
    attention_masks = object()
    sharded = tuple(
        tensor + 1
        for tensor in (inputs, labels, positions, generator_logprobs, loss_mask)
    )
    sharded_attention_masks = object()
    cp_mesh = MagicMock(spec=DeviceMesh)
    extra_kwargs: dict[str, Any] = {
        "positions": positions,
        "attention_masks": attention_masks,
        "generator_logprobs": generator_logprobs,
        "loss_mask": loss_mask,
    }

    with patch(
        "torchtitan.distributed.context_parallel.cp_shard",
        return_value=(sharded, sharded_attention_masks),
    ) as cp_shard:
        actual_inputs, actual_labels, actual_kwargs = prepare_context_parallel_input(
            inputs,
            labels,
            extra_kwargs,
            cp_mesh=cp_mesh,
            device=torch.device("cpu"),
            load_balancer_type="ptrr",
            additional_sequence_input_keys=("generator_logprobs", "loss_mask"),
        )

    cp_shard.assert_called_once_with(
        cp_mesh,
        (inputs, labels, positions, generator_logprobs, loss_mask),
        attention_masks,
        "ptrr",
    )
    assert actual_inputs is sharded[0]
    assert actual_labels is sharded[1]
    assert actual_kwargs["positions"] is sharded[2]
    assert actual_kwargs["attention_masks"] is sharded_attention_masks
    assert actual_kwargs["generator_logprobs"] is sharded[3]
    assert actual_kwargs["loss_mask"] is sharded[4]
