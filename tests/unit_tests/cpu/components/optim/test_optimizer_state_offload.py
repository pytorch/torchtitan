# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn as nn
from torchtitan.components.optim import (
    AdamW,
    Optim,
    OptimizersContainer,
    OptimizerStateOffloader,
)
from torchtitan.components.optim.offload import _pack_params_by_state_bytes


@pytest.mark.parametrize(
    "numels, state_dtype, expected_chunks",
    [
        # fp32 moments: 8 B per element; 1 MiB = 131072 elements. Cut before exceeding it.
        ((60_000, 60_000, 60_000, 10), torch.float32, [[0, 1], [2, 3]]),
        # bf16 moments: 4 B per element, so twice as many params fit.
        ((60_000, 60_000, 60_000, 60_000), torch.bfloat16, [[0, 1, 2, 3]]),
        # A param larger than the target forms its own chunk.
        ((10, 500_000, 10), torch.float32, [[0], [1], [2]]),
    ],
)
def test_packer_keeps_whole_params_in_order(
    numels: tuple[int, ...],
    state_dtype: torch.dtype,
    expected_chunks: list[list[int]],
) -> None:
    params = [torch.empty(numel) for numel in numels]
    chunks = _pack_params_by_state_bytes(
        params, chunk_size_bytes=1024 * 1024, state_dtype=state_dtype
    )
    index_by_id = {id(param): index for index, param in enumerate(params)}
    assert [
        [index_by_id[id(param)] for param in chunk] for chunk in chunks
    ] == expected_chunks


def test_unsupported_configurations_raise() -> None:
    with pytest.raises(ValueError, match="chunk_size_mb"):
        OptimizerStateOffloader.Config(chunk_size_mb=0)
    with pytest.raises(ValueError, match="fused=True"):
        AdamW.Config(
            pattern=r".*", fused=False, state_offload=OptimizerStateOffloader.Config()
        )
    with pytest.raises(ValueError, match="enable_accelerator_graph"):
        Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[
                    AdamW.Config(
                        pattern=r".*", state_offload=OptimizerStateOffloader.Config()
                    )
                ]
            ),
            enable_accelerator_graph=True,
        )
    with pytest.raises(ValueError, match="CUDA-resident"):
        OptimizerStateOffloader.Config().build(
            optimizer=torch.optim.AdamW([nn.Parameter(torch.zeros(4))], fused=True),
            state_dtype=torch.float32,
        )


def test_container_rejects_mixed_param_dtypes() -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2).to(torch.bfloat16))
    config = OptimizersContainer.Config(
        optimizers=[
            AdamW.Config(pattern=r".*", state_offload=OptimizerStateOffloader.Config())
        ],
    )
    with pytest.raises(ValueError, match="one parameter dtype"):
        config.build(model_parts=[model])
