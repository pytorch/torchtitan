# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.qwen3_8_flash_next.model import GatedResidual

# The released model's gated residual: rank 320, 4 streams.
_RANK, _NUM_STREAMS = 320, 4


def _relative_error(value: torch.Tensor, golden: torch.Tensor) -> float:
    return ((value.double() - golden).norm() / golden.norm()).item()


def _run(
    activation, x: torch.Tensor, grad_output: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.clone().requires_grad_()
    out = activation(x, _NUM_STREAMS)
    (grad_x,) = torch.autograd.grad(out, x, grad_output)
    return out.detach(), grad_x


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Inductor GPU kernels")
@pytest.mark.parametrize(
    "activation, width",
    [(GatedResidual.mix_activation, _RANK), (GatedResidual.write_gates, _NUM_STREAMS)],
    ids=["mix_activation", "write_gates"],
)
def test_compiled_activation_matches_eager(activation, width):
    """The ``gated_residual`` region keeps the activations' outputs bitwise.

    Inductor keeps ``x / S`` and the sigmoid in fp32 instead of rounding each
    to bf16. The outputs still equal eager's because dividing by S = 4 and
    multiplying by 2 are exact. The write-gate gradient uses the unrounded
    sigmoid, so it may differ from eager's in the last bit, but must be no
    further from fp64.
    """
    torch.manual_seed(0)
    # 257 tokens leave a partial last block in the compiled kernels.
    x = torch.randn(257, width, device="cuda") * 3
    grad_output = torch.randn_like(x)
    _, golden_grad = _run(activation, x.double(), grad_output.double())
    results = []
    torch._dynamo.reset()
    try:
        for regions in ([], ["gated_residual"]):
            LocalCompileConfig(regions=regions).apply()
            results.append(_run(activation, x.bfloat16(), grad_output.bfloat16()))
    finally:
        LocalCompileConfig().apply()
        torch._dynamo.reset()
    (eager_out, eager_grad), (compiled_out, compiled_grad) = results
    torch.testing.assert_close(compiled_out, eager_out, atol=0, rtol=0)
    assert _relative_error(compiled_grad, golden_grad) <= _relative_error(
        eager_grad, golden_grad
    )
