# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.


import sys

import pytest
import torch
import torch.nn as nn
from torch import Tensor

#sys.path.append("..")
from .fused_rms_norm import FusedRMSNorm
#from .nv_apex import FusedRMSNorm as nv_apex_FusedRMSNorm

from .testing_utils import assert_expected, gpu_test, set_rng_seed


@pytest.fixture(autouse=True)
def set_seed():
    set_rng_seed(2024)


class TorchRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization
    as proposed in: https://arxiv.org/abs/1910.07467

    Calcs are done in fp32.

    original impl: https://github.com/facebookresearch/llama/blob/main/llama/model.py

    """

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        x_normed = self._norm(x.float()).type_as(x)
        return x_normed * self.scale


@gpu_test(1)
class TestRMSNorm:
    @pytest.fixture
    def n_dim(
        self,
    ):
        return 512

    def test_cuda_fused_vs_pytorch_accuracy(self, n_dim):
        batch_size = 30
        layer_weight_size = (n_dim,)
        test_eps = 1e-8
        atol_precision = 1e-2
        rtol_precision = 1e-2

        sample_x = torch.randn(
            layer_weight_size, dtype=torch.float32, device="cuda", requires_grad=True
        )

        expected_rms_func = TorchRMSNorm(layer_weight_size, eps=test_eps).to("cuda")
        fused_rms_norm = FusedRMSNorm(layer_weight_size, eps=test_eps).to("cuda")
        #fused_rms_norm = nv_apex_FusedRMSNorm(layer_weight_size, eps=test_eps).to("cuda")

        expected_rms = expected_rms_func(sample_x)
        fused_out = fused_rms_norm(sample_x)

        print(f"{fused_out.shape=}")
        print(f"{fused_out[0:10]=}")
        print(f"{expected_rms[0:10]=}")

        # Check forward pass accuracy
        assert_expected(
            fused_out, expected_rms, rtol=rtol_precision, atol=atol_precision
        )
        print(f"SUCCESS FWD!")


        # Backward pass
        grad_output = torch.randn_like(expected_rms)

        expected_rms.backward(grad_output)
        dy_expected = sample_x.grad

        sample_x.grad = None

        fused_out.backward(grad_output)
        dy_fused = sample_x.grad

        print(f"{dy_fused[0:5]=}")
        print(f"{dy_expected[0:5]=}")

        # Check backward pass accuracy
        #assert_expected(dy_expected, dy_fused, rtol=rtol_precision, atol=atol_precision)
