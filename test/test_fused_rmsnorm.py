# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.


import pytest
import torch
import torch.nn as nn
from torch import Tensor
import sys
sys.path.append('..')
from torchtrain.models.norms import FusedRMSNorm # fused_rms_norm_fn as triton_rmsnorm

from torchtrain.test.testing_utils import assert_expected, set_rng_seed, gpu_test

@pytest.fixture(autouse=True)
def set_seed():
    set_rng_seed(2023)

class TorchRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization
    as proposed in: https://arxiv.org/abs/1910.07467

    Calcs are done in fp32.

    original impl: https://github.com/facebookresearch/llama/blob/main/llama/model.py

    Args:
        dim(int) = model size
        eps(float) = epsilon
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

import time

class TestRMSNorm:
    @pytest.fixture
    def N(self,):
        return 8192

    def test_triton_fused_vs_pytorch_accuracy(self, N):
        batch_size = 8
        layer_weight_size = (N,N)
        test_eps=1e-8

        sample_x = torch.randn(layer_weight_size, dtype=torch.float32, device='cuda', requires_grad=False)

        expected_rms_func = TorchRMSNorm(layer_weight_size,eps = test_eps).to('cuda')
        fused_rms_norm = FusedRMSNorm(layer_weight_size, eps=test_eps).to('cuda')

        expected_rms = expected_rms_func(sample_x)
        fused_out = fused_rms_norm(sample_x)

        assert_expected(fused_out, expected_rms, rtol=.0001, atol=.0001)
