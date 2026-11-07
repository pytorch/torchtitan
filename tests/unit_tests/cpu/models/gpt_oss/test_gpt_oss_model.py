# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.gpt_oss.model import apply_attention_sink_rescale


@pytest.fixture(autouse=True)
def reset_local_compile():
    LocalCompileConfig().apply()
    torch._dynamo.reset()
    yield
    LocalCompileConfig().apply()


def test_compiled_attention_sinks_match_eager() -> None:
    torch.manual_seed(0)
    out_THD = torch.randn(16, 4, 8, dtype=torch.float64)
    lse_TH = torch.randn(4, 16, dtype=torch.float64).transpose(0, 1)
    sinks_H = torch.randn(4, dtype=torch.float64)
    grad_THD = torch.randn_like(out_THD)
    results = []
    for regions in ([], ["attention_sinks"]):
        LocalCompileConfig(regions=regions).apply()
        inputs = [t.clone().requires_grad_(True) for t in (out_THD, lse_TH, sinks_H)]
        rescaled_THD = apply_attention_sink_rescale(*inputs)
        rescaled_THD.backward(grad_THD)
        results.append((rescaled_THD.detach(), *(t.grad for t in inputs)))
    for eager, compiled in zip(*results, strict=True):
        torch.testing.assert_close(compiled, eager)
