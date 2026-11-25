# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.config import derive
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.common.attention.gdn import GDN
from torchtitan.models.qwen3_8_flash_next import build_model_config
from torchtitan.models.qwen3_8_flash_next.attention import QSAIndexerLoss
from torchtitan.models.qwen3_8_flash_next.gdn import SplitInProjGDN


@pytest.fixture(params=[False, True], ids=["eager", "compiled"])
def split_region(request):
    """Runs both splits eagerly, or as the compiled ``gdn_split_in_proj`` region."""
    torch._dynamo.reset()
    if request.param:
        LocalCompileConfig(regions=["gdn_split_in_proj"]).apply()
    yield
    LocalCompileConfig().apply()
    torch._dynamo.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GDN kernels require CUDA")
def test_split_in_proj_gdn_is_bitwise_equal_to_gdn(split_region):
    """Skipping the interleaved copy leaves every GDN output and gradient unchanged.

    Catches a q/k/v or z/a/b slice that disagrees with the shared GDN's split of
    the interleaved layout (e.g. after a head-layout change in SplitLinear), in
    eager mode and in the compiled region the recipes run.
    """
    torch.manual_seed(0)
    config = (
        build_model_config(
            "debugmodel",
            seq_len=64,
            parallelism=ParallelismConfig(),
            indexer_loss=QSAIndexerLoss.Config(coeff=1.0),
        )
        .layers[0]
        .attention
    )
    assert type(config) is SplitInProjGDN.Config
    split = config.build().cuda().bfloat16()
    split.init_states()
    reference = derive(config, GDN.Config).build().cuda().bfloat16()
    reference.load_state_dict(split.state_dict())
    x = torch.randn(48, config.in_proj.in_features, device="cuda").bfloat16()
    outputs, gradients = [], []
    for module in (reference, split):
        inp = x.clone().requires_grad_()
        out = module(inp)
        out.square().sum().backward()
        outputs.append(out.detach())
        gradients.append([inp.grad, *(p.grad for p in module.parameters())])
    torch.testing.assert_close(outputs[1], outputs[0], atol=0, rtol=0)
    for actual, expected in zip(gradients[1], gradients[0], strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
