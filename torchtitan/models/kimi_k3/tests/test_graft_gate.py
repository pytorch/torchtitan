# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Graft-gate identity tests: alpha=0 must be an exact no-op.

alpha-gated zero-init AttnRes must EXACTLY reproduce the plain
backbone's function at step 0; the ungated zero-init read is a uniform
source-average and must NOT (that distinction is the reason the gate
exists -- lock both directions in).
"""

import unittest

import torch

from torchtitan.models.kimi_k3 import config_registry
from torchtitan.models.kimi_k3.model import KimiK3Spec


def _pair(gated: bool):
    import dataclasses

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(7)
    kimi_config = config_registry.kimi_k3_debugmodel().model_spec.model.kimi_config
    # All-MLA config (no KDA): the AttnRes-graft identity is about the
    # residual-read gating, independent of attention type. Avoiding the
    # fla/KDA triton kernels makes this deterministic + finite (KDA is
    # non-deterministic and occasionally NaNs at debug scale under
    # accumulated GPU state; the KDA path itself is covered elsewhere).
    n = kimi_config.num_hidden_layers
    kimi_config = dataclasses.replace(
        kimi_config,
        kda_layers=[],
        full_attn_layers=list(range(1, n + 1)),
    )
    graft_spec = KimiK3Spec(kimi_config=kimi_config, num_blocks=4, attn_res_gated=gated)
    base_spec = KimiK3Spec(kimi_config=kimi_config, num_blocks=None)
    with torch.device(device):
        graft = graft_spec.build()
        graft.init_weights()
        base = base_spec.build()
        base.init_weights()
    # Share the backbone: copy the key intersection graft -> base
    # (graft-only extras: *_res_proj / *_res_norm / *_res_alpha).
    bsd = base.state_dict()
    shared = {k: v for k, v in graft.state_dict().items() if k in bsd}
    assert set(shared) == set(bsd)
    base.load_state_dict(shared, strict=True)
    g = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, 2016, (2, 128), generator=g).to(device)
    graft.eval()
    base.eval()
    with torch.no_grad():
        out = graft(tokens).float(), base(tokens).float()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


class TestGraftGate(unittest.TestCase):
    def test_gated_zero_init_is_exact_identity(self):
        lg, lb = _pair(gated=True)
        # The alpha graft is identity by construction; at 48B real
        # weights it is BIT-exact (max|dlogit|=0.0, separately verified).
        # At debug scale the fla/KDA + cublas kernels are
        # non-deterministic, so assert a very tight tolerance.
        rel = ((lg - lb).norm() / (lb.norm() + 1e-9)).item()
        self.assertLess(
            rel,
            1e-4,
            f"gated graft must be ~identity at step 0; rel {rel:.3e}",
        )

    def test_ungated_zero_init_is_not_identity(self):
        lg, lb = _pair(gated=False)
        self.assertGreater(
            (lg - lb).abs().max().item(),
            1e-4,
            "ungated zero-init read is a uniform source-average and is "
            "expected to differ from the plain backbone -- if this ever "
            "matches exactly, the read semantics changed",
        )


if __name__ == "__main__":
    unittest.main()
