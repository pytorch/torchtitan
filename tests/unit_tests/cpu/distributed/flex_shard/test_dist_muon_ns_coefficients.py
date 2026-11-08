# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.distributed.flex_shard.dist_muon import (
    _valid_ns_coefficients,
    _zeropower_via_newtonschulz,
)


@pytest.mark.filterwarnings("ignore:mkldnn_matmul failed.*")
def test_dist_muon_applies_shared_or_per_step_ns_coefficients() -> None:
    update = torch.arange(1, 13, dtype=torch.float32).reshape(4, 3)
    shared = (3.4445, -4.7750, 2.0315)
    repeated = (shared, shared)
    per_step = (shared, (3.0, -4.0, 2.0))

    shared_result = _zeropower_via_newtonschulz(
        update,
        ns_coefficients=shared,
        ns_steps=2,
        eps=1e-7,
        ns_dtype=torch.bfloat16,
    )
    repeated_result = _zeropower_via_newtonschulz(
        update,
        ns_coefficients=repeated,
        ns_steps=2,
        eps=1e-7,
        ns_dtype=torch.bfloat16,
    )
    per_step_result = _zeropower_via_newtonschulz(
        update,
        ns_coefficients=per_step,
        ns_steps=2,
        eps=1e-7,
        ns_dtype=torch.bfloat16,
    )

    assert torch.equal(shared_result, repeated_result)
    assert not torch.equal(shared_result, per_step_result)
    assert _valid_ns_coefficients(shared, ns_steps=2)
    assert _valid_ns_coefficients(per_step, ns_steps=2)
    assert not _valid_ns_coefficients(per_step, ns_steps=3)


def test_float16_newton_schulz_tracks_float32_reference() -> None:
    # Low-rank momentum at a tiny scale. A float16 cast before the normalization
    # flushes it to zero; a BF16 division rounds the normalized entries again.
    generator = torch.Generator().manual_seed(0)
    left = torch.randn(256, 8, generator=generator)
    right = torch.randn(8, 512, generator=generator)
    update = ((left * torch.logspace(0, -3, 8)) @ right * 1e-9).to(torch.bfloat16)

    def newton_schulz(ns_dtype: torch.dtype) -> torch.Tensor:
        return _zeropower_via_newtonschulz(
            update,
            ns_coefficients=(3.4445, -4.7750, 2.0315),
            ns_steps=5,
            eps=1e-7,
            ns_dtype=ns_dtype,
        ).float()

    reference = newton_schulz(torch.float32)
    bf16_error = (newton_schulz(torch.bfloat16) - reference).abs().mean()
    fp16_error = (newton_schulz(torch.float16) - reference).abs().mean()

    assert fp16_error < bf16_error / 3
