# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import pytest
import torch
from torchtitan.distributed.flex_shard.dist_muon import (
    _can_use_quack_symmetric_mm,
    _zeropower_via_newtonschulz,
)


pytestmark = [
    pytest.mark.multi_gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


@pytest.mark.parametrize("shape", [(128, 64), (2, 128, 64)])
def test_quack_symmetric_gemm_correctness(shape):
    """Check PyTorch's native symmetric MM without an external QuACK import."""
    if torch.version.hip is not None or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100")
    from torch._inductor.kernel.symmetric_mm import quack_symmetric_mm
    from torch._inductor.utils import ensure_cute_available

    if not ensure_cute_available():
        pytest.skip("requires CuTe DSL")
    torch.manual_seed(0)
    matrices = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    before = matrices.clone()
    actual = quack_symmetric_mm(matrices)
    expected = matrices @ matrices.mT
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual, actual.mT, rtol=0, atol=0)
    torch.testing.assert_close(matrices, before, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(4096, 4096), (2, 4096, 4096)])
def test_quack_newton_schulz_matches_torch(shape):
    """Compare full production NS iterations and confirm native op dispatch."""
    torch.manual_seed(1)
    matrices = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    if not _can_use_quack_symmetric_mm(matrices):
        pytest.skip("requires SM100 and CuTe DSL")
    from torch._inductor.kernel import symmetric_mm

    kwargs = dict(ns_coefficients=(3.4445, -4.7750, 2.0315), ns_steps=5, eps=1e-7)
    with mock.patch(
        "torchtitan.distributed.flex_shard.dist_muon._can_use_quack_symmetric_mm",
        return_value=False,
    ):
        expected = _zeropower_via_newtonschulz(matrices, **kwargs)
    with mock.patch.object(
        symmetric_mm, "quack_symmetric_mm", wraps=symmetric_mm.quack_symmetric_mm
    ) as native_mm:
        actual = _zeropower_via_newtonschulz(matrices, **kwargs)
    assert native_mm.call_count == 2 * kwargs["ns_steps"]
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("shape", [(128, 64), (2, 128, 64), (4097, 4104)])
def test_quack_newton_schulz_fallback_matches_torch(shape):
    """Unsupported sizes and strides retain the original NS arithmetic."""
    torch.manual_seed(1)
    matrices = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    assert not _can_use_quack_symmetric_mm(matrices)
    kwargs = dict(ns_coefficients=(3.4445, -4.7750, 2.0315), ns_steps=2, eps=1e-7)
    with mock.patch(
        "torchtitan.distributed.flex_shard.dist_muon._can_use_quack_symmetric_mm",
        return_value=False,
    ):
        expected = _zeropower_via_newtonschulz(matrices, **kwargs)
    actual = _zeropower_via_newtonschulz(matrices, **kwargs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
