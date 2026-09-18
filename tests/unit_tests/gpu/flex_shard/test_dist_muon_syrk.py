# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import pytest
import torch


pytestmark = pytest.mark.multi_gpu


def _quack_symmetric_gemm(matrices: torch.Tensor) -> torch.Tensor:
    pytest.importorskip("quack")
    try:
        from quack.gemm_symmetric import gemm_symmetric
    except ImportError as exc:
        pytest.skip(f"installed QuACK has no gemm_symmetric kernel: {exc}")

    squeeze_batch = matrices.ndim == 2
    if squeeze_batch:
        matrices = matrices.unsqueeze(0)
    output = torch.empty(
        (*matrices.shape[:-1], matrices.shape[-2]),
        device=matrices.device,
        dtype=matrices.dtype,
    )
    gemm_symmetric(
        matrices,
        matrices,
        output,
        None,
        None,
        tile_M=128,
        tile_N=128,
        cluster_M=1,
        cluster_N=1,
    )
    return output.squeeze(0) if squeeze_batch else output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("shape", [(128, 64), (2, 128, 64)])
def test_quack_symmetric_gemm_correctness(shape):
    """Verify QuACK Gram output shape, symmetry, and numerical correctness."""
    torch.manual_seed(0)
    matrices = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    matrices_before = matrices.clone()
    actual = _quack_symmetric_gemm(matrices)
    expected = (
        matrices @ matrices.T
        if matrices.ndim == 2
        else torch.bmm(matrices, matrices.transpose(-2, -1))
    )

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual, actual.transpose(-2, -1), rtol=0, atol=0)
    torch.testing.assert_close(matrices, matrices_before, rtol=0, atol=0)


def _newton_schulz_step(matrices: torch.Tensor, use_quack: bool) -> torch.Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    if not use_quack:
        gram = torch.bmm(matrices, matrices.transpose(-2, -1))
        gram_update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        return torch.baddbmm(matrices, gram_update, matrices, beta=a)

    gram = _quack_symmetric_gemm(matrices)
    gram_square = _quack_symmetric_gemm(gram)
    gram_update = gram.mul(b).add_(gram_square, alpha=c)
    return a * matrices + torch.bmm(gram_update, matrices)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_quack_newton_schulz_matches_bmm():
    """Verify one complete Newton-Schulz step against the PyTorch path."""
    torch.manual_seed(1)
    matrices = torch.randn(1, 128, 64, device="cuda", dtype=torch.bfloat16)
    matrices.div_(matrices.norm().clamp_min(1e-7))
    expected = _newton_schulz_step(matrices, use_quack=False)
    actual = _newton_schulz_step(matrices, use_quack=True)
    # QuACK and baddbmm use different BF16 reduction schedules. The paths are
    # mathematically equivalent, but they are not expected to be bitwise equal.
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)
