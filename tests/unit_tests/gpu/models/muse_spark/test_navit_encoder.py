# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Real CUDA FlexAttention coverage and opt-in strict numerical comparison."""

import importlib.util
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

spec = importlib.util.spec_from_file_location(
    "navit_reference",
    Path(__file__).resolve().parents[3] / "cpu/models/common/test_navit_encoder.py",
)
assert spec is not None and spec.loader is not None
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
CASES = reference.CASES
check_reference = reference.check_reference
numerical_reference = reference.numerical_reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("case", [CASES[2], CASES[5], CASES[6]])
def test_spark_encoder_cuda_smoke(case):
    # CUDA grid_sample backward is nondeterministic in current PyTorch.
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        result = numerical_reference(case, "cuda")
        assert torch.isfinite(result["encoded"]).all()
        assert torch.isfinite(result["input_grad"]).all()
    finally:
        torch.use_deterministic_algorithms(was_deterministic)


@pytest.mark.parametrize("case", CASES)
def test_encoder_cuda_parity(case):
    if "TORCHTITAN_NAVIT_REFERENCE_MODE" not in os.environ:
        pytest.skip("Requires immutable pre-change references and explicit record/compare mode")
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        # Fail before recording anything if trainable positional sampling is unsupported.
        weights = torch.randn(1, 8, 2, 2, device="cuda", requires_grad=True)
        grid = torch.zeros(1, 1, 4, 2, device="cuda")
        F.grid_sample(weights, grid, align_corners=False).sum().backward()
        check_reference(case, numerical_reference(case, "cuda"), "cuda")
    finally:
        torch.use_deterministic_algorithms(was_deterministic)
