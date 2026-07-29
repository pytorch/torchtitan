# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.tools.utils import get_cuda_flash_attention_impl


@pytest.mark.parametrize(
    ("capability", "expected_impl"),
    [
        ((8, 0), None),
        ((9, 0), "FA3"),
        ((9, 1), "FA3"),
        ((10, 0), "FA4"),
        ((10, 3), "FA4"),
        # SM 11.0+ falls through to the newest known impl (FA4).
        ((11, 0), "FA4"),
    ],
)
def test_get_cuda_flash_attention_impl(monkeypatch, capability, expected_impl):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    monkeypatch.setattr(torch.version, "hip", None)

    assert get_cuda_flash_attention_impl() == expected_impl


def test_get_cuda_flash_attention_impl_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert get_cuda_flash_attention_impl() is None


def test_get_cuda_flash_attention_impl_on_rocm(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "7.0")

    assert get_cuda_flash_attention_impl() is None
