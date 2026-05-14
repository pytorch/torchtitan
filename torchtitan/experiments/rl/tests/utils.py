# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pytest markers for RL hardware requirements."""

import pytest


def _check_num_gpus(num_gpus: int) -> None:
    if num_gpus < 1:
        raise ValueError(f"num_gpus must be >= 1, got {num_gpus}")


class _MarkDecorators(list):
    """Pytest mark list that can also decorate a test object."""

    def __call__(self, obj):
        for mark in self:
            obj = mark(obj)
        return obj


def gpu_test(num_gpus: int = 1):
    """Mark a pytest test that requires CUDA GPUs."""
    _check_num_gpus(num_gpus)
    return pytest.mark.gpu(num_gpus=num_gpus)


def h100_test(num_gpus: int = 1):
    """Mark a pytest test that requires H100 GPUs."""
    _check_num_gpus(num_gpus)
    return _MarkDecorators([pytest.mark.gpu(num_gpus=num_gpus), pytest.mark.h100])
