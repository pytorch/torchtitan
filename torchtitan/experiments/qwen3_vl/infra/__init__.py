# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3-VL parallelization utilities."""

from .parallelize import parallelize_qwen3_vl

__all__ = ["parallelize_qwen3_vl"]
