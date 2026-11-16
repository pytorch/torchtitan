# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Geometry3K: a geometry diagram + problem -> a final answer, graded by Math-Verify."""

from torchtitan.rl.examples.geometry3k.data import Geometry3kDataset, Geometry3kSample
from torchtitan.rl.examples.geometry3k.env import Geometry3kEnv

__all__ = [
    "Geometry3kDataset",
    "Geometry3kEnv",
    "Geometry3kSample",
]
