# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Optimizer container with SinkhornMomentum factory support."""

from torchtitan.components.optim.optimizer import OptimizersContainer

from .optimizer import SinkhornMomentum


class ReferenceOptimizers(OptimizersContainer):
    @staticmethod
    def _resolve_optimizer_factory(name):
        if name == "SinkhornMomentum":
            return SinkhornMomentum
        return OptimizersContainer._resolve_optimizer_factory(name)
