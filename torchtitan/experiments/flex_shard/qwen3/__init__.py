# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.models.qwen3 import model_registry as qwen3_model_registry

from .parallelize import parallelize_qwen3_muon


def model_registry(*args, **kwargs):
    """Qwen3 model spec using the FlexShard + Muon parallelizer.

    Reuses the core Qwen3 model unchanged; only swaps the parallelize_fn for the
    experimental eager FlexShard + communication-efficient Muon path.
    """
    spec = qwen3_model_registry(*args, **kwargs)
    spec.name = "flex_shard.qwen3"
    spec.parallelize_fn = parallelize_qwen3_muon
    spec.pipelining_fn = None
    return spec


__all__ = ["model_registry", "parallelize_qwen3_muon"]
