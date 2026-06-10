# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.models.deepseek_v3 import model_registry as deepseek_v3_model_registry

from .parallelize import parallelize_deepseekv3


def model_registry(*args, **kwargs):
    """DeepSeek V3 model registry using the experimental FlexShard parallelizer."""
    spec = deepseek_v3_model_registry(*args, **kwargs)
    spec.name = "flex_shard.deepseek_v3"
    spec.parallelize_fn = parallelize_deepseekv3
    spec.pipelining_fn = None
    return spec


__all__ = ["model_registry", "parallelize_deepseekv3"]
