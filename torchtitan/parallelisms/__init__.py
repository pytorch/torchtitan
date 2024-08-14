# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from torchtitan.parallelisms.parallel_dims import ParallelDims
from torchtitan.parallelisms.parallelize_llama import parallelize_llama


__all__ = [
    "models_parallelize_fns",
    "ParallelDims",
]

models_parallelize_fns = {
    "llama2": parallelize_llama,
    "llama3": parallelize_llama,
}
