# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Short Trainer recipes exercised by the GPU smoke tests."""

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import FullActivationRematConfig
from torchtitan.experiments.peri_ln.recipes import llama3_debugmodel
from torchtitan.trainer import Trainer


def single_gpu() -> Trainer.Config:
    return llama3_debugmodel()


def tensor_parallel() -> Trainer.Config:
    config = llama3_debugmodel(
        parallelism=ParallelismConfig(
            tensor_parallel_degree=2, enable_sequence_parallel=True
        )
    )
    config.activation_remat = FullActivationRematConfig()
    return config
