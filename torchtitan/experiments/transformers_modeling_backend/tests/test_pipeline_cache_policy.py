# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import Mock

import pytest
import torch

from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.experiments.transformers_modeling_backend.pipeline import (
    pipeline_hf_transformers,
)


def test_hf_pipeline_rejects_ignored_startup_cache_policy() -> None:
    """Native HF pipeline stages would silently skip core's cache release."""
    with pytest.raises(
        ValueError, match="do not support pipeline_parallel_empty_cache"
    ):
        pipeline_hf_transformers(
            Mock(),
            Mock(),
            training=TrainingConfig(),
            parallelism=ParallelismConfig(
                pipeline_parallel_degree=2,
                pipeline_parallel_empty_cache_before_first_backward=True,
            ),
            local_compile=LocalCompileConfig(),
            activation_remat=None,
            dump_folder="unused",
            device=torch.device("cpu"),
            model_config=Mock(),
            loss_fn=Mock(),
        )
