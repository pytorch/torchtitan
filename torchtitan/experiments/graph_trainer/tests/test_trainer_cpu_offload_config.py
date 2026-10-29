# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest

from torchtitan.config import ParallelismConfig
from torchtitan.experiments.graph_trainer.trainer import GraphTrainingEngine


def test_simple_fsdp_rejects_native_layer_cpu_offload_before_initialization():
    """Do not materialize CPU shards for a GPU-only SimpleFSDP policy."""
    config = SimpleNamespace(
        optim=SimpleNamespace(enable_accelerator_graph=False),
        parallelism=ParallelismConfig(fsdp_cpu_offload_layer_fqns=("layers.0",)),
    )
    with pytest.raises(ValueError, match="SimpleFSDP does not support"):
        GraphTrainingEngine(
            config,
            model_config=SimpleNamespace(local_compile=SimpleNamespace(regions=[])),
            max_num_documents=None,
            output_dir="unused",
        )
