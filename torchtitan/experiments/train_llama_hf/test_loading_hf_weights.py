# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import subprocess

from pathlib import Path

import pytest


PRETRAINED_MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.mark.parametrize("dp_shard_degree", [-1])
@pytest.mark.parametrize("tp_degree", [2, 4])
@pytest.mark.parametrize("pp_degree", [2])
@pytest.mark.parametrize("world_size", [8])
def test_load_sharded_state_dict_for_model_from_hf(
    dp_shard_degree, tp_degree, pp_degree, world_size
):
    test_file_path = Path(__file__).parent / "test_loading_hf_weights_helper.py"
    cmd = [
        "torchrun",
        "--local-ranks-filter",
        "0",
        "--nproc_per_node",
        str(world_size),
        str(test_file_path),
        "--experimental.pipeline_parallel_degree",
        str(pp_degree),
        "--training.tensor_parallel_degree",
        str(tp_degree),
        "--training.data_parallel_shard_degree",
        str(dp_shard_degree),
        "--model.name",
        PRETRAINED_MODEL_ID,
        "--model.flavor",
        PRETRAINED_MODEL_ID,
        "--model.tokenizer_path",
        PRETRAINED_MODEL_ID,
    ]
    result = subprocess.run(cmd, check=True)
    assert result.returncode == 0
