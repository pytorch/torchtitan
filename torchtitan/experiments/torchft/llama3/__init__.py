# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.experiments.torchft.config.job_config import FaultTolerantModelSpec
from torchtitan.experiments.torchft.diloco import fragment_llm
from torchtitan.models.llama3 import (
    llama3_configs,
    Llama3StateDictAdapter,
    parallelize_llama,
)


def model_registry(
    flavor: str,
    *,
    seq_len: int | None = None,
    attn_backend: str = "flex",
) -> FaultTolerantModelSpec:
    get_config, max_context_len = llama3_configs[flavor]
    context_len = seq_len or max_context_len
    if context_len > max_context_len:
        raise ValueError(
            f"Requested seq_len {context_len} exceeds max context length "
            f"{max_context_len} for flavor {flavor}"
        )
    config = get_config(attn_backend=attn_backend, seq_len=context_len)
    return FaultTolerantModelSpec(
        name="torchft/llama3",
        flavor=flavor,
        model=config,
        max_context_length=context_len,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=Llama3StateDictAdapter,
        fragment_fn=fragment_llm,
    )
