# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from torchtitan.config_manager import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.train_llama_hf.hf_weights_utils import (
    load_sharded_state_dict_for_model_from_hf,
    normalize_state_dict_key,
)

from torchtitan.experiments.train_llama_hf.model.parallelize_llama import (
    apply_fsdp,
    apply_tp,
)
from torchtitan.experiments.train_llama_hf.model.pipeline_llama import (
    pipeline_llama_manual_split,
)


def main(job_config: JobConfig):
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    parallel_dims = ParallelDims(
        dp_shard=job_config.training.data_parallel_shard_degree,
        dp_replicate=job_config.training.data_parallel_replicate_degree,
        cp=job_config.experimental.context_parallel_degree,
        tp=job_config.training.tensor_parallel_degree,
        pp=job_config.experimental.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=not job_config.training.disable_loss_parallel,
    )
    world_mesh = parallel_dims.build_mesh(device_type="cuda")

    model_config = AutoConfig.from_pretrained(job_config.model.flavor)

    # load model
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(model_config)

    with torch.device("cpu"):
        gold_model_state_dict = AutoModelForCausalLM.from_pretrained(
            job_config.model.flavor
        ).state_dict()
    # apply parallelisms
    if parallel_dims.pp_enabled:
        # apply PT-D Pipeline Parallel
        _, model_parts = pipeline_llama_manual_split(
            model,
            world_mesh["pp"],
            parallel_dims,
            job_config,
            device,
            model_config,
        )
    else:
        model_parts = [model]
    for m in model_parts:
        if parallel_dims.tp_enabled:
            apply_tp(
                m,
                world_mesh["tp"],
                loss_parallel=False,
                enable_float8=False,
                enable_async_tp=False,
            )
        if parallel_dims.dp_shard_enabled:
            apply_fsdp(
                m,
                world_mesh["dp_shard"],
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                pp_enabled=False,
                cpu_offload=False,
                reshard_after_forward_policy="default",
            )

        m.to_empty(device="cuda")
        # load weights
        with torch.no_grad():
            load_sharded_state_dict_for_model_from_hf(job_config.model.flavor, m)
        for k, v in m.state_dict().items():
            if isinstance(v, torch.distributed.tensor.DTensor):
                full_tensor = v.full_tensor().to("cpu")
            else:
                full_tensor = v.to("cpu")
            k = normalize_state_dict_key(k)
            gt_value = gold_model_state_dict[k]
            assert torch.allclose(full_tensor, gt_value), f"tensor mismatch for {k}"


if __name__ == "__main__":
    job_config = JobConfig()
    job_config.parse_args()
    main(job_config)
