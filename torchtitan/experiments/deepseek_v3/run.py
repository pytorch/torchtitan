# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# torchrun --standalone --nproc-per-node 4 run.py

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

from checkpoint import load_weights_from_hf
from model import DeepseekForCausalLM
from model_config import deepseek_config_registry
from transformers import AutoTokenizer

model_id, model_path, bs, mesh_shape = "deepseek-ai/DeepSeek-V2-Lite", "/traindata/llama_hf_ckpt/DeepSeek-V2-Lite", 2, (2, 2)
# model_id, model_path, bs, mesh_shape = "deepseek-ai/deepseek-v3", "/traindata/llama_hf_ckpt/DeepSeek-V3-bf16", 8, (8, 4)

@dataclass
class DistConfig:
    mesh: DeviceMesh
    pp_mesh: DeviceMesh
    ep_mesh: DeviceMesh
    pp_size: int
    ep_size: int
    ep_rank: int
    pp_rank: int
    device: torch.device


def create_model(dist_config: DistConfig):
    # Get model configs
    model_args = deepseek_config_registry[model_id]

    # Apply model parallelism
    model_args.ep_size = dist_config.ep_size
    model_args.num_stages = dist_config.pp_size
    model_args.stage_idx = dist_config.pp_rank
    model_args.max_seq_len = 16384
    # print(model_args)

    # Instantiate model
    with dist_config.device, dist_config.mesh:
        model = DeepseekForCausalLM(model_args)

    # Load weights
    load_weights_from_hf(model, model_path, dist_config.device)
    model.train()

    stage = None
    if dist_config.pp_size > 1:
        stage = PipelineStage(
            model,
            dist_config.pp_rank,
            dist_config.pp_size,
            dist_config.device,
            group=dist_config.pp_mesh.get_group(),
        )
    return stage


# Generate from the model.
def generate(mesh: DeviceMesh):
    rank = dist.get_rank()
    device_count = torch.cuda.device_count()
    device = torch.device("cuda", rank % device_count)

    dist_config = DistConfig(
        mesh=mesh,
        pp_mesh=mesh["pp"],
        ep_mesh=mesh["ep"],
        pp_rank=mesh["pp"].get_local_rank(),
        pp_size=mesh["pp"].size(),
        ep_size=mesh["ep"].size(),
        ep_rank=mesh["ep"].get_local_rank(),
        device=device,
    )

    stage = create_model(dist_config)
    pp_schedule = ScheduleGPipe(stage, bs)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    x = tokenizer.apply_chat_template([messages] * bs, add_generation_prompt=True, return_tensors="pt")
    next_idx = x.shape[-1]
    x = torch.cat([x, torch.zeros(x.shape[0], 10, dtype=torch.int64)], dim=-1)
    x = x.to(device)

    for _ in range(10):
        if dist_config.pp_rank == 0:
            pp_schedule.step(x)
            next_token = torch.zeros((x.shape[0],), dtype=torch.int64, device=device)
            torch.distributed.broadcast(
                next_token, 
                group=dist_config.pp_mesh.get_group(), 
                group_src=dist_config.pp_size - 1
            )
        elif dist_config.pp_rank == dist_config.pp_size - 1:
            preds = pp_schedule.step()
            next_token = torch.argmax(preds[:, next_idx], dim=-1)
            torch.distributed.broadcast(
                next_token, 
                group=dist_config.pp_mesh.get_group(), 
                group_src=dist_config.pp_rank
            )
        else:
            pp_schedule.step()
            next_token = torch.zeros((x.shape[0],), dtype=torch.int64, device=device)
            torch.distributed.broadcast(
                next_token, 
                group=dist_config.pp_mesh.get_group(), 
                group_src=dist_config.pp_rank -1
            )

        x[:, next_idx] = next_token
        next_idx += 1
    print(tokenizer.decode(x[0]))

# Run the full model.
def run_full_model(mesh: DeviceMesh):
    # TODO(eugen): Update.
    raise NotImplementedError("Removed for now because it contains internal data.")


if __name__ == "__main__":
    mesh = dist.init_device_mesh("cuda", mesh_shape, mesh_dim_names=("pp", "ep"))
    generate(mesh)
    dist.destroy_process_group()