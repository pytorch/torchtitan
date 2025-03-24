# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# torchrun --standalone --nproc-per-node 4 generate.py

from dataclasses import dataclass

import torch
import torch.distributed as dist

from checkpoint import load_weights_from_hf
from model import DeepseekForCausalLM
from model_config import deepseek_config_registry
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
from transformers import AutoTokenizer

# Uncomment the model you want to run.
model_id, mesh_shape = "deepseek-ai/DeepSeek-V2-Lite-Chat", (2, 2)
# model_id, mesh_shape = "deepseek-ai/deepseek-v3", (8, 4)


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
    model_args = deepseek_config_registry[model_id]
    model_args.ep_size = dist_config.ep_size
    model_args.num_stages = dist_config.pp_size
    model_args.stage_idx = dist_config.pp_rank
    model_args.max_seq_len = 16384

    with dist_config.device, dist_config.mesh:
        model = DeepseekForCausalLM(model_args)
    load_weights_from_hf(model, model_id, dist_config.device)
    model.eval()
    model.setup_symm_mem(
        torch.bfloat16, dist_config.device, microbatches=dist_config.pp_size
    )

    return model, PipelineStage(
        model,
        dist_config.pp_rank,
        dist_config.pp_size,
        dist_config.device,
        group=dist_config.pp_mesh.get_group(),
    )


def generate(mesh: DeviceMesh, messages: list[dict], n_tokens: int = 10):
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

    model, stage = create_model(dist_config)
    pp_schedule = ScheduleGPipe(stage, dist_config.pp_size)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    x = tokenizer.apply_chat_template(
        [messages] * dist_config.pp_size,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    next_idx = x.shape[-1]
    x = torch.cat([x, torch.zeros(x.shape[0], n_tokens, dtype=torch.int64)], dim=-1)
    x = x.to(device)

    for _ in range(n_tokens):
        if dist_config.pp_size > 1:
            if dist_config.pp_rank == 0:
                pp_schedule.step(x)
                torch.distributed.broadcast(
                    x,
                    group=dist_config.pp_mesh.get_group(),
                    group_src=dist_config.pp_size - 1,
                )
            elif dist_config.pp_rank == dist_config.pp_size - 1:
                preds = pp_schedule.step()
                next_token = torch.argmax(preds[:, next_idx - 1], dim=-1)
                x[:, next_idx] = next_token
                torch.distributed.broadcast(
                    x,
                    group=dist_config.pp_mesh.get_group(),
                    group_src=dist_config.pp_size - 1,
                )
            else:
                pp_schedule.step()
                torch.distributed.broadcast(
                    x,
                    group=dist_config.pp_mesh.get_group(),
                    group_src=dist_config.pp_size - 1,
                )

            next_idx += 1
        else:
            preds = model(x)
            next_token = torch.argmax(preds[:, next_idx - 1], dim=-1)
            x[:, next_idx] = next_token
            next_idx += 1
    if rank == 0:
        print(tokenizer.decode(x[0]))


if __name__ == "__main__":
    mesh = dist.init_device_mesh("cuda", mesh_shape, mesh_dim_names=("pp", "ep"))
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    generate(mesh, messages)
    dist.destroy_process_group()
