# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# torchrun --standalone --nproc-per-node 4 run.py
import torch
import torch.distributed as dist
from checkpoint import load_weights_from_hf
from model import DeepseekForCausalLM
from model_config import deepseek_config_registry

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe


# Use DeepSeek-V2-Lite as a proxy
model_id = "deepseek-ai/DeepSeek-V2-Lite"


# Run full model
def run_full_model(
    mesh: DeviceMesh,
):
    rank = dist.get_rank()
    device_count = torch.cuda.device_count()
    device = torch.device("cuda", rank % device_count)

    pp_mesh = mesh["pp"]
    ep_mesh = mesh["ep"]
    pp_rank = pp_mesh.get_local_rank()
    ep_rank = ep_mesh.get_local_rank()
    pp_size = pp_mesh.size()
    ep_size = ep_mesh.size()

    # Get model configs
    model_args = deepseek_config_registry[model_id]

    # Apply model parallelism
    model_args.ep_size = ep_size
    model_args.num_stages = pp_size
    model_args.stage_idx = pp_rank
    print(model_args)

    # Instantiate model
    with device, mesh:
        model = DeepseekForCausalLM(model_args)
        model.eval()

    # Load weights
    load_weights_from_hf(model, model_id, device)

    # Example inputs
    bs = 2
    microbatches = 2
    seqlen = 128
    x = torch.randint(model_args.vocab_size, (bs, seqlen), device=device)

    # Create pipeline stage
    stage = PipelineStage(
        model,
        pp_rank,
        pp_size,
        device,
        group=pp_mesh.get_group(),
    )

    # Create pipeline schedule
    pp_schedule = ScheduleGPipe(stage, microbatches)

    # Run forward
    if pp_rank == 0:
        y = pp_schedule.step(x)
    else:
        y = pp_schedule.step()

    if pp_rank == pp_size - 1:
        print(y.shape)


if __name__ == "__main__":
    mesh = dist.init_device_mesh("cuda", (2, 2), mesh_dim_names=("pp", "ep"))

    with torch.no_grad():
        run_full_model(mesh)

    dist.destroy_process_group()