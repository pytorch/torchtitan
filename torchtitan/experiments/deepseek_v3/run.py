# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# torchrun --standalone --nproc-per-node 8 run.py
import torch
import torch.distributed as dist
from checkpoint import load_weights_from_hf
from model import DeepseekForCausalLM
from model_config import deepseek_config_registry

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.pipelining import PipelineStage, Schedule1F1B


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
    # [Note]: I am making the model smaller for testing / avoiding OOM. If you
    # have sufficient GPUs for model parallelism, you can remove this line.
    model_args.num_hidden_layers = 16

    # Apply model parallelism
    model_args.ep_size = ep_size
    model_args.num_stages = pp_size
    model_args.stage_idx = pp_rank
    print(model_args)

    # Instantiate model
    with device, mesh:
        model = DeepseekForCausalLM(model_args)

    # Load weights
    load_weights_from_hf(model, model_id, device)
    model.train()

    # Apply data parallelism
    fsdp_mesh = mesh["fsdp"]
    hsdp_mesh = mesh["ep", "fsdp"]
    # Using `reshard_after_forward=False` to implement Zero-2, i.e. sharding the
    # optimizer (Zero-1) and gradients (Zero-2), but not the model weights.
    # Reason: the MoE is "sparsely activated" compared to the dense model, thus
    # it will be ineconomical re-gather the weights.
    for layer in model.model.layers.values():
        # Apply FSDP to experts
        if hasattr(layer.mlp, "experts"):
            for expert in layer.mlp.experts.values():
                fully_shard(expert, mesh=fsdp_mesh, reshard_after_forward=False)
        # Apply HSDP to other parts such as attention, layernorm, because they
        # are doing DDP on EP dimension
        fully_shard(layer, mesh=hsdp_mesh, reshard_after_forward=False)

    # Apply HSDP on root model (lm_head, embeddings, etc)
    fully_shard(model, mesh=hsdp_mesh, reshard_after_forward=False)

    # Synthetic setting
    microbatches = pp_size * 2
    bs = 4  # microbatch size
    seqlen = 128

    # Use Symmetric Memory for MoE token shuffle. The number of tokens in each
    # buffer would be microbatch size * seq_len, i.e. flattened.
    model.setup_symm_mem(bs * seqlen, torch.bfloat16, device, microbatches)

    # Example inputs
    torch.manual_seed(ep_rank)
    x = torch.randint(model_args.vocab_size, (microbatches * bs, seqlen), device=device)
    label = torch.rand(microbatches * bs, seqlen, model_args.vocab_size, device=device)

    # Create loss function
    loss_fn = torch.nn.functional.cross_entropy

    # Run forward and backward
    if pp_size > 1:
        # Create pipeline stage
        stage = PipelineStage(
            model,
            pp_rank,
            pp_size,
            device,
            group=pp_mesh.get_group(),
        )

        # Create pipeline schedule
        losses = []
        pp_schedule = Schedule1F1B(stage, microbatches, loss_fn=loss_fn)

        if pp_rank == 0:
            y = pp_schedule.step(x)
        elif pp_rank == pp_size - 1:
            y = pp_schedule.step(target=label, losses=losses)
            loss = torch.mean(torch.stack(losses))
        else:
            pp_schedule.step()
    else:
        y = model(x)
        loss = loss_fn(y, label)
        loss.backward()

    if pp_rank == pp_size - 1:
        print(f"logits: {y.shape}")
        print(f"{loss=}")

    if pp_rank == 0:
        param = model.get_parameter("model.layers.0.self_attn.q_proj.weight")
        print(f"{torch.linalg.norm(param.grad)=}")

    print("Backward done")


if __name__ == "__main__":
    mesh = dist.init_device_mesh("cuda", (2, 2, 2), mesh_dim_names=("pp", "ep", "fsdp"))

    run_full_model(mesh)

    dist.destroy_process_group()
