# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.5 context parallel: GDN Ulysses all-to-all matches a single GPU.

Run with ``torchrun --nproc_per_node=2 tests/unit_tests/gpu/test_qwen3_5_gdn_cp.py``.
"""

import os

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.config.transform import ContextParallelTransform
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.utils import get_spmd_context
from torchtitan.models.common.cp_attention import UlyssesCPFlexInnerAttention
from torchtitan.models.qwen3_5 import model_registry
from torchtitan.models.qwen3_5.sharding import set_qwen35_sharding_config
from torchtitan.tools.utils import set_default_dtype


def _tiny_config():
    config = model_registry("debugmodel", seq_len=128)
    config.vision_encoder = None
    config.vocab_size = 128
    config.tok_embeddings.num_embeddings = 128
    config.lm_head.out_features = 128
    # Layers 0-2 are GatedDeltaNet, layer 3 is full attention.
    config.layers = config.layers[:4]
    return config


def _plain_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state = {}
    for name, value in model.state_dict().items():
        tensor = value.full_tensor() if isinstance(value, DTensor) else value
        state[name] = tensor.detach()
    return state


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    parallelism = ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=1,
        context_parallel_degree=2,
        tensor_parallel_degree=1,
        pipeline_parallel_degree=1,
        context_parallel_load_balancer=None,
        enable_sequence_parallel=False,
    )
    parallel_dims = ParallelDims.from_config(parallelism, world_size=dist.get_world_size())
    training = TrainingConfig(dtype="bfloat16")

    cp_config = _tiny_config()
    ContextParallelTransform(inner_attention=UlyssesCPFlexInnerAttention).transform(
        cp_config
    )
    set_qwen35_sharding_config(cp_config, enable_sp=False, enable_ep=False)

    with torch.device("meta"), set_default_dtype(torch.bfloat16):
        cp_model = cp_config.build()
    cp_model = cp_model.parallelize(
        parallel_dims=parallel_dims,
        training=training,
        parallelism=parallelism,
        compile_config=None,
        ac_config=None,
        dump_folder=os.path.join("/tmp", "qwen35-gdn-cp"),
        skip_dp=True,
    )
    with get_spmd_context(parallel_dims=parallel_dims):
        cp_model.to_empty(device=device)
        with torch.no_grad():
            cp_model.init_weights()
        for param in cp_model.parameters():
            local = param.to_local() if isinstance(param, DTensor) else param
            dist.broadcast(local, src=0)
        cp_model.train()

    ref_config = _tiny_config()
    with set_default_dtype(torch.bfloat16):
        ref_model = ref_config.build().to(device)
    ref_model.load_state_dict(_plain_state_dict(cp_model))
    ref_model.train()

    seq = 32
    torch.manual_seed(0)
    tokens = torch.randint(0, 128, (seq,), device=device)
    dist.broadcast(tokens, src=0)
    positions = torch.arange(seq, device=device)
    ref_masks = ref_model.get_attention_masks(positions)
    with torch.no_grad():
        ref_logits = ref_model(tokens, attention_masks=ref_masks, positions=positions)

    with get_spmd_context(parallel_dims=parallel_dims):
        cp_inputs, _, cp_extra = cp_model.preprocess_inputs(
            {"input": tokens, "labels": tokens, "positions": positions},
            parallel_dims=parallel_dims,
            parallelism=parallelism,
        )
        cp_logits = cp_model(
            cp_inputs,
            attention_masks=cp_extra.get("attention_masks"),
            positions=cp_extra.get("positions"),
        )
        cp_logits.float().pow(2).mean().backward()

    gathered = [torch.empty_like(cp_logits.detach()) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, cp_logits.detach())
    cp_full = torch.cat(gathered, dim=0)
    torch.testing.assert_close(cp_full, ref_logits, atol=2e-2, rtol=2e-2)

    grad_ok = False
    for name, param in cp_model.named_parameters():
        if param.grad is None or "in_proj_q.weight" not in name:
            continue
        grad = param.grad.to_local() if isinstance(param.grad, DTensor) else param.grad
        grad_ok = grad.abs().sum().item() > 0
        break
    if not grad_ok:
        raise AssertionError("GDN in_proj_q did not receive a gradient")

    dist.barrier()
    if rank == 0:
        print(
            f"qwen3.5 cp=2 ok logits {tuple(cp_full.shape)} "
            f"max_abs {(cp_full - ref_logits).abs().max().item():.3e}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
