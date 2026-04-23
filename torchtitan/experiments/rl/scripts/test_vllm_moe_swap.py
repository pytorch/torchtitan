"""
Test: Replace TorchTitan's MoE with vLLM's FusedMoE in the EP inference path.
If this produces correct output, the bug is in TorchTitan's MoE/dispatcher.
If still garbage, the bug is in attention or other components.

torchrun --nproc_per_node=2 torchtitan/experiments/rl/scripts/test_vllm_moe_swap.py
"""
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_MOE_BACKEND"] = "triton"

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from dataclasses import replace
from torchtitan.models.qwen3 import model_registry
from torchtitan.experiments.rl.models.vllm_registry import (
    register_model_to_vllm_model_registry,
    VLLM_MODEL_NAME,
)
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock
from vllm.config import get_current_vllm_config, set_current_vllm_config

model_path = "/data/users/jianiw/model/Qwen3-30B-A3B"

model_spec = model_registry("30B-A3B", attn_backend="varlen", moe_comm_backend="standard")
register_model_to_vllm_model_registry(model_spec)

engine = LLMEngine.from_engine_args(EngineArgs(
    model=model_path, trust_remote_code=True, dtype="bfloat16",
    tensor_parallel_size=2, enable_expert_parallel=True,
    distributed_executor_backend="external_launcher",
    gpu_memory_utilization=0.40, enforce_eager=True,
    hf_overrides={"architectures": [VLLM_MODEL_NAME]},
    attention_backend="CUSTOM",
))

rank = dist.get_rank()

# Get the TorchTitan model
tt_model = engine.model_executor.driver_worker.model_runner.model.model

# Get vllm_config from the engine's model runner
vllm_config = engine.model_executor.driver_worker.model_runner.vllm_config

for layer_idx, layer in enumerate(tt_model.layers.values()):
    if not layer.moe_enabled:
        continue

    # Create vLLM MoE block within the vllm config context
    with set_current_vllm_config(vllm_config):
        vllm_moe = Qwen3MoeSparseMoeBlock(
            vllm_config=vllm_config,
            prefix=f"model.layers.{layer_idx}.mlp",
        ).to("cuda", dtype=torch.bfloat16)

    # First, list vLLM MoE params to understand the structure
    if layer_idx == 0 and rank == 0:
        w13 = vllm_moe.experts.w13_weight
        w2 = vllm_moe.experts.w2_weight
        print(f"[SWAP] w13: type={type(w13).__name__}, shape={w13.shape}", flush=True)
        print(f"[SWAP] w2: type={type(w2).__name__}, shape={w2.shape}", flush=True)

    # Copy gate weight
    tt_gate = layer.moe.router.gate.weight
    gate_data = tt_gate._local_tensor if isinstance(tt_gate, DTensor) else tt_gate
    vllm_moe.gate.weight.data.copy_(gate_data)

    # Copy expert weights
    tt_w1 = layer.moe.experts.w1
    tt_w2 = layer.moe.experts.w2
    tt_w3 = layer.moe.experts.w3
    w1_data = tt_w1._local_tensor if isinstance(tt_w1, DTensor) else tt_w1
    w2_data = tt_w2._local_tensor if isinstance(tt_w2, DTensor) else tt_w2
    w3_data = tt_w3._local_tensor if isinstance(tt_w3, DTensor) else tt_w3

    # Find the w13 and w2 weight tensors in vLLM's MoE
    w13 = vllm_moe.experts.w13_weight
    w2_vllm = vllm_moe.experts.w2_weight

    hidden = w1_data.shape[1]
    # w13: [num_local_experts, 2*hidden, dim]
    # gate_proj (w1) goes first, up_proj (w3) goes second
    w13.data[:, :hidden, :] = w1_data
    w13.data[:, hidden:, :] = w3_data
    w2_vllm.data.copy_(w2_data)

    # Now monkey-patch the transformer block to use vLLM's MoE
    # The block's forward: x = x + self.moe(self.ffn_norm(x))
    # We need to: convert DTensor→plain, call vllm_moe, convert back
    original_moe = layer.moe

    class VLLMMoEWrapper(torch.nn.Module):
        def __init__(self, vllm_moe_block):
            super().__init__()
            self.vllm_moe = vllm_moe_block

        def forward(self, x):
            # x is DTensor [bs, slen, dim] from ffn_norm
            if isinstance(x, DTensor):
                x_local = x.to_local()
            else:
                x_local = x

            bs, slen, dim = x_local.shape
            x_2d = x_local.view(-1, dim)

            # Run vLLM's MoE
            out = self.vllm_moe(x_2d)

            out = out.view(bs, slen, dim)

            # Convert back to DTensor if needed
            if isinstance(x, DTensor):
                out = DTensor.from_local(out, x.device_mesh, x.placements)

            return out

    layer.moe = VLLMMoEWrapper(vllm_moe)
    if rank == 0 and layer_idx == 0:
        print(f"[SWAP] Replaced layer {layer_idx} MoE with vLLM FusedMoE", flush=True)

if rank == 0:
    print(f"[SWAP] All {sum(1 for l in tt_model.layers.values() if l.moe_enabled)} MoE layers swapped", flush=True)

# Run inference
prompts = ["The capital of France is"]
for i, p in enumerate(prompts):
    engine.add_request(str(i), p, SamplingParams(temperature=0.0, max_tokens=20))

while engine.has_unfinished_requests():
    outputs = engine.step()
    for o in outputs:
        if o.finished and rank == 0:
            print(f"Output: {o.outputs[0].text!r}", flush=True)
            print(f"Tokens: {list(o.outputs[0].token_ids)}", flush=True)

dist.barrier()
