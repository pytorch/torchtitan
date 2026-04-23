"""
Replace TorchTitan's MoE module with vLLM's FusedMoE at the model level.

Strategy:
1. After TorchTitan model is built and weights loaded
2. For each MoE layer, create a vLLM FusedMoE
3. Copy weights from TorchTitan format to vLLM format
4. Replace the MoE forward to use FusedMoE
5. The PrepareModuleInputOutput hooks still handle DTensor boundaries

torchrun --nproc_per_node=4 test_vllm_fused_moe_replace.py
"""
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_MOE_BACKEND"] = "triton"

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed._tensor import DTensor, Partial, Replicate
from torchtitan.models.qwen3 import model_registry
from torchtitan.experiments.rl.models.vllm_registry import (
    register_model_to_vllm_model_registry, VLLM_MODEL_NAME,
)
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.fused_moe import FusedMoE

model_path = "/data/users/jianiw/model/Qwen3-30B-A3B"
model_spec = model_registry("30B-A3B", attn_backend="varlen")
register_model_to_vllm_model_registry(model_spec)

engine = LLMEngine.from_engine_args(EngineArgs(
    model=model_path, trust_remote_code=True, dtype="bfloat16",
    tensor_parallel_size=4, distributed_executor_backend="external_launcher",
    gpu_memory_utilization=0.95, max_model_len=256, enforce_eager=True,
    hf_overrides={"architectures": [VLLM_MODEL_NAME]},
    attention_backend="CUSTOM",
))

rank = dist.get_rank()
model = engine.model_executor.driver_worker.model_runner.model.model
vllm_config = engine.model_executor.driver_worker.model_runner.vllm_config

# For each MoE layer, create vLLM FusedMoE and copy weights
with set_current_vllm_config(vllm_config):
    for layer_idx, layer in enumerate(model.layers.values()):
        if not layer.moe_enabled:
            continue

        # Get TorchTitan MoE weights
        tt_moe = layer.moe
        gate_w = tt_moe.router.gate.weight
        gate_data = gate_w._local_tensor if isinstance(gate_w, DTensor) else gate_w

        w1 = tt_moe.experts.w1
        w2 = tt_moe.experts.w2
        w3 = tt_moe.experts.w3
        w1_data = w1._local_tensor if isinstance(w1, DTensor) else w1
        w2_data = w2._local_tensor if isinstance(w2, DTensor) else w2
        w3_data = w3._local_tensor if isinstance(w3, DTensor) else w3

        num_experts = w1_data.shape[0]
        hidden = w1_data.shape[1]
        dim = w1_data.shape[2]

        # Create vLLM FusedMoE — pass FULL intermediate_size (not TP-sharded)
        # FusedMoE handles TP sharding internally
        full_hidden = 768  # Qwen3-30B-A3B moe_intermediate_size
        vllm_moe = FusedMoE(
            num_experts=128,
            top_k=8,
            hidden_size=2048,
            intermediate_size=full_hidden,
            reduce_results=True,
            renormalize=True,
            prefix=f"model.layers.{layer_idx}.mlp",
        ).to(f"cuda:{rank}", dtype=torch.bfloat16)

        # FusedMoE internally divides by TP, so w13 is [num_local_experts, 2*hidden/TP, dim]
        local_hidden = vllm_moe.w13_weight.shape[1] // 2
        # Copy weights: the TT weights are already TP-sharded (192 = 768/4)
        vllm_moe.w13_weight.data[:, :local_hidden, :] = w1_data
        vllm_moe.w13_weight.data[:, local_hidden:, :] = w3_data
        vllm_moe.w2_weight.data.copy_(w2_data)

        # Create a wrapper that replaces TorchTitan's MoE.forward
        class FusedMoEWrapper(nn.Module):
            def __init__(self, vllm_fused_moe, gate_weight):
                super().__init__()
                self.fused_moe = vllm_fused_moe
                self.gate_w = gate_weight

            def forward(self, x):
                # Strip DTensor
                if isinstance(x, DTensor):
                    mesh = x.device_mesh
                    placements = x.placements
                    x_local = x.to_local()
                    is_dtensor = True
                else:
                    is_dtensor = False
                    x_local = x

                bs, slen, dim = x_local.shape
                x_2d = x_local.view(-1, dim)

                # Router
                router_logits = x_2d @ self.gate_w.t()

                # FusedMoE forward
                out = self.fused_moe(x_2d, router_logits)

                out = out.view(bs, slen, -1)

                # Wrap back to DTensor
                if is_dtensor:
                    # Output is already all-reduced by FusedMoE (reduce_results=True)
                    # so it's Replicate, not Partial
                    out = DTensor.from_local(out, mesh, (Replicate(),), run_check=False)

                return out

        # Trigger kernel init with a dummy forward pass using vLLM's context
        from vllm.forward_context import set_forward_context
        with torch.no_grad(), set_forward_context(None, vllm_config):
            dummy_x = torch.randn(2, 2048, device=f"cuda:{rank}", dtype=torch.bfloat16)
            dummy_logits = torch.randn(2, 128, device=f"cuda:{rank}", dtype=torch.bfloat16)
            try:
                _ = vllm_moe(dummy_x, dummy_logits)
            except Exception as e:
                if rank == 0:
                    print(f"[REPLACE] Kernel init: {e}", flush=True)

        wrapper = FusedMoEWrapper(vllm_moe, gate_data)
        layer.moe = wrapper

        if layer_idx == 0 and rank == 0:
            print(f"[REPLACE] Layer 0: vLLM FusedMoE w13={vllm_moe.w13_weight.shape}, w2={vllm_moe.w2_weight.shape}", flush=True)

if rank == 0:
    print(f"[REPLACE] All MoE layers replaced with vLLM FusedMoE", flush=True)

# Run inference
prompts = ["The capital of France is", "In the year 2050, artificial intelligence will"]
for i, p in enumerate(prompts):
    engine.add_request(str(i), p, SamplingParams(temperature=0.0, max_tokens=20))

while engine.has_unfinished_requests():
    for o in engine.step():
        if o.finished and rank == 0:
            print(f"Prompt: {o.prompt!r}", flush=True)
            print(f"Output: {o.outputs[0].text!r}", flush=True)
            print(flush=True)

dist.barrier()
