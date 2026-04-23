"""
Trace attention output at each layer. Compare TorchTitan vs native vLLM.

Adds hooks to the Qwen3TransformerBlock to capture hidden states after
attention and after MoE/FFN at each layer for the first real inference call.

torchrun --nproc_per_node=4 test_attention_trace.py
"""
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_MOE_BACKEND"] = "triton"

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from torchtitan.models.qwen3 import model_registry
from torchtitan.experiments.rl.models.vllm_registry import (
    register_model_to_vllm_model_registry, VLLM_MODEL_NAME,
)
from vllm import EngineArgs, LLMEngine, SamplingParams

model_path = "/data/users/jianiw/model/Qwen3-30B-A3B"
model_spec = model_registry("30B-A3B", attn_backend="varlen")
register_model_to_vllm_model_registry(model_spec)

engine = LLMEngine.from_engine_args(EngineArgs(
    model=model_path, trust_remote_code=True, dtype="bfloat16",
    tensor_parallel_size=4,
    distributed_executor_backend="external_launcher",
    gpu_memory_utilization=0.95, max_model_len=256,
    enforce_eager=True,
    hf_overrides={"architectures": [VLLM_MODEL_NAME]},
    attention_backend="CUSTOM",
))

rank = dist.get_rank()
model = engine.model_executor.driver_worker.model_runner.model.model

# Instrument each transformer block to trace hidden states
_traces = {}
_fwd_count = [0]

from torchtitan.models.qwen3.model import Qwen3TransformerBlock

original_forward = Qwen3TransformerBlock.forward

def traced_forward(self, x, freqs_cis, attention_masks, positions=None):
    _fwd_count[0] += 1
    # Only trace the first real inference call (skip profiling)
    # Profiling uses large batch sizes; real inference uses small
    is_trace = False
    x_local = x._local_tensor if isinstance(x, DTensor) else x
    # Trace calls 3 and 4 (first real prefill + first decode)
    if _fwd_count[0] in range(3 * 48 + 1, 5 * 48 + 1):
        is_trace = True
        layer_idx = (_fwd_count[0] - 1) % 48
        call_idx = (_fwd_count[0] - 1) // 48

    if not is_trace:
        return original_forward(self, x, freqs_cis, attention_masks, positions)

    # --- Trace input ---
    x_in_local = x._local_tensor if isinstance(x, DTensor) else x
    x_in_sum = x_in_local.float().sum().item()

    # --- Run attention ---
    attn_out = self.attention(
        self.attention_norm(x), freqs_cis, attention_masks, positions
    )
    attn_local = attn_out._local_tensor if isinstance(attn_out, DTensor) else attn_out
    attn_sum = attn_local.float().sum().item()

    x = x + attn_out

    # --- Run MoE/FFN ---
    if self.moe_enabled:
        ffn_out = self.moe(self.ffn_norm(x))
    else:
        ffn_out = self.feed_forward(self.ffn_norm(x))

    ffn_local = ffn_out._local_tensor if isinstance(ffn_out, DTensor) else ffn_out
    ffn_sum = ffn_local.float().sum().item()

    x = x + ffn_out

    x_out_local = x._local_tensor if isinstance(x, DTensor) else x
    x_out_sum = x_out_local.float().sum().item()

    # Gather sums across ranks
    sums = torch.tensor([x_in_sum, attn_sum, ffn_sum, x_out_sum],
                        device=f"cuda:{rank}")
    all_sums = [torch.zeros_like(sums) for _ in range(4)]
    dist.all_gather(all_sums, sums)

    if rank == 0 and layer_idx in [0, 1, 23, 47]:
        in_match = len(set(f"{s[0].item():.4f}" for s in all_sums)) == 1
        attn_match = len(set(f"{s[1].item():.4f}" for s in all_sums)) == 1
        ffn_match = len(set(f"{s[2].item():.4f}" for s in all_sums)) == 1
        out_match = len(set(f"{s[3].item():.4f}" for s in all_sums)) == 1

        print(
            f"[ATTN] call={call_idx} layer={layer_idx} "
            f"in={'MATCH' if in_match else 'DIFF'} "
            f"attn={'MATCH' if attn_match else 'DIFF'} "
            f"ffn={'MATCH' if ffn_match else 'DIFF'} "
            f"out={'MATCH' if out_match else 'DIFF'} "
            f"| in_sums={[f'{s[0].item():.2f}' for s in all_sums]} "
            f"attn_sums={[f'{s[1].item():.2f}' for s in all_sums]} "
            f"ffn_sums={[f'{s[2].item():.2f}' for s in all_sums]} "
            f"out_sums={[f'{s[3].item():.2f}' for s in all_sums]}",
            flush=True,
        )

    return x

Qwen3TransformerBlock.forward = traced_forward

# Run inference
engine.add_request("0", "The capital of France is",
                   SamplingParams(temperature=0.0, max_tokens=5))

while engine.has_unfinished_requests():
    for o in engine.step():
        if o.finished and rank == 0:
            print(f"\nOutput: {o.outputs[0].text!r}", flush=True)
            print(f"Tokens: {list(o.outputs[0].token_ids)}", flush=True)

dist.barrier()
