"""Check if dense model's full_tensor works during 1-token decode."""
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_MOE_BACKEND"] = "triton"

import torch, torch.distributed as dist
from torch.distributed._tensor import DTensor
from dataclasses import replace
from torchtitan.models.qwen3 import model_registry
from torchtitan.experiments.rl.models.parallelize import parallelize_qwen3
from torchtitan.experiments.rl.plugin import register_model_to_vllm_model_registry, VLLM_MODEL_NAME
from torchtitan.experiments.rl.models.vllm_wrapper import TorchTitanVLLMModelWrapper
from vllm import EngineArgs, LLMEngine, SamplingParams

model_spec = model_registry("0.6B", attn_backend="varlen")
model_spec = replace(model_spec, parallelize_fn=parallelize_qwen3)

# Monkey-patch forward to log full_tensor behavior
orig_forward = TorchTitanVLLMModelWrapper.forward

def patched_forward(self, input_ids=None, positions=None, inputs_embeds=None, **kwargs):
    if not hasattr(self, "_pf_cnt"):
        self._pf_cnt = 0
    self._pf_cnt += 1

    # Call original forward but intercept before full_tensor
    if inputs_embeds is not None:
        raise NotImplementedError
    tokens_2d = input_ids.unsqueeze(0)
    h = self.model.tok_embeddings(tokens_2d)
    rope_cache = self.model.freqs_cis
    positions_2d = positions.unsqueeze(0)
    for layer in self.model.layers.values():
        h = layer(h, rope_cache, attention_masks=None, positions=positions_2d)
    h = self.model.norm(h)

    if 4 <= self._pf_cnt <= 6:
        h_local = h._local_tensor if isinstance(h, DTensor) else h
        print(
            f"[FT] fwd#{self._pf_cnt} rank={dist.get_rank()} "
            f"type={'DTensor' if isinstance(h, DTensor) else 'plain'} "
            f"local_shape={h_local.shape} "
            f"input_ids={input_ids.shape}",
            flush=True,
        )

    if isinstance(h, DTensor):
        h = h.full_tensor()

    if 4 <= self._pf_cnt <= 6:
        print(
            f"[FT] fwd#{self._pf_cnt} rank={dist.get_rank()} "
            f"after full_tensor shape={h.shape}",
            flush=True,
        )

    if h.dim() == 3:
        h = h.view(-1, h.size(-1))
    return h

TorchTitanVLLMModelWrapper.forward = patched_forward

register_model_to_vllm_model_registry(model_spec)

engine = LLMEngine.from_engine_args(EngineArgs(
    model="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
    trust_remote_code=True, dtype="bfloat16", tensor_parallel_size=2,
    distributed_executor_backend="external_launcher",
    gpu_memory_utilization=0.30, enforce_eager=True,
    hf_overrides={"architectures": [VLLM_MODEL_NAME]},
    attention_backend="CUSTOM",
))

engine.add_request("0", "Hello", SamplingParams(temperature=0.0, max_tokens=3))
while engine.has_unfinished_requests():
    for o in engine.step():
        if o.finished and dist.get_rank() == 0:
            print(f"Output: {o.outputs[0].text!r}", flush=True)
dist.barrier()
