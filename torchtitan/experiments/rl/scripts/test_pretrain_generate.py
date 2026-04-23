"""
Test: Use core TorchTitan parallelize_qwen3 to load the 30B-A3B model
and generate 5 tokens autoregressively. No vLLM.

Checks if the model produces meaningful text or garbage.

torchrun --nproc_per_node=4 test_pretrain_generate.py
"""
import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor

dist.init_process_group()
rank = dist.get_rank()
torch.cuda.set_device(rank)

from torchtitan.models.qwen3 import model_registry
from torchtitan.models.qwen3.parallelize import parallelize_qwen3
from torchtitan.config import (
    ActivationCheckpointConfig, CompileConfig, ParallelismConfig, TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import ModelConvertersContainer
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions

model_path = "/data/users/jianiw/model/Qwen3-30B-A3B"
spec = model_registry("30B-A3B")

parallel_dims = ParallelDims(
    dp_replicate=1, dp_shard=1, cp=1, tp=4, pp=1, ep=1, etp=1, world_size=4,
)
parallel_dims.build_mesh()

# Disable loss parallel for correct logits
parallelism = ParallelismConfig(
    tensor_parallel_degree=4, enable_sequence_parallel=True,
    disable_loss_parallel=True,
)

with torch.device("meta"):
    model = spec.model.build()

parallelize_qwen3(
    model, parallel_dims=parallel_dims, training=TrainingConfig(),
    model_converters=ModelConvertersContainer.Config(),
    parallelism=parallelism, compile_config=CompileConfig(),
    ac_config=ActivationCheckpointConfig(), dump_folder="",
    inference=True,
)
model.to_empty(device="cuda")
model = model.to(dtype=torch.bfloat16)

if rank == 0:
    print("[GEN] Loading weights...", flush=True)

adapter = spec.state_dict_adapter(spec.model, None)
storage_reader = adapter.get_hf_storage_reader(model_path)
hf_sd = adapter.to_hf(model.state_dict())
hf_keys = set(storage_reader.read_metadata().state_dict_metadata.keys())
hf_sd = {k: v for k, v in hf_sd.items() if k in hf_keys}
dcp.load(hf_sd, storage_reader=storage_reader)
tt_sd = adapter.from_hf(hf_sd)
model_sd = dict(model.state_dict())
for name, tensor in tt_sd.items():
    if name in model_sd and isinstance(model_sd[name], DTensor):
        if not isinstance(tensor, DTensor):
            target = model_sd[name]
            tt_sd[name] = DTensor.from_local(
                tensor.to(target.device_mesh.device_type),
                device_mesh=target.device_mesh,
                placements=[torch.distributed.tensor.Replicate()],
            )
set_model_state_dict(model, tt_sd, options=StateDictOptions(strict=False))
model.eval()

if rank == 0:
    print("[GEN] Weights loaded. Generating...", flush=True)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
prompt = "The capital of France is"
tokens = tokenizer.encode(prompt)
input_ids = torch.tensor([tokens], device="cuda")

# Greedy autoregressive generation
generated = list(tokens)
with torch.no_grad():
    for step in range(10):
        seq = torch.tensor([generated], device="cuda")
        h = model.tok_embeddings(seq)
        freqs_cis = model.freqs_cis[:len(generated)].to("cuda")

        for layer in model.layers.values():
            h = layer(h, freqs_cis, attention_masks=None)

        h = model.norm(h)
        logits = model.output(h)

        if isinstance(logits, DTensor):
            logits = logits.full_tensor()

        # Greedy: take argmax of last position
        next_token = logits[0, -1].argmax().item()
        generated.append(next_token)

        if rank == 0:
            print(f"  step {step}: token={next_token} ({tokenizer.decode([next_token])!r})", flush=True)

if rank == 0:
    full_text = tokenizer.decode(generated)
    print(f"\n[GEN] Prompt: {prompt!r}", flush=True)
    print(f"[GEN] Generated: {full_text!r}", flush=True)

dist.barrier()
