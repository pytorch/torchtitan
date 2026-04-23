"""
Test: pretraining forward with disable_loss_parallel=True and NO MoE TP.
Skip apply_moe_ep_tp entirely — experts stay as plain unsharded params.

torchrun --nproc_per_node=4 test_pretrain_no_moe_tp.py
"""
import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor

dist.init_process_group()
rank = dist.get_rank()
torch.cuda.set_device(rank)

from torchtitan.models.qwen3 import model_registry
from torchtitan.models.qwen3.parallelize import apply_non_moe_tp
from torchtitan.config import (
    ActivationCheckpointConfig, CompileConfig, ParallelismConfig, TrainingConfig,
)
from torchtitan.distributed import ParallelDims
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions

model_path = "/data/users/jianiw/model/Qwen3-30B-A3B"
spec = model_registry("30B-A3B")

parallel_dims = ParallelDims(
    dp_replicate=1, dp_shard=1, cp=1, tp=4, pp=1, ep=1, etp=1, world_size=4,
)
parallel_dims.build_mesh()

with torch.device("meta"):
    model = spec.model.build()

# Apply ONLY non-MoE TP (attention, norms, embeddings, output)
# Skip apply_moe_ep_tp entirely — MoE experts stay unsharded
tp_mesh = parallel_dims.get_mesh("tp")
apply_non_moe_tp(
    model, tp_mesh,
    enable_loss_parallel=False,  # disable loss parallel
    enable_float8_tensorwise_tp=False,
    enable_async_tp=False,
    enable_cp=False,
    enable_sp=True,
)

model.to_empty(device="cuda")
model = model.to(dtype=torch.bfloat16)

if rank == 0:
    total_gb = sum(
        (p._local_tensor.numel() * p._local_tensor.element_size() if isinstance(p, DTensor)
         else p.numel() * p.element_size())
        for p in model.parameters()
    ) / 1e9
    print(f"[PT] Model size: {total_gb:.2f} GB", flush=True)

# Load weights
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
    print("[PT] Weights loaded", flush=True)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokens = tokenizer.encode("The capital of France is")
input_ids = torch.tensor([tokens], device="cuda")

if rank == 0:
    print(f"[PT] Input: {tokens}", flush=True)

with torch.inference_mode():
    h = model.tok_embeddings(input_ids)
    freqs_cis = model.freqs_cis[:len(tokens)].to("cuda")
    for layer in model.layers.values():
        h = layer(h, freqs_cis, attention_masks=None)
    h = model.norm(h)
    logits = model.output(h)
    if isinstance(logits, DTensor):
        logits = logits.full_tensor()

if rank == 0:
    argmax = logits[0, -1].argmax().item()
    top5 = logits[0, -1].topk(5)
    print(f"[PT] Argmax={argmax} ({tokenizer.decode([argmax])})", flush=True)
    print(f"[PT] Top5: {[(t.item(), tokenizer.decode([t.item()])) for t in top5.indices]}", flush=True)
    print(f"[PT] Top5 logits: {top5.values.tolist()}", flush=True)

dist.barrier()
