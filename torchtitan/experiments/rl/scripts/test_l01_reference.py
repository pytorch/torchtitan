"""
Run layers 0-1 on a SINGLE GPU (no TP, no DTensor) to get reference MoE L01 norm.

torchrun --nproc_per_node=1 test_l01_reference.py
"""
import torch
import torch.distributed as dist

dist.init_process_group()
rank = dist.get_rank()
torch.cuda.set_device(rank)

from torchtitan.models.qwen3 import model_registry
import torch.distributed.checkpoint as dcp

model_path = "/data/users/jianiw/model/Qwen3-30B-A3B"
spec = model_registry("30B-A3B")

# Build full model (no TP) — use float32 on CPU to avoid OOM
model = spec.model.build().to("cpu", dtype=torch.float32)

adapter = spec.state_dict_adapter(spec.model, None)
storage_reader = adapter.get_hf_storage_reader(model_path)
hf_sd = adapter.to_hf(model.state_dict())
hf_keys = set(storage_reader.read_metadata().state_dict_metadata.keys())
hf_sd = {k: v for k, v in hf_sd.items() if k in hf_keys}
dcp.load(hf_sd, storage_reader=storage_reader)
tt_sd = adapter.from_hf(hf_sd)
model.load_state_dict(tt_sd, strict=False)
model.eval()

print("[REF] Model loaded (single GPU, no TP)", flush=True)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokens = tokenizer.encode("The capital of France is")
input_ids = torch.tensor([tokens], device="cpu")

# Hooks to capture MoE output at each layer
_moe_outs = {}

def make_hook(idx):
    def hook(mod, inp, out):
        _moe_outs[idx] = out
    return hook

handles = []
for i, layer in enumerate(model.layers.values()):
    if layer.moe_enabled and i < 5:
        handles.append(layer.moe.register_forward_hook(make_hook(i)))

with torch.no_grad():
    h = model.tok_embeddings(input_ids)
    freqs_cis = model.freqs_cis[:len(tokens)]
    for i, layer in enumerate(model.layers.values()):
        if i >= 5:
            break
        h = layer(h, freqs_cis, attention_masks=None)
        moe_out = _moe_outs.get(i)
        moe_norm = moe_out.float().norm().item() if moe_out is not None else -1
        print(
            f"[REF L{i:02d}] h_norm={h.float().norm():10.4f} "
            f"moe_norm={moe_norm:10.4f}",
            flush=True,
        )

for handle in handles:
    handle.remove()

# Final logits from full model
with torch.no_grad():
    h_full = model.tok_embeddings(input_ids)
    for layer in model.layers.values():
        h_full = layer(h_full, freqs_cis, attention_masks=None)
    h_full = model.norm(h_full)
    logits = model.output(h_full)

argmax = logits[0, -1].argmax().item()
top5 = logits[0, -1].topk(5)
print(f"\n[REF] argmax={argmax} ({tokenizer.decode([argmax])})", flush=True)
print(f"[REF] top5={[(t.item(), tokenizer.decode([t.item()])) for t in top5.indices]}", flush=True)

dist.barrier()
