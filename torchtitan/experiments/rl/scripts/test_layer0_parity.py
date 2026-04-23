"""
Compare layer 0 output: TP=4 parallel model vs single-GPU reference.
Uses the same weights loaded directly from safetensors (no adapter).

torchrun --nproc_per_node=4 test_layer0_parity.py
"""
import os, json, torch, torch.distributed as dist
from torch.distributed._tensor import DTensor
from safetensors.torch import load_file

dist.init_process_group()
rank = dist.get_rank()
torch.cuda.set_device(rank)

from torchtitan.models.qwen3 import model_registry
from torchtitan.models.qwen3.parallelize import parallelize_qwen3, apply_non_moe_tp
from torchtitan.config import (
    ActivationCheckpointConfig, CompileConfig, ParallelismConfig, TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import ModelConvertersContainer
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions

model_path = "/data/users/jianiw/model/Qwen3-30B-A3B"
spec = model_registry("30B-A3B")

# --- TP=4 Model ---
parallel_dims = ParallelDims(
    dp_replicate=1, dp_shard=1, cp=1, tp=4, pp=1, ep=1, etp=1, world_size=4,
)
parallel_dims.build_mesh()

with torch.device("meta"):
    model = spec.model.build()

parallelize_qwen3(
    model, parallel_dims=parallel_dims, training=TrainingConfig(),
    model_converters=ModelConvertersContainer.Config(),
    parallelism=ParallelismConfig(
        tensor_parallel_degree=4, enable_sequence_parallel=True,
        disable_loss_parallel=True,
    ),
    compile_config=CompileConfig(),
    ac_config=ActivationCheckpointConfig(), dump_folder="",
    inference=True,
)
model.to_empty(device="cuda")
model = model.to(dtype=torch.bfloat16)

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
    print("[TP4] Model loaded", flush=True)

# --- Forward layer 0 only ---
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokens = tokenizer.encode("The capital of France is")
input_ids = torch.tensor([tokens], device="cuda")

with torch.no_grad():
    h = model.tok_embeddings(input_ids)
    freqs_cis = model.freqs_cis[:len(tokens)].to("cuda")

    # Get embedding in full
    if isinstance(h, DTensor):
        h_emb_full = h.full_tensor()
    else:
        h_emb_full = h

    # Run layer 0
    layer0 = list(model.layers.values())[0]
    h = layer0(h, freqs_cis, attention_masks=None)

    if isinstance(h, DTensor):
        h_l0_full = h.full_tensor()
    else:
        h_l0_full = h

if rank == 0:
    print(f"[TP4] Embedding: shape={h_emb_full.shape} norm={h_emb_full.float().norm():.4f}", flush=True)
    print(f"[TP4] Layer 0 output: shape={h_l0_full.shape} norm={h_l0_full.float().norm():.4f}", flush=True)
    print(f"[TP4] Layer 0 [0,0,:5]={h_l0_full[0,0,:5].tolist()}", flush=True)
    print(f"[TP4] Layer 0 [0,-1,:5]={h_l0_full[0,-1,:5].tolist()}", flush=True)

# Free TP model to make room for reference
if rank == 0:
    torch.save(h_l0_full.cpu(), "/tmp/tp4_layer0.pt")
    torch.save(h_emb_full.cpu(), "/tmp/tp4_emb.pt")
del model, h, h_emb_full, h_l0_full
torch.cuda.empty_cache()

# --- Reference: load the same model without TP on rank 0 ---
# All ranks participate in dcp.load (collective), but only rank 0 uses the result
ref_model = spec.model.build().to("cuda", dtype=torch.bfloat16)

ref_adapter = spec.state_dict_adapter(spec.model, None)
ref_reader = ref_adapter.get_hf_storage_reader(model_path)
ref_hf_sd = ref_adapter.to_hf(ref_model.state_dict())
ref_hf_keys = set(ref_reader.read_metadata().state_dict_metadata.keys())
ref_hf_sd = {k: v for k, v in ref_hf_sd.items() if k in ref_hf_keys}
dcp.load(ref_hf_sd, storage_reader=ref_reader)
ref_tt_sd = ref_adapter.from_hf(ref_hf_sd)
ref_model.load_state_dict(ref_tt_sd, strict=False)
ref_model.eval()

if rank == 0:
    print("[REF] Reference model loaded (no TP)", flush=True)

    with torch.no_grad():
        ref_h = ref_model.tok_embeddings(input_ids)
        ref_freqs = ref_model.freqs_cis[:len(tokens)].to("cuda")
        ref_l0 = list(ref_model.layers.values())[0]
        ref_h = ref_l0(ref_h, ref_freqs, attention_masks=None)

    print(f"[REF] Layer 0 output: shape={ref_h.shape} norm={ref_h.float().norm():.4f}", flush=True)
    print(f"[REF] Layer 0 [0,0,:5]={ref_h[0,0,:5].tolist()}", flush=True)
    print(f"[REF] Layer 0 [0,-1,:5]={ref_h[0,-1,:5].tolist()}", flush=True)

    # Compare with saved TP4 output
    h_l0_full = torch.load("/tmp/tp4_layer0.pt").to("cuda")
    diff = (h_l0_full.float() - ref_h.float()).abs()
    print(f"\n[COMPARE] Layer 0 max_diff={diff.max().item():.6f} avg_diff={diff.mean().item():.8f}", flush=True)
    if diff.max().item() < 0.01:
        print("[COMPARE] PASS — TP=4 matches reference", flush=True)
    else:
        print("[COMPARE] FAIL — TP=4 diverges from reference", flush=True)
        # Show where the max diff is
        max_pos = diff.argmax()
        print(f"[COMPARE] Max diff at flat index {max_pos.item()}", flush=True)

del ref_model
torch.cuda.empty_cache()
dist.barrier()
