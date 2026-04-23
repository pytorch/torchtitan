"""
Test core TorchTitan forward ONLY (no vLLM).
Loads the 30B-A3B model with pretraining parallelize_qwen3(inference=True),
runs a forward pass, and checks logits.

torchrun --nproc_per_node=4 test_pretrain_only.py
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

parallelism = ParallelismConfig(
    tensor_parallel_degree=4, enable_sequence_parallel=True,
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
    print("[PT] Model built and parallelized, loading weights...", flush=True)

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

# Forward pass
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokens = tokenizer.encode("The capital of France is")
input_ids = torch.tensor([tokens], device="cuda")

if rank == 0:
    print(f"[PT] Input: {tokens} ({tokenizer.decode(tokens)})", flush=True)

with torch.no_grad():
    h = model.tok_embeddings(input_ids)
    freqs_cis = model.freqs_cis[:len(tokens)].to("cuda")

    for i, layer in enumerate(model.layers.values()):
        h = layer(h, freqs_cis, attention_masks=None)

    h = model.norm(h)
    logits = model.output(h)

    if isinstance(logits, DTensor):
        logits_full = logits.full_tensor()
    else:
        logits_full = logits

    # Debug: check logits details
    logits_local = logits._local_tensor if isinstance(logits, DTensor) else logits
    logits_placement = logits.placements if isinstance(logits, DTensor) else "plain"
    print(
        f"[PT] rank={rank} logits_type={'DTensor' if isinstance(logits, DTensor) else 'plain'} "
        f"placement={logits_placement} local_shape={logits_local.shape} "
        f"full_shape={logits_full.shape}",
        flush=True,
    )
    # Check if full_tensor actually has different values across vocab
    last_logits = logits_full[0, -1]
    print(
        f"[PT] rank={rank} full logits stats: "
        f"min={last_logits.min().item():.4f} max={last_logits.max().item():.4f} "
        f"mean={last_logits.float().mean().item():.4f} std={last_logits.float().std().item():.4f}",
        flush=True,
    )

if rank == 0:
    global_argmax = logits_full[0, -1].argmax().item()
    top5 = logits_full[0, -1].topk(5)
    print(f"[PT] Global argmax={global_argmax} ({tokenizer.decode([global_argmax])})", flush=True)
    print(f"[PT] Top5: {[(t.item(), tokenizer.decode([t.item()])) for t in top5.indices]}", flush=True)
    print(f"[PT] Top5 logits: {top5.values.tolist()}", flush=True)

    # Also try with loss_parallel DISABLED
    print(f"\n[PT] Now testing with disable_loss_parallel=True...", flush=True)

# --- Test 2: Disable loss parallel ---
# Rebuild with disable_loss_parallel=True
with torch.device("meta"):
    model2 = spec.model.build()

parallelism2 = ParallelismConfig(
    tensor_parallel_degree=4, enable_sequence_parallel=True,
    disable_loss_parallel=True,
)

parallelize_qwen3(
    model2, parallel_dims=parallel_dims, training=TrainingConfig(),
    model_converters=ModelConvertersContainer.Config(),
    parallelism=parallelism2, compile_config=CompileConfig(),
    ac_config=ActivationCheckpointConfig(), dump_folder="",
    inference=True,
)
model2.to_empty(device="cuda")
model2 = model2.to(dtype=torch.bfloat16)

# Load same weights
adapter2 = spec.state_dict_adapter(spec.model, None)
storage_reader2 = adapter2.get_hf_storage_reader(model_path)
hf_sd2 = adapter2.to_hf(model2.state_dict())
hf_sd2 = {k: v for k, v in hf_sd2.items() if k in hf_keys}
dcp.load(hf_sd2, storage_reader=storage_reader2)
tt_sd2 = adapter2.from_hf(hf_sd2)

model_sd2 = dict(model2.state_dict())
for name, tensor in tt_sd2.items():
    if name in model_sd2 and isinstance(model_sd2[name], DTensor):
        if not isinstance(tensor, DTensor):
            target = model_sd2[name]
            tt_sd2[name] = DTensor.from_local(
                tensor.to(target.device_mesh.device_type),
                device_mesh=target.device_mesh,
                placements=[torch.distributed.tensor.Replicate()],
            )
set_model_state_dict(model2, tt_sd2, options=StateDictOptions(strict=False))
model2.eval()

with torch.inference_mode():
    h2 = model2.tok_embeddings(input_ids)
    for layer in model2.layers.values():
        h2 = layer(h2, freqs_cis, attention_masks=None)
    h2 = model2.norm(h2)
    logits2 = model2.output(h2)
    if isinstance(logits2, DTensor):
        logits2_full = logits2.full_tensor()
    else:
        logits2_full = logits2

if rank == 0:
    logits2_local = logits2._local_tensor if isinstance(logits2, DTensor) else logits2
    print(
        f"[PT2] logits type={'DTensor' if isinstance(logits2, DTensor) else 'plain'} "
        f"local_shape={logits2_local.shape} full_shape={logits2_full.shape}",
        flush=True,
    )
    global_argmax2 = logits2_full[0, -1].argmax().item()
    top5_2 = logits2_full[0, -1].topk(5)
    print(f"[PT2] Global argmax={global_argmax2} ({tokenizer.decode([global_argmax2])})", flush=True)
    print(f"[PT2] Top5: {[(t.item(), tokenizer.decode([t.item()])) for t in top5_2.indices]}", flush=True)
    print(f"[PT2] Top5 logits: {top5_2.values.tolist()}", flush=True)
    # Check if Paris (12095) is in top-20
    top20_2 = logits2_full[0, -1].topk(20)
    paris_logit = logits2_full[0, -1, 12095].item()
    print(f"[PT2] Token 12095 (' Paris') logit={paris_logit:.4f}", flush=True)
    print(f"[PT2] Top20 tokens: {top20_2.indices.tolist()}", flush=True)

dist.barrier()
