"""
Compare forward path: core TorchTitan (pretraining) vs vLLM wrapper.

Both use the same model definition, same TP=4 parallelize, same weights.
Runs a single prefill pass on the same input tokens and compares hidden
states after each layer.

This isolates whether the bug is in:
- The vLLM attention wrapper (PyTorchVarlenAttentionImpl)
- The vLLM forward path (unsqueeze, full_tensor, etc.)
- Or the model computation itself

torchrun --nproc_per_node=4 test_pretrain_vs_vllm.py
"""
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_MOE_BACKEND"] = "triton"

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import init_device_mesh

# --- Part 1: Build TorchTitan model with pretraining parallelize ---
dist.init_process_group()
rank = dist.get_rank()
torch.cuda.set_device(rank)

from torchtitan.models.qwen3 import model_registry
from torchtitan.models.qwen3.parallelize import parallelize_qwen3
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import ModelConvertersContainer

model_path = "/data/users/jianiw/model/Qwen3-30B-A3B"

# Build model spec
spec = model_registry("30B-A3B")

# Create ParallelDims for TP=4
parallel_dims = ParallelDims(
    dp_replicate=1, dp_shard=1, cp=1, tp=4, pp=1, ep=1, etp=1,
    world_size=4,
)
parallel_dims.build_mesh()

parallelism = ParallelismConfig(
    tensor_parallel_degree=4,
    enable_sequence_parallel=True,
)

# Build on meta device
with torch.device("meta"):
    model_pt = spec.model.build()

# Apply pretraining parallelize (inference=True to skip FSDP/AC/compile)
parallelize_qwen3(
    model_pt,
    parallel_dims=parallel_dims,
    training=TrainingConfig(),
    model_converters=ModelConvertersContainer.Config(),
    parallelism=parallelism,
    compile_config=CompileConfig(),
    ac_config=ActivationCheckpointConfig(),
    dump_folder="",
    inference=True,
)

# Materialize
model_pt.to_empty(device="cuda")
model_pt = model_pt.to(dtype=torch.bfloat16)

# Load weights using adapter
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions

adapter = spec.state_dict_adapter(spec.model, None)
storage_reader = adapter.get_hf_storage_reader(model_path)
hf_sd = adapter.to_hf(model_pt.state_dict())
hf_keys = set(storage_reader.read_metadata().state_dict_metadata.keys())
hf_sd = {k: v for k, v in hf_sd.items() if k in hf_keys}
dcp.load(hf_sd, storage_reader=storage_reader)
tt_sd = adapter.from_hf(hf_sd)

# Convert plain tensors to DTensor where needed
model_sd = dict(model_pt.state_dict())
for name, tensor in tt_sd.items():
    if name in model_sd and isinstance(model_sd[name], DTensor):
        if not isinstance(tensor, DTensor):
            target = model_sd[name]
            tt_sd[name] = DTensor.from_local(
                tensor.to(target.device_mesh.device_type),
                device_mesh=target.device_mesh,
                placements=[torch.distributed.tensor.Replicate()],
            )
set_model_state_dict(model_pt, tt_sd, options=StateDictOptions(strict=False))
model_pt.eval()

if rank == 0:
    print("[PT] Pretraining model loaded", flush=True)

# --- Create test input ---
# Tokenize "The capital of France is" using the model's tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokens = tokenizer.encode("The capital of France is")
input_ids = torch.tensor([tokens], device="cuda")  # [1, seq_len]

if rank == 0:
    print(f"[INPUT] tokens={tokens}, shape={input_ids.shape}", flush=True)

# --- Run pretraining model forward ---
with torch.inference_mode():
    h_pt = model_pt.tok_embeddings(input_ids)
    freqs_cis = model_pt.freqs_cis[:input_ids.shape[1]].to("cuda")

    pt_layer_outputs = []
    for i, layer in enumerate(model_pt.layers.values()):
        h_pt = layer(h_pt, freqs_cis, attention_masks=None)
        # Collect local tensor for comparison
        h_local = h_pt._local_tensor if isinstance(h_pt, DTensor) else h_pt
        pt_layer_outputs.append(h_local.clone())

    h_pt = model_pt.norm(h_pt)
    h_pt_final = h_pt._local_tensor if isinstance(h_pt, DTensor) else h_pt

    # Get logits through the model's output layer (handles DTensor all-gather)
    logits_pt = model_pt.output(h_pt)
    # logits_pt may be DTensor — extract
    if isinstance(logits_pt, DTensor):
        logits_pt_local = logits_pt._local_tensor
        logits_pt_placement = logits_pt.placements
    else:
        logits_pt_local = logits_pt
        logits_pt_placement = "plain"

if rank == 0:
    print(f"[PT] Final norm output local shape={h_pt_final.shape}", flush=True)
    print(f"[PT] Logits local shape={logits_pt_local.shape} placement={logits_pt_placement}", flush=True)
    # If logits are Replicate, argmax is global
    # If logits are Shard, need to account for offset
    pt_argmax = logits_pt_local[0, -1].argmax().item()
    print(f"[PT] Last token LOCAL argmax={pt_argmax}", flush=True)
    # Get full logits for correct global argmax
    if isinstance(logits_pt, DTensor):
        logits_full = logits_pt.full_tensor()
    else:
        logits_full = logits_pt
    global_argmax = logits_full[0, -1].argmax().item()
    print(f"[PT] Last token GLOBAL argmax={global_argmax} ({tokenizer.decode([global_argmax])})", flush=True)
    print(f"[PT] Top5 global: {logits_full[0, -1].topk(5).indices.tolist()}", flush=True)

# Free pretraining model memory before loading vLLM
del model_pt, pt_layer_outputs, h_pt, h_pt_final, logits_pt
if 'logits_full' in dir():
    del logits_full
torch.cuda.empty_cache()

# --- Part 2: Build vLLM wrapper model ---
from torchtitan.experiments.rl.models.vllm_registry import (
    register_model_to_vllm_model_registry, VLLM_MODEL_NAME,
)
from vllm import EngineArgs, LLMEngine, SamplingParams

model_spec_vllm = model_registry("30B-A3B", attn_backend="varlen")
register_model_to_vllm_model_registry(model_spec_vllm)

engine = LLMEngine.from_engine_args(EngineArgs(
    model=model_path, trust_remote_code=True, dtype="bfloat16",
    tensor_parallel_size=4, distributed_executor_backend="external_launcher",
    gpu_memory_utilization=0.95, max_model_len=256, enforce_eager=True,
    hf_overrides={"architectures": [VLLM_MODEL_NAME]},
    attention_backend="CUSTOM",
))

model_vllm = engine.model_executor.driver_worker.model_runner.model.model

if rank == 0:
    print(f"\n[VLLM] vLLM model loaded", flush=True)

# --- Run vLLM model forward on the same tokens ---
with torch.inference_mode():
    input_ids_1d = torch.tensor(tokens, device="cuda")
    tokens_2d = input_ids_1d.unsqueeze(0)
    positions = torch.arange(len(tokens), device="cuda").unsqueeze(0)

    h_vllm = model_vllm.tok_embeddings(tokens_2d)

    freqs_cis_vllm = model_vllm.freqs_cis
    vllm_layer_outputs = []
    for i, layer in enumerate(model_vllm.layers.values()):
        h_vllm = layer(h_vllm, freqs_cis_vllm, attention_masks=None, positions=positions)
        h_local = h_vllm._local_tensor if isinstance(h_vllm, DTensor) else h_vllm
        vllm_layer_outputs.append(h_local.clone())

    h_vllm = model_vllm.norm(h_vllm)
    h_vllm_final = h_vllm._local_tensor if isinstance(h_vllm, DTensor) else h_vllm

# --- Compare layer by layer ---
if rank == 0:
    print(f"\n[COMPARE] Layer-by-layer comparison (rank 0 local tensors):", flush=True)
    for i in [0, 1, 5, 10, 23, 47]:
        if i >= len(pt_layer_outputs):
            break
        pt_out = pt_layer_outputs[i]
        vllm_out = vllm_layer_outputs[i]
        if pt_out.shape != vllm_out.shape:
            print(f"  layer {i}: SHAPE MISMATCH pt={pt_out.shape} vllm={vllm_out.shape}", flush=True)
            continue
        diff = (pt_out.float() - vllm_out.float()).abs()
        max_diff = diff.max().item()
        avg_diff = diff.mean().item()
        pt_norm = pt_out.float().norm().item()
        vllm_norm = vllm_out.float().norm().item()
        status = "OK" if max_diff < 0.1 else "DIFF"
        print(
            f"  layer {i}: max_diff={max_diff:.6f} avg_diff={avg_diff:.8f} "
            f"pt_norm={pt_norm:.4f} vllm_norm={vllm_norm:.4f} [{status}]",
            flush=True,
        )

    # Compare final output
    diff_final = (h_pt_final.float() - h_vllm_final.float()).abs()
    print(
        f"\n  final norm: max_diff={diff_final.max().item():.6f} "
        f"pt_norm={h_pt_final.float().norm().item():.4f} "
        f"vllm_norm={h_vllm_final.float().norm().item():.4f}",
        flush=True,
    )

dist.barrier()
