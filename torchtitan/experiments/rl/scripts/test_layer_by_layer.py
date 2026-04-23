"""
Print hidden state norm/sum/first-5-values after EVERY layer.
Identifies which layer first produces divergent output.

Also prints separate norms for attention output and MoE output
within each layer to isolate the component.

torchrun --nproc_per_node=4 test_layer_by_layer.py
"""
import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor

dist.init_process_group()
rank = dist.get_rank()
torch.cuda.set_device(rank)

from torchtitan.models.qwen3 import model_registry
from torchtitan.models.qwen3.parallelize import parallelize_qwen3
from torchtitan.models.qwen3.model import Qwen3TransformerBlock
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
    print("[LOAD] Loading weights...", flush=True)

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
    print("[LOAD] Done.", flush=True)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokens = tokenizer.encode("The capital of France is")
input_ids = torch.tensor([tokens], device="cuda")

if rank == 0:
    print(f"[INPUT] {tokens} = '{tokenizer.decode(tokens)}'", flush=True)


def get_full(t):
    """Get full tensor from DTensor or plain tensor."""
    if isinstance(t, DTensor):
        return t.full_tensor()
    return t


with torch.no_grad():
    h = model.tok_embeddings(input_ids)
    h_full = get_full(h)
    if rank == 0:
        print(f"[EMB] norm={h_full.float().norm():.4f} sum={h_full.float().sum():.4f}", flush=True)

    freqs_cis = model.freqs_cis[:len(tokens)].to("cuda")

    for i, layer in enumerate(model.layers.values()):
        # Only trace first 5 layers in detail, then every 10th
        trace = (i < 5) or (i % 10 == 0) or (i >= 45)

        if i == 1 and layer.moe_enabled:
            # Deep trace layer 1 MoE
            # Capture: ffn_norm output, MoE input (after PrepareModuleInputOutput),
            # router output, expert output, MoE raw output, post-hook output
            _l1 = {}

            def ffn_norm_hook(mod, inp, out):
                _l1["ffn_norm_out"] = out
            def moe_pre_hook(mod, inp):
                _l1["moe_input"] = inp[0]
            def moe_post_hook(mod, inp, out):
                _l1["moe_output"] = out
            def router_hook(mod, inp, out):
                _l1["router_out"] = out
            def experts_hook(mod, inp, out):
                _l1["experts_out"] = out

            hh = []
            hh.append(layer.ffn_norm.register_forward_hook(ffn_norm_hook))
            hh.append(layer.moe.register_forward_pre_hook(moe_pre_hook))
            hh.append(layer.moe.register_forward_hook(moe_post_hook))
            hh.append(layer.moe.router.register_forward_hook(router_hook))
            hh.append(layer.moe.experts.register_forward_hook(experts_hook))

            h = layer(h, freqs_cis, attention_masks=None)

            for handle in hh:
                handle.remove()

            # Print all captured values
            for key in ["ffn_norm_out", "moe_input", "router_out", "experts_out", "moe_output"]:
                val = _l1.get(key)
                if val is None:
                    if rank == 0:
                        print(f"[L01 DEEP] {key}: NOT CAPTURED", flush=True)
                    continue
                if isinstance(val, tuple):
                    val = val[0]
                val_full = get_full(val)
                val_local = val._local_tensor if isinstance(val, DTensor) else val
                placement = val.placements if isinstance(val, DTensor) else "plain"
                if rank == 0:
                    print(
                        f"[L01 DEEP] {key}: norm={val_full.float().norm():10.4f} "
                        f"local_shape={val_local.shape} place={placement} "
                        f"full_shape={val_full.shape}",
                        flush=True,
                    )

            # Check experts_out per rank
            exp_out = _l1.get("experts_out")
            if exp_out is not None:
                exp_local = exp_out._local_tensor if isinstance(exp_out, DTensor) else exp_out
                exp_local_norm = exp_local.float().norm().item()
                exp_norms = torch.tensor([exp_local_norm], device=f"cuda:{rank}")
                all_exp_norms = [torch.zeros_like(exp_norms) for _ in range(4)]
                dist.all_gather(all_exp_norms, exp_norms)
                if rank == 0:
                    print(
                        f"[L01 DEEP] experts_out local norms per rank: "
                        f"{[f'{n.item():.4f}' for n in all_exp_norms]}",
                        flush=True,
                    )

            # Also check: what is the MoE output on each rank BEFORE full_tensor?
            moe_out = _l1.get("moe_output")
            if moe_out is not None:
                moe_local = moe_out._local_tensor if isinstance(moe_out, DTensor) else moe_out
                local_norm = moe_local.float().norm().item()
                # Gather norms across ranks
                norms = torch.tensor([local_norm], device=f"cuda:{rank}")
                all_norms = [torch.zeros_like(norms) for _ in range(4)]
                dist.all_gather(all_norms, norms)
                if rank == 0:
                    print(
                        f"[L01 DEEP] moe_output local norms per rank: "
                        f"{[f'{n.item():.4f}' for n in all_norms]}",
                        flush=True,
                    )

            h_full = get_full(h)
            if rank == 0:
                print(f"[L01] out={h_full.float().norm():10.4f}", flush=True)
        else:
            # Normal forward for other layers
            _captured = {}

            def make_moe_hook(layer_id):
                def hook(module, input, output):
                    _captured[f"moe_{layer_id}"] = output
                return hook

            handles = []
            if trace:
                if layer.moe_enabled:
                    handles.append(layer.moe.register_forward_hook(make_moe_hook(i)))
                else:
                    handles.append(layer.feed_forward.register_forward_hook(make_moe_hook(i)))

            h = layer(h, freqs_cis, attention_masks=None)

            if trace:
                for handle in handles:
                    handle.remove()
                h_full = get_full(h)
                moe_val = _captured.get(f"moe_{i}")
                moe_full = get_full(moe_val) if moe_val is not None else None
                if rank == 0:
                    moe_norm = moe_full.float().norm().item() if moe_full is not None else -1
                    print(
                        f"[L{i:02d}] moe={moe_norm:10.4f} out={h_full.float().norm():10.4f}",
                        flush=True,
                    )

    h = model.norm(h)
    logits = model.output(h)
    if isinstance(logits, DTensor):
        logits = logits.full_tensor()

if rank == 0:
    argmax = logits[0, -1].argmax().item()
    top5 = logits[0, -1].topk(5)
    print(f"\n[OUT] argmax={argmax} ({tokenizer.decode([argmax])})", flush=True)
    print(f"[OUT] top5={[(t.item(), tokenizer.decode([t.item()])) for t in top5.indices]}", flush=True)
    print(f"[OUT] top5_logits={top5.values.tolist()}", flush=True)

dist.barrier()
