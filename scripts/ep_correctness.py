"""
EP correctness test.

Verifies that Expert Parallelism produces numerically identical results
to the non-parallelized baseline by comparing a single forward pass.

Both runs use 4 GPUs with FSDP — the only difference is ep=1 vs ep=4.

Run with:
    torchrun --nproc_per_node=4 scripts/ep_correctness.py --ep 1
    torchrun --nproc_per_node=4 scripts/ep_correctness.py --ep 4

Both should print the same loss value (within fp precision).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.config import TORCH_DTYPE_MAP
from src.config.config import ModelConfig
from src.distributed import ParallelDims
from src.logging import init_logger, logger
from src.models.moe.model import MoETransformer
from src.models.parallelize import apply_fsdp, apply_moe_ep


def main():
    init_logger()

    ep = int(sys.argv[sys.argv.index("--ep") + 1]) if "--ep" in sys.argv else 1

    # Init distributed
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    assert world_size == 4, f"This test requires exactly 4 GPUs, got {world_size}"

    # Build mesh: always dp_shard=4, only EP varies
    parallel_dims = ParallelDims(
        dp_replicate=1,
        dp_shard=4,
        cp=1,
        tp=1,
        pp=1,
        ep=ep,
        etp=1,
        world_size=4,
    )
    parallel_dims.build_mesh()

    # Fixed tiny model config with force_load_balance for deterministic routing
    cfg = ModelConfig(
        n_layers=2,
        dim=256,
        n_heads=4,
        n_kv_heads=2,
        head_dim=64,
        num_experts=8,
        top_k=2,
        num_shared_experts=1,
        ffn_dim=128,
        vocab_size=1024,
        rope_theta=500_000.0,
        n_dense_layers=0,
        force_load_balance=True,
        max_seq_len=128,  # type: ignore[call-arg]
    )

    # Build model from same seed on all ranks
    torch.manual_seed(42)
    with torch.device("meta"):
        model = MoETransformer(cfg)

    # Apply parallelism: EP (if enabled) then FSDP (always, since dp_shard=4)
    if parallel_dims.ep_enabled:
        apply_moe_ep(model, parallel_dims.get_mesh("ep"))

    apply_fsdp(
        model,
        parallel_dims.get_mesh("fsdp"),
        param_dtype=TORCH_DTYPE_MAP["bfloat16"],
        reduce_dtype=TORCH_DTYPE_MAP["float32"],
        ep_degree=ep,
        edp_mesh=parallel_dims.get_optional_mesh("efsdp"),
        gradient_divide_factor=parallel_dims.fsdp_gradient_divide_factor,
    )

    # Materialize and init with same seed
    model.to_empty(device=device)
    torch.manual_seed(42)
    model.init_weights(buffer_device=device)
    model.eval()

    # Same input on all ranks
    torch.manual_seed(123)
    tokens = torch.randint(0, cfg.vocab_size, (2, 128), device=device)
    labels = torch.randint(0, cfg.vocab_size, (2, 128), device=device)

    # Forward
    with torch.no_grad():
        pred = model(tokens)
        loss = F.cross_entropy(pred.flatten(0, 1).float(), labels.flatten(0, 1))

    # All-reduce loss across ranks to get a comparable number
    dist.all_reduce(loss, op=dist.ReduceOp.AVG)

    if rank == 0:
        logger.info(f"EP={ep}, world_size={world_size}, loss={loss.item():.6f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
