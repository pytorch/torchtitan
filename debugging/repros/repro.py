import logging
import os
import socket
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from datetime import datetime, timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)
from torchao.float8.config import Float8LinearRecipeName, recipe_name_to_linear_config
from torchao.float8.float8_linear_utils import convert_to_float8_training

# Keep a max of 100,000 alloc/free events in the recorded history
# leading up to the snapshot.
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000

# logging
logging.basicConfig(
    format="%(levelname)s:%(asctime)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class LinearModel(nn.Module):
    def __init__(self, num_layers: int = 1):
        super(LinearModel, self).__init__()
        self.layers = nn.Sequential(*[nn.Linear(4096, 4096) for _ in range(num_layers)])

    def forward(self, x):
        return self.layers(x)


# Simplified FFN from Llama3 https://github.com/pytorch/torchtitan/blob/cca07028e440de6a13189d251c28337bd34256ef/torchtitan/models/llama/model.py#L217
class FFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super(FFN, self).__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


def main(args: Namespace):
    assert torch.cuda.is_available()
    try:
        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats(device)

        # start memory profile
        start_record_memory_history()

        # allocate model and inputs
        model = FFN(4096, 4 * 4096).to(torch.bfloat16).to(device)
        x = torch.randn(16, 4096, dtype=torch.bfloat16).to(device)

        # fp8 rowwise quant
        if args.float8:
            apply_fp8_rowwise_quant(model)

        # selective per op AC
        if args.per_op_ac:
            model = apply_ac(model)

        # compile
        if args.compile:
            model = apply_compile(model)

        # FSDP2 (2 GPUs or more required to avoid _scaled_mm error:
        # "RuntimeError: Only bf16 high precsion output types are supported for row-wise scaling."
        if args.fsdp:
            setup_distributed()
            apply_fsdp(model)

        # one fwd + backward
        out = model(x)
        out.sum().backward()

        # only 1 process should snapshot memory
        if not (args.fsdp and dist.get_rank() != 0):
            export_memory_snapshot(args.snapshot_file)

        stop_record_memory_history()

        peak_memory = torch.cuda.max_memory_allocated(device)
        print(f"Peak GPU memory usage: {peak_memory / (1024 ** 2):.2f} MB")
    finally:
        if args.fsdp:
            clean_up_distributed()


def apply_compile(model: nn.Module):
    model = torch.compile(model, fullgraph=True)
    logger.info("Compiled model")
    return model


# modified version of per op AC implementation from torchtitan.
# this applies per op selective AC to a model, without assuming it is a transformer model,
# and supports no other AC settings.
# source: https://github.com/pytorch/torchtitan/blob/cca07028e440de6a13189d251c28337bd34256ef/torchtitan/parallelisms/parallelize_llama.py#L288
def apply_ac(model: nn.Module):
    """Apply activation checkpointing to the model."""
    model = _apply_per_op_ac_to_model(model)
    logger.info(f"Applied selective per op activation checkpointing to the model")
    return model


def apply_fp8_rowwise_quant(model: nn.Module):
    recipe = Float8LinearRecipeName("all_axiswise")
    config = recipe_name_to_linear_config(recipe)
    convert_to_float8_training(model, config=config)
    logger.info("Applied fp8 rowwise quantization to model")


def apply_fsdp(model: nn.Module):
    fully_shard(model)
    logger.info("Applied FSDP2 to model")


def _apply_per_op_ac_to_model(module: nn.Module):
    _save_list = {
        torch.ops.aten.mm.default,
        torch.ops.aten._scaled_dot_product_efficient_attention.default,
        torch.ops.aten._scaled_dot_product_flash_attention.default,
        torch.ops._c10d_functional.reduce_scatter_tensor.default,
        # for low precision training, it's useful to always save
        # the result of max(abs(tensor))
        torch.ops.aten.abs.default,
        torch.ops.aten.max.default,
    }

    def _get_custom_policy(meta):
        def _custom_policy(ctx, func, *args, **kwargs):
            mode = "recompute" if ctx.is_recompute else "forward"
            mm_count_key = f"{mode}_mm_count"
            if func == torch.ops.aten.mm.default:
                meta[mm_count_key] += 1
            # Saves output of all compute ops, except every second mm
            to_save = func in _save_list and not (
                func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0
            )
            return (
                CheckpointPolicy.MUST_SAVE
                if to_save
                else CheckpointPolicy.PREFER_RECOMPUTE
            )

        return _custom_policy

    def selective_checkpointing_context_fn():
        meta = defaultdict(int)
        return create_selective_checkpoint_contexts(_get_custom_policy(meta))

    return ptd_checkpoint_wrapper(
        module,
        context_fn=selective_checkpointing_context_fn,
        preserve_rng_state=False,
    )


# memory snapshotting functions from:
# https://pytorch.org/blog/understanding-gpu-memory-1/#appendix-a---resnet50-memory-snapshot-code-example
def start_record_memory_history() -> None:
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not recording memory history")
        return

    logger.info("Starting snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(
        max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
    )


def stop_record_memory_history() -> None:
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not recording memory history")
        return

    logger.info("Stopping snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(enabled=None)


def export_memory_snapshot(filepath: str) -> None:
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not exporting memory snapshot")
        return

    try:
        logger.info(f"Saving snapshot to local file: {filepath}")
        torch.cuda.memory._dump_snapshot(f"{filepath}")
    except Exception as e:
        logger.error(f"Failed to capture memory snapshot {e}")
        return


def setup_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    logger.info("Set up process group")


def clean_up_distributed():
    dist.destroy_process_group()
    logger.info("Destroyed process group")


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--float8", action="store_true")
    argparser.add_argument("--fsdp", action="store_true")
    argparser.add_argument("--compile", action="store_true")
    argparser.add_argument("--per-op-ac", action="store_true")
    argparser.add_argument("--num-layers", type=int, default=1)
    argparser.add_argument("--snapshot-file", type=str, required=True)
    args = argparser.parse_args()
    main(args)
