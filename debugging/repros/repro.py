import logging
import os
import socket
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from datetime import datetime, timedelta
from functools import partial

import torch
import torch.distributed as dist
import torch.nn.functional as F

from torch import nn
from torch.autograd.profiler import record_function
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


# logging
logging.basicConfig(
    format="%(levelname)s:%(asctime)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def main(args: Namespace):
    assert torch.cuda.is_available()

    fsdp_enabled = args.fsdp
    memory_snapshotting_enabled = args.snapshot_file is not None
    use_float8 = args.float8
    use_compile = args.compile
    model_type = args.model_type
    use_per_op_ac = args.per_op_ac
    run_training_loop = args.train

    try:
        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats(device)

        # start memory profile
        if memory_snapshotting_enabled:
            start_record_memory_history()

        # allocate model and inputs
        if model_type == "linear":
            model = LinearModel(args.num_layers).to(torch.bfloat16).to(device)
        elif model_type == "ffn":
            dim = 4096
            hidden_dim = 4 * dim
            model = FFN(dim, hidden_dim).to(torch.bfloat16).to(device)
        elif model_type == "attn":
            head_dim = 4096
            heads = 4
            kv_heads = 4
            model = Attention(head_dim, heads, kv_heads).to(torch.bfloat16).to(device)
        else:
            raise ValueError(
                f"invalid model type: {model_type} (must be one of: linear,ffn,attn)"
            )

        # fp8 rowwise quant
        if use_float8:
            apply_fp8_rowwise_quant(model)

        # selective per op AC
        if use_per_op_ac:
            model = apply_ac(model)

        # compile
        if use_compile:
            model = apply_compile(model)

        # FSDP2 (2 GPUs or more required to avoid _scaled_mm error:
        # "RuntimeError: Only bf16 high precsion output types are supported for row-wise scaling."
        if fsdp_enabled:
            setup_distributed()
            apply_fsdp(model)

        x = torch.randn(1, 16, 4096, dtype=torch.bfloat16).to(device)

        # if training is enabled, perform 5 training iterations with optimizer steps.
        if run_training_loop:
            logger.info("Training for 5 steps")
            optimizer = torch.optim.AdamW(model.parameters())
            label = torch.ones((1,), device=device, dtype=torch.bfloat16)
            for _ in range(5):
                out = model(x)
                F.mse_loss(out.sum().unsqueeze(-1), label).backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        else:
            logger.info(
                "Performing one forward+backward iteration with no optimizer step"
            )
            # if training is not enabled, do one fwd+bwd pass without any optimizer steps.
            out = model(x)
            out.sum().backward()

        torch.cuda.synchronize()

        # snapshot memory. only 1 process should snapshot memory
        if memory_snapshotting_enabled:
            is_rank_0 = fsdp_enabled and dist.get_rank() == 0
            if not fsdp_enabled or (fsdp_enabled and is_rank_0):
                export_memory_snapshot(args.snapshot_file)

            stop_record_memory_history()

        peak_memory = torch.cuda.max_memory_allocated(device)
        print(f"Peak GPU memory usage: {peak_memory / (1024 ** 2):.2f} MB")
    finally:
        if args.fsdp:
            clean_up_distributed()


################################
# Compile/FSDP2/SAC/Float8 utils
################################


def apply_compile(model: nn.Module):
    model = torch.compile(model, fullgraph=True)
    logger.info("Compiled model")
    return model


# modified version of per op AC implementation from torchtitan.
# this applies per op selective AC to a model, without assuming it is a transformer model,
# and supports no other AC settings.
# source: https://github.com/pytorch/torchtitan/blob/cca07028e440de6a13189d251c28337bd34256ef/torchtitan/parallelisms/parallelize_llama.py#L288
def apply_ac(model: nn.Module):
    if hasattr(model, "layers"):
        for layer_id, layer in model.layers.named_children():
            layer = _apply_per_op_ac_to_model(layer)
            model.layers.register_module(layer_id, layer)
        logger.info(
            f"Applied selective per op activation checkpoitning to multi-layer model"
        )
    else:
        model = _apply_per_op_ac_to_model(model)
        logger.info(
            f"Applied selective per op activation checkpointing to single layer model"
        )
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


##################
# Memory profiling
##################

# Keep a max of 100,000 alloc/free events in the recorded history
# leading up to the snapshot.
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000


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


###################
# Distributed utils
###################


def setup_distributed():
    assert "RANK" in os.environ, "env var RANK must be set for FSDP"
    assert "WORLD_SIZE" in os.environ, "env var WORLD_SIZE must be set for FSDP"
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    logger.info("Set up process group")


def clean_up_distributed():
    dist.destroy_process_group()
    logger.info("Destroyed process group")


###################
# Layer definitions
###################


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


# MHA layer from Llama3 https://github.com/pytorch/torchtitan/blob/cca07028e440de6a13189d251c28337bd34256ef/torchtitan/models/llama/model.py#L128
class Attention(nn.Module):
    """
    Multi-head attention module.

    Attributes:
        n_kv_heads (int): Number of key and value heads.
        n_heads (int): Number of query heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        wq (Linear): Linear transformation for queries.
        wk (Linear): Linear transformation for keys.
        wv (Linear): Linear transformation for values.
        wo (Linear): Linear transformation for output.

    """

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int = 1024,
        rope_theta: int = 10000,
    ):
        super().__init__()
        self.n_heads = num_heads
        self.n_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = head_dim // num_heads

        self.wq = nn.Linear(head_dim, num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(head_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(head_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(num_heads * self.head_dim, head_dim, bias=False)
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        # TODO persistent should be set to false, since this buffer can be recomputed.
        # however, we set it to true for 2 reasons.  (1) due to pytorch/pytorch#123411,
        # compile or pipeline-tracer will not correctly handle non-persistent buffers,
        # so we need to fix that.  (2) if we initialize pipeline-parallel models from
        # a seed checkpoint rather than calling init_weights, we need freqs_cis to be
        # initialized by the checkpoint, or we need to add a separate initializer for
        # just the non-persistent buffers that is called after loading checkpoints.
        self.register_buffer(
            "freqs_cis",
            self._precompute_freqs_cis(head_dim, num_heads, max_seq_len, rope_theta),
            persistent=True,
        )

    def _precompute_freqs_cis(
        self,
        head_dim: int,
        num_heads: int,
        max_seq_len: int,
        rope_theta: int,
    ):
        return precompute_freqs_cis(
            head_dim // num_heads,
            # Need to compute until at least the max token limit for generation
            # TODO: explain in docs/composability.md why we removed the 2x
            # relaxing in our CP enablement PR
            max_seq_len,
            rope_theta,
        )

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=self.freqs_cis)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        # we use casual mask for training
        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        output = output.transpose(
            1, 2
        ).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
        output = output.view(bs, seqlen, -1)
        return self.wo(output)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    The input freqs_cis tensor is assumed to be of shape (max_seqlen, dim),
    and the first seqlen elements will be sliced, but dim must match x.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    assert freqs_cis.shape == (seqlen, x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--float8", action="store_true")
    argparser.add_argument("--fsdp", action="store_true")
    argparser.add_argument("--compile", action="store_true")
    argparser.add_argument("--per-op-ac", action="store_true")
    argparser.add_argument("--num-layers", type=int, default=1)
    argparser.add_argument(
        "--model-type", type=str, required=True, help="[linear,ffn,attn]"
    )
    argparser.add_argument(
        "--snapshot-file",
        type=str,
        help="where to write the memory snapshot pickle file",
    )
    argparser.add_argument(
        "--train",
        action="store_true",
        help="If set, train for 5 steps w/ AdamW optimizer and MSE loss. Otherwise, only do one fwd+bwd with no optimizer step.",
    )
    args = argparser.parse_args()
    main(args)
