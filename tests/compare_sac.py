"""
Compare old (counter-based) vs new (FQN-based) SAC by running forward+backward
with a fixed seed and comparing loss + memory.

Usage:
    python tests/compare_sac.py
"""

import sys
from collections import defaultdict
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

from torchtitan.config import ActivationCheckpointConfig as ACConfig
from torchtitan.distributed.activation_checkpoint import apply_ac


# A Llama-style transformer block for testing.
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

    def forward(self, x):
        bs, seq_len, dim = x.shape
        q = self.wq(x).view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(bs, seq_len, dim)
        return self.wo(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, hidden_dim):
        super().__init__()
        self.attention = Attention(dim, n_heads)
        self.feed_forward = FeedForward(dim, hidden_dim)
        self.attention_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class ToyTransformer(nn.Module):
    def __init__(self, dim=128, n_heads=4, hidden_dim=256, n_layers=4, vocab_size=256):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleDict(
            {str(i): TransformerBlock(dim, n_heads, hidden_dim) for i in range(n_layers)}
        )
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens):
        x = self.tok_embeddings(tokens)
        for layer in self.layers.values():
            x = layer(x)
        x = self.norm(x)
        return self.output(x)


_ALWAYS_SAVE_OPS = {
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    torch.ops.aten._scaled_dot_product_attention_math.default,
    torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
    torch.ops.aten.max.default,
    torch._higher_order_ops.flex_attention,
    torch._higher_order_ops.inductor_compiled_code,
}

_SAC_SAVE_LIST = [
    "layers.*.attention.mm_0_0",       # wq
    "layers.*.attention.mm_2_0",       # wv
    "layers.*.feed_forward.mm_0_0",    # w1
    "layers.*.feed_forward.mm_2_0",    # w2
]


def apply_old_sac(model):
    """Old counter-based SAC."""
    op_sac_save_list = _ALWAYS_SAVE_OPS | {
        torch.ops.aten.mm.default,
        torch.ops.aten.linear.default,
    }
    mm_ops = (torch.ops.aten.mm.default, torch.ops.aten.linear.default)

    def context_fn():
        meta = defaultdict(int)

        def policy(ctx, func, *args, **kwargs):
            mode = "recompute" if ctx.is_recompute else "forward"
            mm_count_key = f"{mode}_mm_count"
            if func in mm_ops:
                meta[mm_count_key] += 1
            to_save = func in op_sac_save_list and not (
                func in mm_ops and meta[mm_count_key] % 2 == 0
            )
            return (
                CheckpointPolicy.MUST_SAVE
                if to_save
                else CheckpointPolicy.PREFER_RECOMPUTE
            )

        return create_selective_checkpoint_contexts(policy)

    layers = model.get_submodule("layers")
    for layer_id, block in layers.named_children():
        wrapped = checkpoint_wrapper(block, context_fn=context_fn)
        layers.register_module(layer_id, wrapped)


def apply_new_sac(model):
    """New FQN-based SAC."""
    ac_config = ACConfig(
        mode="selective",
        selective_ac_option="op",
        per_op_sac_force_recompute_mm_shapes_by_fqns=[],
    )
    apply_ac(
        model,
        ac_config,
        always_save_ops=_ALWAYS_SAVE_OPS,
        sac_save_list=_SAC_SAVE_LIST,
    )


def run_steps(model, seed, steps=5, device="cpu"):
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.reset_peak_memory_stats()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    vocab_size = model.output.out_features
    losses = []

    for step in range(steps):
        torch.manual_seed(seed + step + 1000)
        input_ids = torch.randint(0, vocab_size, (2, 64), device=device)

        optimizer.zero_grad()
        logits = model(input_ids)
        targets = input_ids[:, 1:]
        logits = logits[:, :-1, :].reshape(-1, vocab_size)
        loss = torch.nn.functional.cross_entropy(logits, targets.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    peak_mem = 0.0
    if device == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
    return losses, peak_mem


def main():
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"Comparing old (counter-based) vs new (FQN-based) SAC on {device}")
    print(f"{'='*60}")

    # Old SAC
    torch.manual_seed(seed)
    model_old = ToyTransformer().to(device)
    state = {k: v.clone() for k, v in model_old.state_dict().items()}
    apply_old_sac(model_old)
    old_losses, old_mem = run_steps(model_old, seed, device=device)
    del model_old

    print("\n--- Old (counter-based) SAC ---")
    for i, loss in enumerate(old_losses):
        print(f"  Step {i}: loss={loss:.6f}")
    if device == "cuda":
        print(f"  Peak memory: {old_mem:.1f} MB")

    # New SAC
    torch.manual_seed(seed)
    model_new = ToyTransformer().to(device)
    model_new.load_state_dict(state)
    apply_new_sac(model_new)
    new_losses, new_mem = run_steps(model_new, seed, device=device)
    del model_new

    print("\n--- New (FQN-based) SAC ---")
    for i, loss in enumerate(new_losses):
        print(f"  Step {i}: loss={loss:.6f}")
    if device == "cuda":
        print(f"  Peak memory: {new_mem:.1f} MB")

    # Compare
    print(f"\n{'='*60}")
    print("Comparison:")
    all_match = True
    for i, (old_l, new_l) in enumerate(zip(old_losses, new_losses)):
        match = abs(old_l - new_l) < 1e-5
        status = "MATCH" if match else f"DIFFER (delta={abs(old_l - new_l):.8f})"
        if not match:
            all_match = False
        print(f"  Step {i}: old={old_l:.6f} new={new_l:.6f} -> {status}")

    if device == "cuda":
        mem_diff = abs(old_mem - new_mem)
        mem_ok = mem_diff < 1.0
        if not mem_ok:
            all_match = False
        print(f"  Memory: old={old_mem:.1f} MB, new={new_mem:.1f} MB, diff={mem_diff:.1f} MB")

    if all_match:
        print("\nAll checks passed!")
    else:
        print("\nSome checks failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
