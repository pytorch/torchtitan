# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Standalone benchmark for torch.compile on compute_policy_gradient_loss.

Runs the loss function under several compilation modes and reports per-step
timing so you can iterate on the function without needing a full distributed
RL job.

Three model modes:
  --model mock        Tiny Embedding+Linear. Fast to compile, isolates the
                      loss-specific ops (KL, PPO clip, entropy, log-ratio).
  --model debugmodel  Qwen3 debugmodel (256-dim, 8 layers, random weights).
                      Adds transformer forward overhead without needing a checkpoint.
  --model checkpoint  Actual Qwen3-0.6B loaded from --hf-assets-path.
                      Most representative of what the real RL job compiles.

Usage:
    # Quick iteration, loss ops only
    python torchtitan/experiments/rl/unified/bench_compile_loss.py --backend aot_eager

    # Real model architecture, no checkpoint needed
    python torchtitan/experiments/rl/unified/bench_compile_loss.py --model debugmodel --backend aot_eager

    # Fully representative (actual 0.6B weights)
    python torchtitan/experiments/rl/unified/bench_compile_loss.py --model checkpoint --backend aot_eager

    # Full inductor benchmark
    python torchtitan/experiments/rl/unified/bench_compile_loss.py --model checkpoint --backend inductor
"""

import argparse
import time

import torch
import torch.nn as nn

from torchtitan.experiments.rl.unified.actors.utils import compute_policy_gradient_loss


# ---------------------------------------------------------------------------
# Mock model: Embedding + Linear, matches compute_token_log_probs interface:
#   model(token_ids_2d, attention_masks=None) -> (batch, seq, vocab) logits
# ---------------------------------------------------------------------------

class _MockModel(nn.Module):
    def __init__(self, vocab_size: int, hidden: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.proj = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, x: torch.Tensor, attention_masks=None) -> torch.Tensor:
        return self.proj(self.embed(x))


def _build_mock_model(args, device: torch.device) -> nn.Module:
    return _MockModel(args.vocab_size, args.hidden).to(device)


# ---------------------------------------------------------------------------
# Real model: Qwen3 debugmodel (no checkpoint, no TP, single GPU)
# ---------------------------------------------------------------------------

def _build_qwen3_model(flavor: str, device: torch.device, hf_assets_path: str | None = None) -> nn.Module:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
    from torchtitan.models.qwen3 import model_registry, qwen3_configs
    from torchtitan.tools import utils as titan_utils

    cfg = qwen3_configs[flavor]
    with torch.device("meta"):
        with titan_utils.set_default_dtype(torch.bfloat16):
            model = cfg.build()

    model.to_empty(device=device)
    with torch.no_grad():
        model.init_weights(buffer_device=None)

    if hf_assets_path is not None:
        print(f"  Loading weights from {hf_assets_path} ...")
        spec = model_registry(flavor)
        adapter = spec.state_dict_adapter(cfg, hf_assets_path)
        storage_reader = adapter.get_hf_storage_reader(hf_assets_path)
        hf_state_dict = adapter.to_hf(model.state_dict())
        dcp.load(hf_state_dict, storage_reader=storage_reader)
        torchtitan_state_dict = adapter.from_hf(hf_state_dict)
        set_model_state_dict(
            model=model,
            model_state_dict=torchtitan_state_dict,
            options=StateDictOptions(strict=True),
        )
        print("  Weights loaded.")

    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_batch(
    batch_size: int,
    prompt_len: int,
    gen_len: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[list[list[int]], list[list[int]], torch.Tensor, list[torch.Tensor]]:
    """Return (vllm_token_ids, prompt_token_ids, advantages, ref_log_probs)."""
    vllm_token_ids = [
        torch.randint(0, vocab_size, (gen_len,)).tolist() for _ in range(batch_size)
    ]
    prompt_token_ids = [
        torch.randint(0, vocab_size, (prompt_len,)).tolist() for _ in range(batch_size)
    ]
    advantages = torch.randn(batch_size, device=device)
    ref_log_probs = [torch.randn(gen_len, device=device) for _ in range(batch_size)]
    return vllm_token_ids, prompt_token_ids, advantages, ref_log_probs


def _run_step(fn, model, vllm_token_ids, prompt_token_ids, advantages, ref_log_probs) -> float:
    t0 = time.perf_counter()
    loss, _metrics, _lps = fn(
        model, vllm_token_ids, prompt_token_ids, advantages, ref_log_probs
    )
    loss.backward()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def _bench(label, fn, model, vllm_token_ids, prompt_token_ids, advantages, ref_log_probs, steps):
    model.zero_grad(set_to_none=True)

    print(f"\n[{label}] compiling / warming up ...")
    t_warmup = _run_step(fn, model, vllm_token_ids, prompt_token_ids, advantages, ref_log_probs)
    print(f"[{label}] warmup step : {t_warmup:.3f}s")

    times = []
    for i in range(steps):
        model.zero_grad(set_to_none=True)
        t = _run_step(fn, model, vllm_token_ids, prompt_token_ids, advantages, ref_log_probs)
        times.append(t)
        print(f"[{label}] step {i + 1:2d}    : {t:.4f}s")

    avg = sum(times) / len(times)
    print(
        f"[{label}] avg {steps} steps : {avg:.4f}s  "
        f"(min={min(times):.4f}s  max={max(times):.4f}s)"
    )
    return avg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark compile_loss")
    parser.add_argument(
        "--model",
        choices=["mock", "debugmodel", "checkpoint"],
        default="mock",
        help=(
            "mock: tiny Embedding+Linear, isolates loss ops. "
            "debugmodel: Qwen3 256-dim/8-layer, no weights loaded. "
            "checkpoint: actual Qwen3-0.6B from --hf-assets-path (most representative)."
        ),
    )
    parser.add_argument(
        "--hf-assets-path",
        default="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        help="Path to HF checkpoint for --model checkpoint.",
    )
    parser.add_argument("--backend", default="inductor", help="torch.compile backend")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--gen-len", type=int, default=24)
    parser.add_argument("--steps", type=int, default=5, help="Timed steps after warmup")
    parser.add_argument("--no-eager", action="store_true", help="Skip eager baseline")
    # Mock-only knobs
    parser.add_argument("--vocab-size", type=int, default=256, help="Mock model vocab size")
    parser.add_argument("--hidden", type=int, default=128, help="Mock model hidden dim")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "debugmodel":
        print("Building Qwen3 debugmodel (256-dim, 8 layers, random weights) ...")
        model = _build_qwen3_model("debugmodel", device)
        vocab_size = 2048
        print("Qwen3 debugmodel ready.")
    elif args.model == "checkpoint":
        print(f"Building Qwen3-0.6B from {args.hf_assets_path} ...")
        model = _build_qwen3_model("0.6B", device, hf_assets_path=args.hf_assets_path)
        vocab_size = 151936
        print("Qwen3-0.6B ready.")
    else:
        model = _build_mock_model(args, device)
        vocab_size = args.vocab_size

    print(f"\nDevice      : {device}")
    print(f"Model       : {args.model}")
    print(f"Backend     : {args.backend}")
    print(
        f"Batch       : size={args.batch_size}  "
        f"prompt_len={args.prompt_len}  gen_len={args.gen_len}"
    )

    vllm_token_ids, prompt_token_ids, advantages, ref_log_probs = _make_batch(
        args.batch_size, args.prompt_len, args.gen_len, vocab_size, device
    )

    results = {}

    if not args.no_eager:
        results["eager"] = _bench(
            "eager",
            compute_policy_gradient_loss,
            model,
            vllm_token_ids,
            prompt_token_ids,
            advantages,
            ref_log_probs,
            args.steps,
        )

    compiled_fn = torch.compile(compute_policy_gradient_loss, backend=args.backend)
    results[args.backend] = _bench(
        args.backend,
        compiled_fn,
        model,
        vllm_token_ids,
        prompt_token_ids,
        advantages,
        ref_log_probs,
        args.steps,
    )

    if "eager" in results and args.backend in results:
        speedup = results["eager"] / results[args.backend]
        print(f"\nSpeedup ({args.backend} vs eager): {speedup:.2f}x")


if __name__ == "__main__":
    main()
