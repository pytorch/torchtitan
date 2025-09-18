"""
Compare logits and generations of GPT-OSS implemented in TorchTitan and HuggingFace.
This requires at least a 2xH100.

First ensure you convert the HF model to a TorchTitan DCP checkpoint:
uv run torchtitan/experiments/gpt_oss/scripts/convert_gptoss.py hf-to-dcp --input-path  openai/gpt-oss-20b --output-path gptoss_dcp/

Then you can run a comparison like this:
uv run torchtitan/experiments/gpt_oss/scripts/compare_hf_to_tt.py \
    --tt_config  torchtitan/models/gpt_oss/train_configs/gpt_oss_20b.toml \
    --tt_checkpoint_path gptoss_dcp/ \
    --hf_model_path openai/gpt-oss-20b \
    --prompt       "Once upon a time, in a land far away," \
    --temperature  0.8 \
    --max_new_tokens 256 \
    --batch_size  1 \
    --out
"""

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, NamedTuple

import torch
import torch.nn as nn
import torch.distributed.checkpoint as dcp
import tyro
from transformers import AutoModelForCausalLM, AutoTokenizer

from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.utils import device_module, device_type
from torchtitan.components.metrics import build_device_memory_monitor
from torchtitan.config_manager import ConfigManager
from torchtitan.protocols.train_spec import get_train_spec
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torch.distributed import DeviceMesh
from torch.distributed.elastic.multiprocessing.errors import record

# -------- Torchtitan Sampling Utils --------
def multinomial_sample_one(
    probs: torch.Tensor, rng: Optional[torch.Generator] = None
) -> torch.Tensor:
    q = torch.empty_like(probs).exponential_(1, generator=rng)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.long)


def logits_to_probs(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
        pivot = v.select(dim=-1, index=-1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def generate_next_token(
    model,
    x: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    logits = model(x)  # (B, T, vocab_size)
    probs = logits_to_probs(logits[:, -1, :], temperature, top_k)
    next_token = multinomial_sample_one(probs, rng=rng)
    return next_token


@torch.no_grad()
def tt_generate_text(
    model,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    # ensure batch dimension (T,) --> (B, T)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)

    rng = None
    if seed is not None:
        rng = torch.Generator(input_ids.device).manual_seed(seed)

    generated_tokens = input_ids.clone()

    for i in range(max_new_tokens):
        next_token = generate_next_token(
            model,
            x=generated_tokens.to(input_ids.device),
            temperature=temperature,
            top_k=top_k,
            rng=rng,
        )
        print(f"generated token {i}: {next_token}")

        generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

    return generated_tokens

@dataclass
class GenerateConfig:
    """Configuration for test generation."""
    hf_model_path: Optional[str] = None
    """HuggingFace model path to load (if provided)."""
    tt_config: Optional[str] = None
    """TOML config file path for TorchTitan model."""
    tt_checkpoint_path: Optional[str] = None
    """Checkpoint path for the TorchTitan model (if provided)."""
    tt_tokenizer_path: Optional[str] = "libs/torchtitan/torchtitan/models/gpt_oss_20b/tokenizer"
    """Tokenizer path to load."""
    temperature: float = 1.0
    """Sampling temperature (0 for greedy)."""
    max_new_tokens: int = 32
    """Max number of tokens to generate."""
    batch_size: int = 1
    """Batch size for inputs."""
    top_k: Optional[int] = None
    """Top-k sampling (optional)."""
    seed: Optional[int] = None
    """Random seed for reproducibility."""
    deterministic: bool = False
    """Use deterministic algorithms."""
    prompt: str = ""
    """Input prompt string."""
    out: bool = False
    """If true, print JSON report at end."""


class LogitsComparison(NamedTuple):
    max_abs_diff: float
    mean_abs_diff: float
    max_rel_diff: float
    mean_rel_diff: float
    allclose_results: Sequence[Tuple[float, float, str, bool]]
    sample_diffs: Optional[torch.Tensor]
    systematic_offset: Optional[Tuple[float, float]]


def load_hf_model(path: str, device: torch.device) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(path).to(device)
    model.eval()
    return model

def print_param_dtypes_first_block(model):
    """
    Prints the dtype of every parameter in the given model.
    For any parameters under a 'layers' module (e.g., layers.<idx>),
    only prints those from the first block (idx == "0").
    This works for both GptOssForCausalLM (with a .model submodule)
    and GptOssModel architectures.
    """
    for name, param in model.named_parameters():
        parts = name.split('.')
        # If this parameter is under a 'layers' module, check its index
        if 'layers' in parts:
            idx = parts.index('layers') + 1
            if idx < len(parts) and parts[idx] != '0':
                continue
        print(f"{name:50s} → {param.dtype}")

def get_logits(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        out = model(input_ids)
        if hasattr(out, "logits"):
            return out.logits
        else:
            return out


def compare_logits(
    tt_logits: torch.Tensor,
    hf_logits: torch.Tensor,
    tolerances: Sequence[Tuple[float, float, str]] = (
        (1e-4, 1e-6, "Very Strict"),
        (1e-2, 1e-4, "Strict"),
        (1e-1, 1e-2, "Moderate"),
    ),
) -> LogitsComparison:
    # Apply softmax to convert logits to probabilities
    hf_logits = torch.nn.functional.softmax(hf_logits.float(), dim=-1)
    tt_logits = torch.nn.functional.softmax(tt_logits.float(), dim=-1)

    diff = torch.abs(tt_logits - hf_logits)
    max_abs = float(torch.max(diff))
    mean_abs = float(torch.mean(diff))
    rel = diff / (torch.abs(tt_logits) + 1e-8)
    max_rel = float(torch.max(rel))
    mean_rel = float(torch.mean(rel))

    results = []
    any_match = False
    for rtol, atol, name in tolerances:
        match = torch.allclose(tt_logits, hf_logits, rtol=rtol, atol=atol)
        results.append((rtol, atol, name, bool(match)))
        if match:
            any_match = True
            break

    sample_diffs = None
    sys_offset = None
    if not any_match:
        flat = (tt_logits - hf_logits).flatten()
        sample_diffs = flat[:25]
        sys_offset = (float(torch.mean(flat)), float(torch.std(flat)))

    return LogitsComparison(max_abs, mean_abs, max_rel, mean_rel, results, sample_diffs, sys_offset)


def generate_text(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    do_sample = temperature > 0
    temp_arg = temperature if do_sample else None
    with torch.no_grad():
        return model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temp_arg,
            top_k=top_k,
        )


def print_logits_comparison(comp: LogitsComparison):
    print("\n" + "="*70)
    print("LOGITS COMPARISON")
    print("="*70)
    print(f"Max abs diff: {comp.max_abs_diff:.6f}")
    print(f"Mean abs diff: {comp.mean_abs_diff:.6f}")
    print(f"Max rel diff: {comp.max_rel_diff:.6f}")
    print(f"Mean rel diff: {comp.mean_rel_diff:.6f}\n")
    print("Tolerance tests:")
    for rtol, atol, name, match in comp.allclose_results:
        print(f"  {'✅' if match else '❌'} {name} (rtol={rtol}, atol={atol})")
    if comp.sample_diffs is not None:
        print("\n🔍 Sample diffs (first 25):")
        for v in comp.sample_diffs.tolist():
            print(f"  {v:.6f}")
        mean, std = comp.systematic_offset
        print(f"\nSystematic offset: mean={mean:.6f}, std={std:.6f}")


def print_generation(title: str, outputs: torch.Tensor, tokenizer):
    text = tokenizer.decode(outputs[0].tolist())
    print("\n" + "="*60)
    print(title)
    print("="*60)
    print(text)
    print("="*60)


def print_generation_comparison(
    tt_out: torch.Tensor,
    hf_out: torch.Tensor,
    tokenizer,
    prompt_len: int,
):
    tt_tokens = tt_out[0][prompt_len:].tolist()
    hf_tokens = hf_out[0][prompt_len:].tolist()
    n = min(len(tt_tokens), len(hf_tokens))
    matches = sum(1 for i in range(n) if tt_tokens[i] == hf_tokens[i])
    print("\n" + "="*70)
    print("GENERATION COMPARISON")
    print("="*70)
    print(f"Match rate: {matches}/{n} ({matches/n*100:.1f}%)")
    if matches != n or len(tt_tokens) != len(hf_tokens):
        print("First mismatches:")
        for i in range(min(10, n)):
            if tt_tokens[i] != hf_tokens[i]:
                tt_txt = tokenizer.decode([tt_tokens[i]])
                hf_txt = tokenizer.decode([hf_tokens[i]])
                print(f"  Pos {i}: TT='{tt_txt}' vs HF='{hf_txt}'")


@record
def test_generate(args: GenerateConfig):
    init_logger()

    if not args.hf_model_path and not args.tt_config:
        raise ValueError("Either hf_model_path or tt_config must be provided.")
    if not args.prompt:
        logger.warning("Empty prompt; generating from scratch.")

    # --- Common setup: tokenizer & inputs ---
    if args.hf_model_path:
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path)
        input_ids = tokenizer.encode(args.prompt, add_special_tokens=False, return_tensors="pt")
        print(input_ids)
    if args.tt_config:
        config_mgr = ConfigManager()
        config = config_mgr.parse_args([
            f"--job.config_file={args.tt_config}",
            f"--model.tokenizer_path={args.tt_tokenizer_path}",
        ])
        train_spec = get_train_spec(config.model.name)

    # --- HuggingFace model (optional) ---
    hf_model = None
    hf_logits = None
    hf_out = None
    if args.hf_model_path: # NOTE: comment this block out for rapid tt testing
        hf_device = torch.device(f"{device_type}:0")
        hf_model = load_hf_model(args.hf_model_path, hf_device)
        print("\n" + "="*60)
        print("HUGGINGFACE MODEL ARCHITECTURE:")
        print(hf_model)
        print("="*60)
        print_param_dtypes_first_block(hf_model)
        print("="*60)

        hf_in = input_ids.to(hf_device)
        hf_logits = get_logits(hf_model, hf_in).to(input_ids.device)
        print(f"hf_logits: {hf_logits[:, :, 42069:42072]}")
        hf_out = generate_text(
            hf_model, hf_in,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_k=args.top_k,
        ).to(input_ids.device)

    # --- TorchTitan model (optional) ---
    tt_model = None
    tt_logits = None
    tt_out = None
    if args.tt_config:
        # (Original TT setup: distributed, device, checkpoint load, etc.)
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        device = torch.device(f"{device_type}:1")
        device_module.set_device(device)
        dist_utils.set_determinism(None, device, args.seed, args.deterministic)

        # instantiate & load TT model
        model_args = train_spec.config[config.model.flavor]
        model_args.update_from_config(config, tokenizer)
        init_dev = "meta" if world_size > 1 else device
        with torch.device(init_dev):
            tt_model = train_spec.cls(model_args)
        if world_size > 1:
            # parallelize if needed
            pass
        print("\n" + "="*60)
        print("TORCHTITAN MODEL ARCHITECTURE:")
        print(tt_model)
        print("="*60)
        print_param_dtypes_first_block(tt_model)
        print("="*60)

        tt_model.eval()
        if args.tt_checkpoint_path: # only load checkpoint if provided
            tt_state = tt_model.state_dict()
            tt_state.pop("freqs_cis", None)
            state = {"model": tt_state}
            dcp.load(state, checkpoint_id=args.tt_checkpoint_path)

        tt_logits = get_logits(tt_model, input_ids.to(device)).to(hf_logits.device if hf_logits is not None else device)
        print(f"✅ Torchtitan model forward pass succeeded: {tt_logits.shape=}")
        print(f"tt_logits: {tt_logits[:, :, 42069:42072]}")

        tt_out = tt_generate_text(
            tt_model, input_ids.to(device),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
        )

    # --- Logits comparison (if both present) ---
    if hf_logits is not None and tt_logits is not None:
        comp = compare_logits(tt_logits, hf_logits)
        print_logits_comparison(comp)

    # --- Print generations ---
    if hf_out is not None:
        print_generation("HUGGINGFACE MODEL OUTPUT:", hf_out, tokenizer)
    if tt_out is not None:
        print_generation("TORCHTITAN MODEL OUTPUT:", tt_out, tokenizer)

    # --- Generation comparison ---
    if hf_out is not None and tt_out is not None:
        prompt_len = input_ids.size(1)
        print_generation_comparison(tt_out, hf_out, tokenizer, prompt_len)


if __name__ == "__main__":
    args = tyro.cli(GenerateConfig)
    test_generate(args)
