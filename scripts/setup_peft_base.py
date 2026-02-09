#!/usr/bin/env python3
"""
Setup a PEFT LoRA adapter base for vLLM inference.

This script checks if a PEFT adapter with the specified configuration already exists,
and if not, creates one. The adapter is saved with empty/random weights since only
the structure is needed - the actual trained weights come from torchtitan.

Usage:
    python scripts/setup_peft_base.py \
        --base-model moonshotai/Kimi-K2-Instruct \
        --rank 16 \
        --alpha 1 \
        --dropout 0.0

    # Or with a torchtitan config file:
    python scripts/setup_peft_base.py --config-file path/to/config.toml
"""

import argparse
import hashlib
import io
import json
import os
import stat
import sys
from contextlib import redirect_stdout
from pathlib import Path

import torch
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoConfig, AutoModelForCausalLM


DEFAULT_PEFT_BASE_DIR = "/home/shared/peft_bases"

# Standard target modules for transformer models
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def get_config_hash(
    base_model: str,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: list[str],
) -> str:
    """Generate a short hash of the LoRA configuration for unique identification."""
    config_str = json.dumps(
        {
            "base_model": base_model,
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout,
            "target_modules": sorted(target_modules),
        },
        sort_keys=True,
    )
    return hashlib.sha256(config_str.encode()).hexdigest()[:12]


def get_peft_path(
    base_dir: str,
    base_model: str,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: list[str],
) -> Path:
    """
    Generate the path for a PEFT adapter based on its configuration.

    Path structure: {base_dir}/{model_name_safe}/r{rank}_a{alpha}_d{dropout}_{hash}
    """
    # Sanitize model name for filesystem
    model_name_safe = base_model.replace("/", "__").replace("\\", "__")

    # Create a readable but unique directory name
    config_hash = get_config_hash(base_model, rank, alpha, dropout, target_modules)
    dir_name = f"r{rank}_a{alpha}_d{dropout}_{config_hash}"

    return Path(base_dir) / model_name_safe / dir_name


def create_empty_model_from_config(model_name: str):
    """
    Create a model with empty weights from config only.
    This avoids downloading/loading the actual model weights.
    """
    print(f"Loading config for {model_name}...", file=sys.stderr)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    print(
        "Creating empty model from config (no pretrained weights)...", file=sys.stderr
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    return model, config


def create_peft_adapter(
    output_dir: Path,
    base_model: str,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: list[str],
) -> None:
    """Create and save a PEFT LoRA adapter with empty weights."""

    # Create LoRA config - no modules_to_save, just LoRA adapters
    lora_config = LoraConfig(
        base_model_name_or_path=base_model,
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        modules_to_save=[],  # Explicitly empty - no full modules saved
    )

    print("LoRA Config:", file=sys.stderr)
    print(f"  rank: {rank}", file=sys.stderr)
    print(f"  alpha: {alpha}", file=sys.stderr)
    print(f"  dropout: {dropout}", file=sys.stderr)
    print(f"  target_modules: {target_modules}", file=sys.stderr)

    # Create empty model
    model, config = create_empty_model_from_config(base_model)

    # Apply LoRA - move to CPU with empty weights first
    print("Applying LoRA to model...", file=sys.stderr)
    model = model.to_empty(device="cpu")
    peft_model = get_peft_model(model, lora_config)

    # Print trainable parameters info (redirect to stderr to avoid polluting stdout for eval)
    f = io.StringIO()
    with redirect_stdout(f):
        peft_model.print_trainable_parameters()
    print(f.getvalue(), file=sys.stderr, end="")

    # Save the adapter
    print(f"Saving LoRA adapter to {output_dir}...", file=sys.stderr)
    output_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(output_dir)

    # Also save a metadata file for easy inspection
    metadata = {
        "base_model": base_model,
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "target_modules": target_modules,
    }
    with open(output_dir / "torchtitan_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Set group read/write permissions on all created files
    for path in output_dir.rglob("*"):
        if path.is_file():
            current_mode = path.stat().st_mode
            path.chmod(current_mode | stat.S_IRGRP | stat.S_IWGRP)
    # Also set on the directory itself and parent directories we created
    for parent in [output_dir] + list(output_dir.parents):
        if not parent.exists():
            break
        try:
            current_mode = parent.stat().st_mode
            parent.chmod(current_mode | stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP)
        except PermissionError:
            break  # Stop at directories we don't own

    print(f"Done! LoRA adapter saved to {output_dir}", file=sys.stderr)


def setup_peft_base(
    base_model: str,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: list[str],
    base_dir: str = DEFAULT_PEFT_BASE_DIR,
    force: bool = False,
) -> tuple[Path, str]:
    """
    Setup a PEFT base adapter, creating it if it doesn't exist.

    Returns:
        tuple of (peft_path, base_model)
    """
    peft_path = get_peft_path(
        base_dir, base_model, rank, alpha, dropout, target_modules
    )

    # Check if adapter already exists
    adapter_config_path = peft_path / "adapter_config.json"

    if adapter_config_path.exists() and not force:
        print(f"PEFT adapter already exists at {peft_path}", file=sys.stderr)
    else:
        if force and adapter_config_path.exists():
            print(f"Force recreating PEFT adapter at {peft_path}", file=sys.stderr)
        else:
            print(f"Creating new PEFT adapter at {peft_path}", file=sys.stderr)

        create_peft_adapter(
            output_dir=peft_path,
            base_model=base_model,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
        )

    return peft_path, base_model


def main():
    parser = argparse.ArgumentParser(
        description="Setup a PEFT LoRA adapter base for vLLM inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create adapter with explicit parameters
    python scripts/setup_peft_base.py \\
        --base-model moonshotai/Kimi-K2-Instruct \\
        --rank 16 --alpha 1 --dropout 0.0

    # Use torchtitan config file
    python scripts/setup_peft_base.py --config-file configs/my_lora_run.toml

Output (printed to stdout for bash scripts):
    PEFT_PATH=/home/shared/peft_bases/moonshotai__Kimi-K2-Instruct/r16_a1_d0.0_abc123
    BASE_MODEL=moonshotai/Kimi-K2-Instruct
""",
    )

    # Config file option
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to torchtitan config file (reads peft.* and model.hf_assets_path)",
    )

    # Explicit config options
    parser.add_argument(
        "--base-model",
        type=str,
        help="Base model name or path (e.g., moonshotai/Kimi-K2-Instruct)",
    )
    parser.add_argument(
        "--rank",
        "-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--alpha",
        "-a",
        type=float,
        default=1.0,
        help="LoRA alpha (default: 1.0)",
    )
    parser.add_argument(
        "--dropout",
        "-d",
        type=float,
        default=0.0,
        help="LoRA dropout (default: 0.0)",
    )
    parser.add_argument(
        "--target-modules",
        type=str,
        nargs="+",
        default=None,
        help="Target modules for LoRA (default: q_proj k_proj v_proj o_proj gate_proj up_proj down_proj)",
    )

    # Path options
    parser.add_argument(
        "--base-dir",
        type=str,
        default=DEFAULT_PEFT_BASE_DIR,
        help=f"Base directory for PEFT adapters (default: {DEFAULT_PEFT_BASE_DIR})",
    )

    # Behavior options
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreation of adapter even if it exists",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only output the path variables (suppress progress messages)",
    )

    args = parser.parse_args()

    # Load config from file if provided
    if args.config_file:
        from torchtitan.config.manager import ConfigManager

        config_manager = ConfigManager()
        config = config_manager.parse_args(["--job.config_file", args.config_file])

        base_model = config.model.hf_assets_path
        rank = config.peft.lora_rank
        alpha = config.peft.lora_alpha
        dropout = config.peft.lora_dropout
        target_modules = args.target_modules or DEFAULT_TARGET_MODULES
    else:
        if not args.base_model:
            parser.error("--base-model is required when not using --config-file")

        base_model = args.base_model
        rank = args.rank
        alpha = args.alpha
        dropout = args.dropout
        target_modules = args.target_modules or DEFAULT_TARGET_MODULES

    # Suppress stderr if quiet mode
    if args.quiet:
        sys.stderr = open(os.devnull, "w")

    # Setup the PEFT base
    peft_path, base_model = setup_peft_base(
        base_model=base_model,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
        base_dir=args.base_dir,
        force=args.force,
    )

    # Output for bash scripts (to stdout)
    print(f"PEFT_PATH={peft_path}")
    print(f"BASE_MODEL={base_model}")
    print(f"LORA_RANK={rank}")
    print(f"LORA_ALPHA={alpha}")
    print(f"LORA_DROPOUT={dropout}")


if __name__ == "__main__":
    main()
