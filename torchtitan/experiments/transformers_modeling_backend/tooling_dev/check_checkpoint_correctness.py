"""
python tooling_dev/check_checkpoint_correctness.py \
    --tt-config torchtitan/models/llama3/train_configs/debug_model.toml \
    --tt-checkpoint outputs/checkpoint/step-10 \
    --hf-checkpoint outputs/checkpoint/step-10-hf
"""

import argparse
import tempfile
from pathlib import Path
from typing import Optional

import torch
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from torch.distributed.checkpoint import HuggingFaceStorageReader, HuggingFaceStorageWriter
from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.config import ConfigManager
from torchtitan.protocols.train_spec import get_train_spec
from torchtitan.tools.logging import logger

device_type = "cuda" if torch.cuda.is_available() else "cpu"


# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def loss_fn(logits1, logits2):
    """Calculate KL Divergence between two sets of logits."""
    probs1 = F.log_softmax(logits1, dim=-1)
    probs2 = F.softmax(logits2, dim=-1)
    kl_loss = F.kl_div(probs1, probs2, reduction="batchmean")
    return kl_loss


@torch.no_grad
def forward_tt(config_path: str, checkpoint_path: str, test_set, config_overrides=None):
    """Run forward pass with TorchTitan model."""
    config_manager = ConfigManager()
    
    # Build config args
    config_args = [f"--job.config_file={config_path}"]
    if config_overrides:
        config_args.extend(config_overrides)
    
    config = config_manager.parse_args(config_args)
    train_spec = get_train_spec(config.model.name)

    model_args = train_spec.model_args[config.model.flavor]
    model_args.update_from_config(config)

    model = train_spec.model_cls(model_args)

    # Materialize model
    device = torch.device(device_type)
    model.to_empty(device=device)
    model.init_weights(buffer_device=device)
    model.eval()

    modelWrapper = ModelWrapper(model)
    state_dict = modelWrapper._get_state_dict()

    # Checkpoint Loading
    logger.info(f"Loading checkpoint at: {checkpoint_path}")
    dcp.load(state_dict, checkpoint_id=checkpoint_path)

    output_list = []
    for prompt in test_set:
        input_ids = prompt.to(device_type)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        predictions = model(input_ids)[:, -1, :].unsqueeze(1)
        output_list.append(predictions)

    del model
    torch.cuda.empty_cache()
    return output_list


@torch.no_grad
def load_checkpoint_via_hf_roundtrip(
    config_path: str,
    hf_checkpoint_path: str,
    config_overrides=None
):
    """Load a checkpoint that was converted to HF format by doing HF->TT conversion."""
    config_manager = ConfigManager()
    
    # Build config args
    config_args = [f"--job.config_file={config_path}"]
    if config_overrides:
        config_args.extend(config_overrides)
    
    config = config_manager.parse_args(config_args)
    train_spec = get_train_spec(config.model.name)

    model_args = train_spec.model_args[config.model.flavor]
    model_args.update_from_config(config)

    with torch.device("cpu"):
        model = train_spec.model_cls(model_args)
    model = ModelWrapper(model)

    sd_adapter = train_spec.state_dict_adapter(model_args, None)
    if sd_adapter is None:
        raise ValueError("State dict adapter is required for HF conversion")

    # Get state dict in TT format with allocated memory
    state_dict = model._get_state_dict()
    
    # Convert empty state dict to HF format so that HF weights can be loaded into it
    hf_state_dict = sd_adapter.to_hf(state_dict)
    
    # Load HF format checkpoint
    logger.info(f"Loading HF checkpoint from: {hf_checkpoint_path}")
    dcp.load(
        hf_state_dict,
        storage_reader=HuggingFaceStorageReader(path=hf_checkpoint_path),
    )
    
    # Convert state dict format back hf->tt
    state_dict = sd_adapter.from_hf(hf_state_dict)
    
    return state_dict


@torch.no_grad
def forward_tt_from_hf(
    config_path: str,
    hf_checkpoint_path: str,
    test_set,
    config_overrides=None
):
    """Run forward pass loading from HF-format checkpoint."""
    config_manager = ConfigManager()
    
    # Build config args
    config_args = [f"--job.config_file={config_path}"]
    if config_overrides:
        config_args.extend(config_overrides)
    
    config = config_manager.parse_args(config_args)
    train_spec = get_train_spec(config.model.name)

    model_args = train_spec.model_args[config.model.flavor]
    model_args.update_from_config(config)

    # Load state dict via HF roundtrip (on CPU)
    state_dict = load_checkpoint_via_hf_roundtrip(
        config_path, hf_checkpoint_path, config_overrides
    )

    # Create model for inference
    model = train_spec.model_cls(model_args)
    device = torch.device(device_type)
    
    # Load the state dict on CPU first
    model.load_state_dict(state_dict, assign=True)
    
    # Now move the model with loaded weights to device
    model.to(device)
    model.eval()

    output_list = []
    for prompt in test_set:
        input_ids = prompt.to(device)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        predictions = model(input_ids)[:, -1, :].unsqueeze(1)
        output_list.append(predictions)

    del model
    torch.cuda.empty_cache()
    return output_list


def run_comparison(
    tt_config_path: str,
    tt_checkpoint_path: str,
    hf_checkpoint_path: str,
    prompt_len: int = 8,
    test_size: int = 100,
    config_overrides: Optional[list] = None,
):
    """Run numerical comparison between original DCP and converted HF checkpoints."""
    
    # Build tokenizer
    config_manager = ConfigManager()
    config_args = [f"--job.config_file={tt_config_path}"]
    if config_overrides:
        config_args.extend(config_overrides)
    
    config = config_manager.parse_args(config_args)
    train_spec = get_train_spec(config.model.name)
    tokenizer = train_spec.build_tokenizer_fn(config)

    # Build test set of randomly generated token ids
    print(f"{Colors.OKCYAN}Building test set with {test_size} samples of length {prompt_len}...{Colors.ENDC}")
    test_set = [
        torch.randint(
            0,
            tokenizer.get_vocab_size(),
            (1, prompt_len),
        )
        for _ in range(test_size)
    ]

    # Run original DCP checkpoint
    print(f"{Colors.OKBLUE}Running TorchTitan model from DCP checkpoint...{Colors.ENDC}")
    dcp_outputs = forward_tt(tt_config_path, tt_checkpoint_path, test_set, config_overrides)
    print(f"{Colors.OKGREEN}✓ DCP checkpoint inference complete{Colors.ENDC}")

    # Run HF-converted checkpoint (roundtrip: DCP -> HF -> DCP)
    print(f"{Colors.OKBLUE}Running TorchTitan model from HF-converted checkpoint...{Colors.ENDC}")
    hf_roundtrip_outputs = forward_tt_from_hf(tt_config_path, hf_checkpoint_path, test_set, config_overrides)
    print(f"{Colors.OKGREEN}✓ HF-converted checkpoint inference complete{Colors.ENDC}")

    # Calculate loss
    print(f"{Colors.OKCYAN}Calculating KL divergence...{Colors.ENDC}")
    total_loss = 0
    for dcp_out, hf_out in zip(dcp_outputs, hf_roundtrip_outputs):
        total_loss += loss_fn(dcp_out, hf_out)
    avg_loss = total_loss / len(test_set)

    print(f"{Colors.OKGREEN}✓ Comparison complete{Colors.ENDC}")
    return avg_loss.item()


def main():
    parser = argparse.ArgumentParser(
        description="Test checkpoint conversion correctness by comparing DCP and HF-converted outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare original DCP checkpoint with HF-converted version
  %(prog)s \\
      --tt-config torchtitan/models/llama3/train_configs/debug_model.toml \\
      --tt-checkpoint outputs/checkpoint/step-10 \\
      --hf-checkpoint outputs/checkpoint/step-10-hf

  # With custom test parameters
  %(prog)s \\
      --tt-config torchtitan/models/llama3/train_configs/debug_model.toml \\
      --tt-checkpoint outputs/checkpoint/step-10 \\
      --hf-checkpoint outputs/checkpoint/step-10-hf \\
      --prompt-len 16 \\
      --test-size 50

  # Override config values
  %(prog)s \\
      --tt-config torchtitan/models/llama3/train_configs/debug_model.toml \\
      --tt-checkpoint outputs/checkpoint/step-10 \\
      --hf-checkpoint outputs/checkpoint/step-10-hf \\
      --config-override "--model.hf_assets_path=./assets/hf/Llama-3.2-1B"
        """
    )

    parser.add_argument(
        "--tt-config",
        type=str,
        required=True,
        help="Path to TorchTitan config file (TOML)"
    )
    parser.add_argument(
        "--tt-checkpoint",
        type=str,
        required=True,
        help="Path to original TorchTitan DCP checkpoint directory"
    )
    parser.add_argument(
        "--hf-checkpoint",
        type=str,
        required=True,
        help="Path to HF-converted checkpoint directory"
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=8,
        help="Length of test prompts (default: 8)"
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=100,
        help="Number of test samples (default: 100)"
    )
    parser.add_argument(
        "--config-override",
        type=str,
        action="append",
        help="Override config values (can be specified multiple times)"
    )

    args = parser.parse_args()

    # Print header
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}Checkpoint Conversion Numerical Test{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}\n")

    # Run comparison
    avg_loss = run_comparison(
        tt_config_path=args.tt_config,
        tt_checkpoint_path=args.tt_checkpoint,
        hf_checkpoint_path=args.hf_checkpoint,
        prompt_len=args.prompt_len,
        test_size=args.test_size,
        config_overrides=args.config_override,
    )

    # Print colored results
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}Checkpoint Conversion Test Results{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}TT Config:{Colors.ENDC}        {args.tt_config}")
    print(f"{Colors.OKCYAN}DCP Checkpoint:{Colors.ENDC}   {args.tt_checkpoint}")
    print(f"{Colors.OKCYAN}HF Checkpoint:{Colors.ENDC}    {args.hf_checkpoint}")
    print(f"{Colors.OKCYAN}Prompt Length:{Colors.ENDC}    {args.prompt_len}")
    print(f"{Colors.OKCYAN}Test Size:{Colors.ENDC}        {args.test_size}")
    print(f"{Colors.BOLD}{Colors.OKBLUE}Average KL Div:{Colors.ENDC}   {avg_loss:.2e}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")

    # Interpretation with colors
    if avg_loss < 1e-10:
        print(f"{Colors.BOLD}{Colors.OKGREEN}✓ EXCELLENT: Checkpoints produce identical outputs (perfect conversion){Colors.ENDC}")
    elif avg_loss < 1e-6:
        print(f"{Colors.OKGREEN}✓ VERY GOOD: Checkpoints produce nearly identical outputs{Colors.ENDC}")
    elif avg_loss < 1e-3:
        print(f"{Colors.OKGREEN}✓ GOOD: Checkpoints produce very similar outputs{Colors.ENDC}")
    elif avg_loss < 0.01:
        print(f"{Colors.WARNING}⚠ WARNING: Some divergence detected in conversion{Colors.ENDC}")
    else:
        print(f"{Colors.BOLD}{Colors.FAIL}✗ FAILURE: Significant divergence - conversion may be incorrect{Colors.ENDC}")

    return 0 if avg_loss < 0.01 else 1


if __name__ == "__main__":
    exit(main())