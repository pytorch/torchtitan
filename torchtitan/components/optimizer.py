# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Any, Generic, Iterator, TypeVar

import torch
import torch.distributed.tensor
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor import DTensor, Replicate
from torch.optim import Optimizer

from torchtitan.components.ft import FTManager, has_torchft
from torchtitan.config import LRMultipliers, Optimizer as OptimizerConfig, WeightDecayMultipliers
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger

__all__ = [
    "OptimizersContainer",
    "OptimizersContainerWithParamGroups",
    "CPUOffloadOptimizersContainer",
    "build_optimizers",
    "build_optimizers_with_moe_load_balancing",
]

# Check if DeepSpeed CPU Adam is available
HAS_DEEPSPEED_CPU_ADAM = False
DeepSpeedCPUAdam = None
try:
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    # Test if it can actually load the C++ extension
    # This may fail due to CUDA version mismatches
    HAS_DEEPSPEED_CPU_ADAM = True
except (ImportError, ModuleNotFoundError, RuntimeError):
    # ImportError/ModuleNotFoundError: DeepSpeed not installed
    # RuntimeError: C++ extension failed to load (CUDA mismatch, etc.)
    pass


# Pre-compiled regex patterns for parameter group classification
# IMPORTANT: Pattern order matters!
# - 'bias' must be FIRST to catch all bias parameters before other patterns
# - 'mhc' must come before 'norms' since mhc modules contain norm submodules
# Note: norm.weight is the final RMSNorm before output, grouped with output
# Patterns handle optional _checkpoint_wrapped_module wrapper from activation checkpointing
import re
_PARAM_GROUP_PATTERNS = {
    # Bias pattern FIRST - catches all bias parameters
    'bias': re.compile(r'\.bias$'),
    'embeddings': re.compile(r'^tok_embeddings\.weight$'),
    'output': re.compile(r'^(output\.weight|norm\.weight)$'),
    'attention': re.compile(r'layers\.\d+\.(_checkpoint_wrapped_module\.)?attention\.(wq|wk|wv|wo|sinks)'),
    'experts': re.compile(r'layers\.\d+\.(_checkpoint_wrapped_module\.)?moe\.experts\.mlp[12]_weight'),
    'routers': re.compile(r'layers\.\d+\.(_checkpoint_wrapped_module\.)?moe\.router\.gate'),
    # mHC (Manifold-Constrained Hyper-Connections) parameters - must be before 'norms'
    # Matches: mhc_attn/mhc_moe modules (phi_*, b_*, alpha_*, norm) and stream projections
    'mhc': re.compile(
        r'(layers\.\d+\.(_checkpoint_wrapped_module\.)?(mhc_attn|mhc_moe)\.'
        r'(phi_pre|phi_post|phi_res|b_pre|b_post|b_res|alpha_pre|alpha_post|alpha_res|norm\.weight))|'
        r'^(stream_expander|stream_collapser)\.weight$'
    ),
    'norms': re.compile(r'layers\.\d+\.(_checkpoint_wrapped_module\.)?(attention_norm|ffn_norm)\.weight'),
}

# Field names for multiplier classes
_LR_MULTIPLIER_FIELDS = ['embeddings', 'output', 'attention', 'experts', 'routers', 'norms', 'mhc']
_WD_MULTIPLIER_FIELDS = ['embeddings', 'output', 'attention', 'experts', 'routers', 'norms', 'mhc', 'bias']


def _has_non_default_multipliers(multipliers, field_names: list[str]) -> bool:
    """Check if any multiplier field differs from the default value of 1.0."""
    if multipliers is None:
        return False
    return any(getattr(multipliers, name, 1.0) != 1.0 for name in field_names)


def _log_non_default_multipliers(multipliers, field_names: list[str], label: str) -> None:
    """Log non-default multiplier values."""
    if multipliers is None:
        return
    non_default = [
        (name, getattr(multipliers, name))
        for name in field_names
        if getattr(multipliers, name) != 1.0
    ]
    if non_default:
        logger.info(f"Using differential {label}:")
        for name, mult in non_default:
            logger.info(f"  {name}: {mult:.1f}x")


def classify_parameters_for_groups(
    model_parts: list[nn.Module],
    base_lr: float,
    lr_multipliers: LRMultipliers,
    base_weight_decay: float = 0.0,
    weight_decay_multipliers: WeightDecayMultipliers | None = None,
) -> list[dict[str, Any]]:
    """Classify parameters into groups with different learning rates and weight decay.

    Args:
        model_parts: List of model parts containing parameters
        base_lr: Base learning rate
        lr_multipliers: LR multipliers config object
        base_weight_decay: Base weight decay value
        weight_decay_multipliers: Weight decay multipliers config object (optional)

    Returns:
        List of param_groups dicts for optimizer, each with:
        - 'params': list of parameters
        - 'lr': effective learning rate for this group
        - 'weight_decay': effective weight decay for this group
        - 'group_name': string identifier for logging
    """
    # Helper to get weight decay for a group
    def get_weight_decay(group_name: str) -> float:
        if weight_decay_multipliers is None:
            return base_weight_decay
        mult = getattr(weight_decay_multipliers, group_name, 1.0)
        return base_weight_decay * mult

    # Helper to get LR for a group
    def get_lr(group_name: str) -> float:
        mult = getattr(lr_multipliers, group_name, 1.0)
        return base_lr * mult

    # Initialize groups (including default for unmatched params)
    param_groups_dict = {
        group_name: {
            'params': [],
            'lr': get_lr(group_name),
            'weight_decay': get_weight_decay(group_name),
            'group_name': group_name,
        }
        for group_name in _PARAM_GROUP_PATTERNS.keys()
    }

    # Add default group for any unmatched parameters (uses base lr/wd with 1.0 multiplier)
    param_groups_dict['default'] = {
        'params': [],
        'lr': base_lr,  # Use base_lr directly (1.0x multiplier)
        'weight_decay': base_weight_decay,  # Use base weight decay directly
        'group_name': 'default',
    }

    # Track unmatched parameters
    unmatched_params = []
    param_count_by_group = {k: 0 for k in _PARAM_GROUP_PATTERNS.keys()}
    param_count_by_group['default'] = 0

    # Classify each parameter
    for model_part in model_parts:
        for name, param in model_part.named_parameters():
            if not param.requires_grad:
                continue

            matched = False
            for group_name, pattern in _PARAM_GROUP_PATTERNS.items():
                if pattern.search(name):
                    param_groups_dict[group_name]['params'].append(param)
                    param_count_by_group[group_name] += 1
                    matched = True
                    break

            if not matched:
                # Unmatched params go to default group
                unmatched_params.append((name, param))
                param_groups_dict['default']['params'].append(param)
                param_count_by_group['default'] += 1

    # Log parameter classification as a single message
    using_differential_wd = weight_decay_multipliers is not None
    lines = ["Parameter Group Classification:"]
    for group_name, group_dict in param_groups_dict.items():
        count = param_count_by_group[group_name]
        if count == 0:
            continue
        lr = group_dict['lr']
        wd = group_dict['weight_decay']
        lr_mult = lr / base_lr if base_lr > 0 else 1.0
        if using_differential_wd:
            wd_mult = wd / base_weight_decay if base_weight_decay > 0 else (0.0 if wd == 0 else 1.0)
            lines.append(f"  {group_name:12s}: {count:4d} params, lr={lr:.2e} ({lr_mult:.1f}x), wd={wd:.2e} ({wd_mult:.1f}x)")
        else:
            lines.append(f"  {group_name:12s}: {count:4d} params, lr={lr:.2e} ({lr_mult:.1f}x)")
    logger.info("\n".join(lines))

    if unmatched_params:
        unmatched_names = [name for name, _ in unmatched_params[:10]]
        suffix = f" ... and {len(unmatched_params) - 10} more" if len(unmatched_params) > 10 else ""
        logger.warning(
            f"Found {len(unmatched_params)} unmatched parameters assigned to 'default' group: "
            f"{unmatched_names}{suffix}"
        )

    # Filter out empty groups and return
    param_groups = [
        group for group in param_groups_dict.values()
        if len(group['params']) > 0
    ]

    return param_groups


# Backward compatibility alias
def classify_parameters_for_lr_groups(
    model_parts: list[nn.Module],
    base_lr: float,
    lr_multipliers: LRMultipliers,
) -> list[dict[str, Any]]:
    """Backward compatible wrapper for classify_parameters_for_groups."""
    return classify_parameters_for_groups(
        model_parts, base_lr, lr_multipliers
    )


if has_torchft:
    import torchft as ft


T = TypeVar("T", bound=Optimizer)


class OptimizersContainer(Optimizer, Stateful, Generic[T]):
    """A container for multiple optimizers.

    This class is used to wrap multiple optimizers into a single object that can be
    used to reduce the complexity of the training loop. This mimics the behavior of
    ``torch.optim.Optimizer``. This class currently only supports ``Adam`` and ``AdamW``.

    **Note**
    Users who want to customize the optimizer behavior can inherit from this class and
    extend the functionality as needed. The following methods must follow the same signature
    as ``torch.optim.Optimizer`` class: ``step()``, ``zero_grad()``, ``state_dict()``,
    ``load_state_dict()``.

    **Limitations**
    This class assumes that all the optimizers are the same type and have the same
    configurations. With this assumption, TorchTitan can support lr scheduler resharding
    (e.g., loading a checkpoint with a different number of GPUs and/or different
    parallelization strategy). Note that ``get_optimizer_state_dict`` already enables the
    resharding for the optimizer state but not for the lr scheduler state, hence the limitation.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizer_kwargs (Dict[str, Any]): Keyword arguments for the optimizers.
        name (str): Name of the optimizers.
    """

    optimizers: list[T]
    model_parts: list[nn.Module]

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: dict[str, Any],
    ) -> None:
        all_params = []
        self.optimizers = []
        self.model_parts = model_parts
        for model in self.model_parts:
            params = [p for p in model.parameters() if p.requires_grad]
            self.optimizers.append(optimizer_cls(params, **optimizer_kwargs))
            all_params.extend(params)
        self._validate_length(len(self.model_parts))
        self._post_init(all_params, optimizer_kwargs)

    def __iter__(self) -> Iterator[T]:
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    # pyrefly: ignore [bad-override]
    def step(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(*args, **kwargs)

    def state_dict(self) -> dict[str, Any]:
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return {
            k: v
            for sd in map(func, self.model_parts, self.optimizers)
            for k, v in sd.items()
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        list(map(func, self.model_parts, self.optimizers))

    def _validate_length(self, expected_length: int) -> None:
        assert expected_length == len(self.optimizers), (
            "Must pass one optimizer per model part or per param if "
            "using OptimizersInBackwardContainer."
        )

    def _post_init(
        self, all_params: list[nn.Parameter], optimizer_kwargs: dict[str, Any]
    ) -> None:
        # We need to call Optimizer.__init__() to initialize some necessary optimizer
        # functionality such as hooks.
        Optimizer.__init__(self, all_params, optimizer_kwargs)


class OptimizersContainerWithParamGroups(OptimizersContainer):
    """OptimizersContainer with per-parameter-group learning rates and weight decay.

    This class extends OptimizersContainer to support differential learning rates
    and weight decay for different parameter groups (embeddings, attention, experts, etc.).
    """

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: dict[str, Any],
        lr_multipliers: LRMultipliers,
        weight_decay_multipliers: WeightDecayMultipliers | None = None,
    ) -> None:
        all_params = []
        self.optimizers = []
        self.model_parts = model_parts

        base_lr = optimizer_kwargs.get('lr', 1e-5)
        base_weight_decay = optimizer_kwargs.get('weight_decay', 0.0)

        # Classify parameters into groups
        param_groups = classify_parameters_for_groups(
            model_parts,
            base_lr,
            lr_multipliers,
            base_weight_decay,
            weight_decay_multipliers,
        )

        # Collect all params for _post_init
        for group in param_groups:
            all_params.extend(group['params'])

        # Remove lr and weight_decay from kwargs since they're per-group now
        base_kwargs = {
            k: v for k, v in optimizer_kwargs.items()
            if k not in ('lr', 'weight_decay')
        }

        # Create a single optimizer with all param groups
        # Note: This differs from OptimizersContainer which creates one optimizer per model_part
        # For differential settings, we need a single optimizer to handle all groups together
        optimizer = optimizer_cls(param_groups, **base_kwargs)
        self.optimizers = [optimizer]

        self._validate_length(1)  # Single optimizer for all param groups
        self._post_init(all_params, optimizer_kwargs)

    def state_dict(self) -> dict[str, Any]:
        """Save optimizer state dict for all model parts.

        Overrides base class because we have a single optimizer for all model parts,
        but the state dict API expects per-model-part handling.
        """
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        # Single optimizer handles all model parts
        combined_state = {}
        for model_part in self.model_parts:
            sd = func(model_part, self.optimizers[0])
            combined_state.update(sd)
        return combined_state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load optimizer state dict for all model parts.

        Overrides base class because we have a single optimizer for all model parts,
        but the state dict API expects per-model-part handling.
        """
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        for model_part in self.model_parts:
            func(model_part, self.optimizers[0])


class OptimizersInBackwardContainer(OptimizersContainer):
    """OptimizersContainer for executing ``optim.step()`` in backward pass.

    This class extend ``OptimizersContainer`` to support optimizer step in
    backward pass. ``step()`` and ``zero_grad()`` are no-op in this class.
    Instead, ``register_post_accumulate_grad_hook`` is used to register a hook to
    execute these methods when the gradient is accumulated.
    """

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: dict[str, Any],
    ) -> None:
        all_params = []
        self.model_parts = model_parts

        optim_dict = {}
        for model in self.model_parts:
            for p in model.parameters():
                if p.requires_grad:
                    optim_dict[p] = optimizer_cls([p], **optimizer_kwargs)
                all_params.append(p)

        def optim_hook(param) -> None:
            optim_dict[param].step()
            optim_dict[param].zero_grad()

        for model in self.model_parts:
            for param in model.parameters():
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(optim_hook)

        self.optimizers = list(optim_dict.values())

        self._validate_length(
            sum(len(list(model.parameters())) for model in self.model_parts)
        )
        self._post_init(all_params, optimizer_kwargs)

    # pyrefly: ignore [bad-override]
    def step(self) -> None:
        pass

    # pyrefly: ignore [bad-override]
    def zero_grad(self) -> None:
        pass


class CPUOffloadOptimizersContainer(OptimizersContainer):
    """OptimizersContainer that offloads optimizer state and computation to CPU.

    This class uses DeepSpeed's CPUAdam optimizer which provides 5-7x speedup
    over naive PyTorch CPU optimizer. Optimizer states (momentum, variance) are
    kept on CPU, and parameter updates happen on CPU before being copied back
    to GPU.

    This is useful for:
    - Training larger models that don't fit in GPU memory with optimizer states
    - Systems where CPU memory is abundant
    - Reducing GPU memory pressure at the cost of some throughput

    The workflow per step:
    1. Gradients are computed on GPU (normal backward pass)
    2. Gradients are copied to CPU
    3. DeepSpeedCPUAdam updates parameters on CPU
    4. Updated parameters are copied back to GPU
    """

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_kwargs: dict[str, Any],
        adamw_mode: bool = True,
        lr_multipliers: LRMultipliers | None = None,
        weight_decay_multipliers: WeightDecayMultipliers | None = None,
    ) -> None:
        all_params = []
        self.model_parts = model_parts
        self.optimizers = []

        # Map between model params (may be DTensor) and CPU shadow params
        self._gpu_to_cpu_param: dict[nn.Parameter, torch.Tensor] = {}
        self._cpu_to_gpu_param: dict[torch.Tensor, nn.Parameter] = {}

        # CUDA events for fine-grained synchronization (avoids full device sync)
        self._grad_copy_event: torch.cuda.Event | None = None
        self._param_copy_event: torch.cuda.Event | None = None

        # Collect all trainable parameters
        for model in self.model_parts:
            for p in model.parameters():
                if p.requires_grad:
                    all_params.append(p)

        # Create CPU shadow copies of parameters
        # For DTensor (FSDP2), we get the local tensor
        cpu_params = []
        for model_param in all_params:
            if isinstance(model_param, DTensor):
                local_tensor = model_param.to_local()
            else:
                local_tensor = model_param.data

            # Create CPU copy as nn.Parameter (leaf tensor with requires_grad=True)
            # Use pinned memory for faster CPU<->GPU transfers
            # Cast to float32 for better optimizer precision (gradients are bf16, but
            # accumulated optimizer state benefits from fp32)
            cpu_data = local_tensor.detach().cpu().to(torch.float32).pin_memory()
            cpu_param = nn.Parameter(cpu_data, requires_grad=True)
            cpu_params.append(cpu_param)

            self._gpu_to_cpu_param[model_param] = cpu_param
            self._cpu_to_gpu_param[cpu_param] = model_param

        logger.info(f"Created {len(cpu_params)} CPU shadow parameters (fp32 for optimizer precision)")

        # Determine if using parameter groups with differential LR or weight decay
        using_param_groups = lr_multipliers is not None or weight_decay_multipliers is not None

        if using_param_groups:
            # Classify parameters into groups with different LRs and weight decay
            base_lr = optimizer_kwargs.get('lr', 1e-5)
            base_weight_decay = optimizer_kwargs.get('weight_decay', 0.0)

            # Use default LRMultipliers if not provided (all 1.0)
            effective_lr_multipliers = lr_multipliers if lr_multipliers is not None else LRMultipliers()

            param_groups = classify_parameters_for_groups(
                model_parts,
                base_lr,
                effective_lr_multipliers,
                base_weight_decay,
                weight_decay_multipliers,
            )

            # Build CPU param_groups from model param_groups
            cpu_param_groups = []
            for group in param_groups:
                cpu_group_params = []
                for model_param in group['params']:
                    # Find corresponding CPU param
                    if model_param in self._gpu_to_cpu_param:
                        cpu_group_params.append(self._gpu_to_cpu_param[model_param])

                cpu_param_groups.append({
                    'params': cpu_group_params,
                    'lr': group['lr'],
                    'weight_decay': group['weight_decay'],
                    'group_name': group.get('group_name', 'unknown'),
                })

            features = []
            if lr_multipliers is not None:
                features.append("differential learning rates")
            if weight_decay_multipliers is not None:
                features.append("differential weight decay")
            logger.info(
                f"CPUOffloadOptimizer: Using {len(cpu_param_groups)} parameter groups "
                f"with {' and '.join(features)}"
            )
        else:
            # Original behavior: single param list
            cpu_param_groups = cpu_params
            logger.info(
                f"CPUOffloadOptimizer: Using single parameter group with uniform LR"
            )

        # Create CPU optimizer using DeepSpeed's optimized CPUAdam
        # Remove fused/foreach flags as they don't apply to CPU optimizers
        cpu_kwargs = {
            k: v for k, v in optimizer_kwargs.items()
            if k not in ("fused", "foreach")
        }
        cpu_kwargs["adamw_mode"] = adamw_mode

        # Require DeepSpeed for CPU offload
        if not HAS_DEEPSPEED_CPU_ADAM:
            raise RuntimeError(
                "CPU offload optimizer requires DeepSpeed. "
                "Install with: pip install deepspeed"
            )

        cpu_optimizer = DeepSpeedCPUAdam(cpu_param_groups, **cpu_kwargs)
        logger.info(
            f"CPUOffloadOptimizer: Using DeepSpeedCPUAdam "
            f"with {len(all_params)} parameters"
        )

        self.optimizers = [cpu_optimizer]

        self._validate_length(1)  # Single optimizer for all params
        self._post_init(all_params, optimizer_kwargs)

    @torch.no_grad()
    def step(self, *args, **kwargs) -> None:
        """Execute optimizer step on CPU and sync back to GPU.

        Warning:
            For DTensor gradients, redistribute() is a collective operation.
            All ranks MUST have gradients for the same parameters, or the
            collective will hang. This is guaranteed after a proper backward
            pass, but can fail if:
            - Parameters are frozen inconsistently across ranks
            - Custom backward hooks skip certain gradients on some ranks
            - Model parallelism configurations differ between ranks
        """
        # Copy gradients from GPU to CPU (with dtype conversion bf16→fp32)
        skipped_params = 0
        for model_param, cpu_param in self._gpu_to_cpu_param.items():
            if model_param.grad is None:
                # Skip parameters without gradients (e.g., frozen params).
                # IMPORTANT: This is safe ONLY if the parameter is frozen on ALL ranks.
                # If grad is None on this rank but exists on other ranks, the
                # redistribute() collective below will hang indefinitely.
                skipped_params += 1
                continue

            if isinstance(model_param.grad, DTensor):
                # For TP, gradients may have Partial placement (need reduction).
                # Redistribute to match parameter's placement before getting local tensor.
                # This handles both FSDP (Shard) and TP (Partial -> Shard) cases.
                if isinstance(model_param, DTensor):
                    grad_redistributed = model_param.grad.redistribute(
                        model_param.device_mesh, model_param.placements
                    )
                    grad_local = grad_redistributed.to_local()
                else:
                    grad_local = model_param.grad.to_local()
            else:
                grad_local = model_param.grad

            # Non-blocking copy to overlap with other transfers
            # grad_local is bf16, cpu_param.grad must be fp32 for precision
            if cpu_param.grad is None:
                # Explicitly create fp32 pinned gradient buffer
                # Pinned memory avoids 2x allocation during non-blocking transfer
                cpu_param.grad = torch.empty_like(cpu_param.data, dtype=torch.float32).pin_memory()
            elif cpu_param.grad.shape != grad_local.shape:
                # Shape mismatch - reallocate buffer (should be rare, only with dynamic shapes)
                logger.warning(
                    f"Gradient shape changed: {cpu_param.grad.shape} -> {grad_local.shape}. "
                    "Reallocating CPU gradient buffer."
                )
                cpu_param.grad = torch.empty(
                    grad_local.shape, dtype=torch.float32, device="cpu"
                ).pin_memory()
            # .copy_() will cast bf16 → fp32 automatically
            cpu_param.grad.copy_(grad_local, non_blocking=True)

        # Log skipped params at debug level (useful for debugging frozen param issues)
        if skipped_params > 0:
            logger.debug(
                f"CPUOffloadOptimizer: Skipped {skipped_params} params without gradients"
            )

        # Sync to ensure all gradient copies are complete before CPU optimizer runs
        # Use CUDA event for fine-grained sync (only waits on our copies, not all CUDA ops)
        if torch.cuda.is_available():
            if self._grad_copy_event is None:
                self._grad_copy_event = torch.cuda.Event()
            self._grad_copy_event.record()
            self._grad_copy_event.synchronize()

        # Run optimizer on CPU
        for optimizer in self.optimizers:
            optimizer.step(*args, **kwargs)

        # Copy updated parameters back to GPU (with dtype conversion fp32→bf16)
        for cpu_param, model_param in self._cpu_to_gpu_param.items():
            # cpu_param.data is fp32, model_param may be bf16 → automatic downcast
            if isinstance(model_param, DTensor):
                # NOTE: Using _local_tensor instead of to_local() because:
                # 1. to_local() may return a view that doesn't propagate in-place updates
                # 2. We need direct access to the underlying storage for copy_()
                # 3. _local_tensor is stable in PyTorch's DTensor implementation
                local_tensor = model_param._local_tensor  # noqa: SLF001
                local_tensor.copy_(cpu_param.data, non_blocking=True)
            else:
                model_param.data.copy_(cpu_param.data, non_blocking=True)

        # Sync to ensure parameter updates are visible before next forward pass
        # Use CUDA event for fine-grained sync
        if torch.cuda.is_available():
            if self._param_copy_event is None:
                self._param_copy_event = torch.cuda.Event()
            self._param_copy_event.record()
            self._param_copy_event.synchronize()

    def zero_grad(self, *args, **kwargs) -> None:
        """Zero gradients on both GPU and CPU."""
        # Zero GPU gradients (standard behavior)
        for model in self.model_parts:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()

        # Zero CPU gradients
        for optimizer in self.optimizers:
            optimizer.zero_grad(*args, **kwargs)

    def state_dict(self) -> dict[str, Any]:
        """Get optimizer state dict.

        Note: CPU offload optimizer state is stored on CPU, so this returns
        the raw optimizer state without DCP resharding support.
        """
        # For CPU offload, we return the raw optimizer state
        # This may need enhancement for checkpoint compatibility
        return {"cpu_optimizer": self.optimizers[0].state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load optimizer state dict."""
        if "cpu_optimizer" in state_dict:
            self.optimizers[0].load_state_dict(state_dict["cpu_optimizer"])
        else:
            logger.warning(
                "Loading state dict without 'cpu_optimizer' key - "
                "this may be from a non-CPU-offload checkpoint"
            )

    def sync_params_gpu_to_cpu(self) -> None:
        """Sync GPU parameters to CPU shadow copies.

        Call this AFTER loading model checkpoint to ensure CPU optimizer
        has the correct parameter values (not random init).
        Converts bf16 GPU params → fp32 CPU params for better optimizer precision.
        """
        with torch.no_grad():
            for model_param, cpu_param in self._gpu_to_cpu_param.items():
                if isinstance(model_param, DTensor):
                    # See NOTE in step() for why we use _local_tensor
                    local_tensor = model_param._local_tensor  # noqa: SLF001
                else:
                    local_tensor = model_param.data
                # model param is bf16, cpu param is fp32 → automatic upcast
                cpu_param.data.copy_(local_tensor)
        logger.info("Synced GPU parameters to CPU optimizer shadow copies (bf16→fp32)")

    def cleanup(self) -> None:
        """Release CPU shadow parameters and pinned memory resources.

        Call this when the optimizer is no longer needed to free pinned memory,
        which is a limited system resource. This method:
        - Clears CPU gradient buffers (pinned memory)
        - Clears CPU shadow parameters (pinned memory)
        - Releases CUDA events
        - Clears the optimizer state

        After calling cleanup(), this optimizer instance should not be used.
        """
        # Clear gradient buffers first (pinned memory)
        for cpu_param in self._gpu_to_cpu_param.values():
            if cpu_param.grad is not None:
                cpu_param.grad = None

        # Clear optimizer state (releases references to CPU params)
        for optimizer in self.optimizers:
            optimizer.state.clear()

        # Clear the parameter mappings (releases CPU shadow params)
        self._gpu_to_cpu_param.clear()
        self._cpu_to_gpu_param.clear()

        # Release CUDA events
        self._grad_copy_event = None
        self._param_copy_event = None

        logger.info("CPUOffloadOptimizer: Released pinned memory and cleaned up resources")


class FTOptimizersContainer(OptimizersContainer):
    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: dict[str, Any],
        ft_manager: "ft.Manager",
        use_ft_optimizer: bool = True,
    ) -> None:
        super().__init__(model_parts, optimizer_cls, optimizer_kwargs)

        # Force to initialize the optimizer state so that `optim.step()`
        # won't be called by state_dict() and load_state_dict().
        _ = {
            k: v
            for sd in map(get_optimizer_state_dict, model_parts, self.optimizers)
            for k, v in sd.items()
        }
        self.cache_state_dict: dict[str, Any] = {}
        self._ft_optimizer = ft.Optimizer(ft_manager, self)
        # Whether to determine quorum using FT.optimizer,
        # in semi-sync training we use the synchronization step to start quorum
        self._use_ft_optimizer: bool = use_ft_optimizer

    def init_cache_state_dict(self) -> None:
        self.cache_state_dict = super().state_dict()

    def state_dict(self) -> dict[str, Any]:
        return self.cache_state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # We have to invalidate the `cache_state_dict` because optimizer uses
        # assign instead of copy when doing `load_state_dict()`. Without
        # invalidating the `cache_state_dict`, there will be memory leakage.
        self.cache_state_dict = {}
        super().load_state_dict(state_dict)
        self.init_cache_state_dict()

    def step(self, *args, **kwargs) -> None:
        """Calling the correct step() depending on the caller.

        TorchFT's OptimizerWrapper.step() is designed to be called only once
        per train step per ft.Manager regardless how many optimizers are used.
        Hence we will need to appropriately dispatch the call.
        """
        if self._use_ft_optimizer:
            self._use_ft_optimizer = False
            self._ft_optimizer.step(*args, **kwargs)
            self._use_ft_optimizer = True
        else:
            super().step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        """Calling the correct zero_grad() depending on the caller.

        Check the comment in ``step()``.
        """
        if self._use_ft_optimizer:
            self._use_ft_optimizer = False
            self._ft_optimizer.zero_grad(*args, **kwargs)
            self._use_ft_optimizer = True
        else:
            super().zero_grad(*args, **kwargs)


def build_optimizers(
    model_parts: list[nn.Module],
    optimizer_config: OptimizerConfig,
    parallel_dims: ParallelDims,
    ft_manager: FTManager | None = None,
) -> OptimizersContainer:
    """Create a OptimizersContainer for the given model parts and job config.

    This function creates a ``OptimizersContainer`` for the given model parts.
    ``optimizer_config`` should define the correct optimizer name and parameters.
    This function currently supports creating ``OptimizersContainer``,
    ``OptimizersInBackwardContainer``, and ``CPUOffloadOptimizersContainer``.

    **Note**
    Users who want to customize the optimizer behavior can create their own
    ``OptimizersContainer`` subclass and ``build_optimizers``. Passing the
    customized ``build_optimizers`` to ``TrainSpec`` will create the customized
    ``OptimizersContainer``.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizer_config (OptimizerConfig): Optimizer config containing the optimizer name and parameters.
        parallel_dims (ParallelDims): Parallel dimensions for the model.
    """
    optim_in_bwd = optimizer_config.early_step_in_backward
    cpu_offload = getattr(optimizer_config, "cpu_offload", False)

    if optim_in_bwd:
        if parallel_dims.ep_enabled:
            raise NotImplementedError(
                "Optimizers in backward is not supported with Expert Parallel."
            )
        if parallel_dims.pp_enabled:
            raise NotImplementedError(
                "Optimizers in backward is not supported with Pipeline Parallel."
            )
        if ft_manager and ft_manager.enabled:
            raise NotImplementedError(
                "TorchFT is not supported with optimizers in backward."
            )
        if cpu_offload:
            raise NotImplementedError(
                "CPU offload is not supported with optimizers in backward."
            )

    if cpu_offload:
        if ft_manager and ft_manager.enabled:
            raise NotImplementedError(
                "TorchFT is not supported with CPU offload optimizer."
            )

    name = optimizer_config.name
    lr = optimizer_config.lr
    beta1 = optimizer_config.beta1
    beta2 = optimizer_config.beta2
    eps = optimizer_config.eps
    weight_decay = optimizer_config.weight_decay

    optim_implementation = optimizer_config.implementation
    assert optim_implementation in ["fused", "foreach", "for-loop"]

    fused = optim_implementation == "fused"
    foreach = optim_implementation == "foreach"

    optimizer_kwargs = {
        "lr": lr,
        "betas": (beta1, beta2),
        "eps": eps,
        "weight_decay": weight_decay,
        "fused": fused,
        "foreach": foreach,
    }

    optimizer_classes = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
    }
    if name not in optimizer_classes:
        raise NotImplementedError(f"Optimizer {name} not added.")
    optimizer_cls = optimizer_classes[name]

    if optim_in_bwd:
        return OptimizersInBackwardContainer(
            model_parts, optimizer_cls, optimizer_kwargs
        )

    # CPU offload uses DeepSpeed's CPUAdam
    if cpu_offload:
        adamw_mode = (name == "AdamW")

        # Check if using differential learning rates or weight decay
        lr_multipliers = getattr(optimizer_config, 'lr_multipliers', None)
        weight_decay_multipliers = getattr(optimizer_config, 'weight_decay_multipliers', None)

        # Check and log differential LR
        if _has_non_default_multipliers(lr_multipliers, _LR_MULTIPLIER_FIELDS):
            _log_non_default_multipliers(lr_multipliers, _LR_MULTIPLIER_FIELDS, "learning rates")
        else:
            lr_multipliers = None  # All multipliers are 1.0, disable feature

        # Check and log differential weight decay
        if _has_non_default_multipliers(weight_decay_multipliers, _WD_MULTIPLIER_FIELDS):
            _log_non_default_multipliers(weight_decay_multipliers, _WD_MULTIPLIER_FIELDS, "weight decay")
        else:
            weight_decay_multipliers = None  # All multipliers are 1.0, disable feature

        return CPUOffloadOptimizersContainer(
            model_parts,
            optimizer_kwargs,
            adamw_mode=adamw_mode,
            lr_multipliers=lr_multipliers,
            weight_decay_multipliers=weight_decay_multipliers,
        )

    if ft_manager and ft_manager.enabled:
        return FTOptimizersContainer(
            model_parts,
            optimizer_cls,
            optimizer_kwargs,
            ft_manager.manager,
            use_ft_optimizer=ft_manager.use_async_quorum,
        )

    # Check if using differential weight decay or LR for standard optimizer
    lr_multipliers = getattr(optimizer_config, 'lr_multipliers', None)
    weight_decay_multipliers = getattr(optimizer_config, 'weight_decay_multipliers', None)

    using_differential_lr = _has_non_default_multipliers(lr_multipliers, _LR_MULTIPLIER_FIELDS)
    using_differential_wd = _has_non_default_multipliers(weight_decay_multipliers, _WD_MULTIPLIER_FIELDS)

    if using_differential_wd or using_differential_lr:
        # Use parameter groups with differential settings
        effective_lr_multipliers = lr_multipliers if lr_multipliers is not None else LRMultipliers()
        effective_wd_multipliers = weight_decay_multipliers if using_differential_wd else None

        return OptimizersContainerWithParamGroups(
            model_parts,
            optimizer_cls,
            optimizer_kwargs,
            effective_lr_multipliers,
            effective_wd_multipliers,
        )

    return OptimizersContainer(model_parts, optimizer_cls, optimizer_kwargs)


def build_optimizers_with_moe_load_balancing(
    model_parts: list[nn.Module],
    optimizer_config: OptimizerConfig,
    parallel_dims: ParallelDims,
    ft_manager: FTManager | None = None,
) -> OptimizersContainer:
    optimizers = build_optimizers(
        model_parts=model_parts,
        optimizer_config=optimizer_config,
        parallel_dims=parallel_dims,
        ft_manager=ft_manager,
    )

    def _should_register_moe_balancing_hook(model_parts: list[nn.Module]) -> bool:
        for model_part in model_parts:
            # pyrefly: ignore [not-callable]
            for transformer_block in model_part.layers.values():
                # pyrefly: ignore [missing-attribute]
                if transformer_block.moe_enabled:
                    # Assumption: load_balance_coeff is set universally on all moe blocks.
                    # pyrefly: ignore [missing-attribute]
                    return bool(transformer_block.moe.load_balance_coeff)
        return False

    # for MoE auxiliary-loss-free load balancing
    def _is_recomputation_enabled(module):
        return getattr(module, "checkpoint_impl", None) is CheckpointImpl.NO_REENTRANT

    def _update_expert_bias(
        model_parts: list[nn.Module],
        parallel_dims: ParallelDims,
    ):
        loss_mesh = parallel_dims.get_optional_mesh("loss")
        # TODO: Currently this sync is blocking (thus exposed) and happens on the
        # default compute stream. Need to assess if this is OK performance-wise.
        tokens_per_expert_list = []
        for model_part in model_parts:
            # pyrefly: ignore [not-callable]
            for transformer_block in model_part.layers.values():
                # pyrefly: ignore [missing-attribute]
                if not transformer_block.moe_enabled:
                    continue
                # pyrefly: ignore [missing-attribute]
                if transformer_block.moe.load_balance_coeff is None:
                    return
                # pyrefly: ignore [missing-attribute]
                tokens_per_expert = transformer_block.moe.tokens_per_expert
                if _is_recomputation_enabled(transformer_block):
                    # TODO: This is a hack, we assume with full AC, the tokens_per_expert is counted twice.
                    # This does not affect to expert choice, but affects the experts usage metrics.
                    # We divide by 2 to correct for this double-counting due to recomputation
                    # TODO: new API to help determine if AC is enabled https://github.com/pytorch/pytorch/pull/160888
                    tokens_per_expert = tokens_per_expert // 2
                tokens_per_expert_list.append(tokens_per_expert)

        tokens_per_expert_by_layer = torch.vstack(tokens_per_expert_list)

        if loss_mesh is not None:
            if isinstance(tokens_per_expert_by_layer, torch.distributed.tensor.DTensor):
                tokens_per_expert_by_layer = tokens_per_expert_by_layer.redistribute(
                    placements=[Replicate()]
                    * tokens_per_expert_by_layer.device_mesh.ndim
                )
            else:
                # Perform single all-reduce to get global statistics across all processes
                pg = loss_mesh.get_group()
                torch.distributed.all_reduce(
                    tokens_per_expert_by_layer,
                    group=pg,
                    op=torch.distributed.ReduceOp.SUM,
                )

        moe_layer_idx = 0
        with torch.no_grad():
            for model_part in model_parts:
                # pyrefly: ignore [not-callable]
                for transformer_block in model_part.layers.values():
                    # pyrefly: ignore [missing-attribute]
                    if not transformer_block.moe_enabled:
                        continue
                    # pyrefly: ignore [missing-attribute]
                    moe = transformer_block.moe

                    tokens_per_expert = tokens_per_expert_by_layer[
                        moe_layer_idx
                    ].float()
                    moe_layer_idx += 1

                    # update the expert bias
                    # this is not exactly the same as https://arxiv.org/pdf/2408.15664 proposed
                    expert_bias_delta = moe.load_balance_coeff * torch.sign(
                        tokens_per_expert.mean() - tokens_per_expert
                    )
                    expert_bias_delta = expert_bias_delta - expert_bias_delta.mean()
                    moe.expert_bias.add_(expert_bias_delta)
                    moe.tokens_per_expert.zero_()

    if _should_register_moe_balancing_hook(model_parts):
        optimizers.register_step_pre_hook(
            lambda *args, **kwargs: _update_expert_bias(
                model_parts, parallel_dims=parallel_dims
            )
        )

    return optimizers
