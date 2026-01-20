# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Activation Checkpointing with CPU Offloading Support

This module extends torchtitan's activation checkpointing with CPU offloading capability,
inspired by DeepSpeed's CPU checkpointing implementation.

CPU offloading moves activation tensors to CPU RAM during the forward pass and brings them
back to GPU during the backward pass, trading memory for PCIe bandwidth.
"""

from collections import defaultdict

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

from torchtitan.config.job_config import ActivationCheckpoint as ACConfig
from torchtitan.tools.logging import logger


def _cpu_offload_context_fn():
    """
    Create a context function for CPU offloading of activation checkpoints.

    This function returns a tuple of contexts that uses saved_tensors_hooks to automatically
    offload tensors to CPU when they're saved during forward pass and reload them
    to GPU during backward pass.

    Returns:
        A tuple of (forward_context, recompute_context)
    """

    def pack_to_cpu(tensor):
        """Move tensor to CPU during forward pass"""
        if not isinstance(tensor, torch.Tensor):
            return tensor
        # Only offload CUDA tensors that are floating point and large enough
        if tensor.is_cuda and tensor.is_floating_point() and tensor.numel() > 0:
            # Use non-blocking transfer for better performance
            return tensor.to("cpu", non_blocking=True)
        return tensor

    def unpack_from_cpu(tensor):
        """Move tensor back to GPU during backward pass"""
        if not isinstance(tensor, torch.Tensor):
            return tensor
        # If tensor is on CPU, move it back to the current CUDA device
        if tensor.device.type == "cpu":
            return tensor.to(torch.cuda.current_device(), non_blocking=True)
        return tensor

    # Return the same context for both forward and recompute phases
    ctx = torch.autograd.graph.saved_tensors_hooks(pack_to_cpu, unpack_from_cpu)
    return (ctx, ctx)


def _cpu_offload_selective_context_fn(ac_config: ACConfig, mm_recompute_shapes: set):
    """
    Create a selective checkpoint context with CPU offloading support.

    This combines selective activation checkpointing (choosing which ops to save vs recompute)
    with CPU offloading (moving saved tensors to CPU).

    Args:
        ac_config: Activation checkpoint configuration
        mm_recompute_shapes: Set of matrix multiplication shapes to force recompute

    Returns:
        A context function for selective checkpointing with CPU offloading
    """
    # Get the default op save list for selective AC
    op_sac_save_list = {
        torch.ops.aten.mm.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten.linear.default,
    }

    def _get_custom_policy(meta):
        def _custom_policy(ctx, func, *args, **kwargs):
            # Always save CPU offload ops
            if (
                func == torch.ops.aten._to_copy.default
                and "cuda" in str(args[0].device)
                and "device" in kwargs
                and str(kwargs["device"]) == "cpu"
            ):
                return CheckpointPolicy.MUST_SAVE

            mode = "recompute" if ctx.is_recompute else "forward"
            mm_count_key = f"{mode}_mm_count"
            if func == torch.ops.aten.mm.default:
                if args[1].shape in mm_recompute_shapes:
                    return CheckpointPolicy.PREFER_RECOMPUTE
                meta[mm_count_key] += 1

            # Saves output of all compute ops, except every second mm
            to_save = func in op_sac_save_list and not (
                func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0
            )
            return (
                CheckpointPolicy.MUST_SAVE
                if to_save
                else CheckpointPolicy.PREFER_RECOMPUTE
            )

        return _custom_policy

    def selective_checkpointing_with_cpu_offload():
        """Combined context for selective AC + CPU offload"""
        meta = defaultdict(int)
        (
            selective_forward_ctx,
            selective_recompute_ctx,
        ) = create_selective_checkpoint_contexts(_get_custom_policy(meta))
        cpu_offload_forward_ctx, cpu_offload_recompute_ctx = _cpu_offload_context_fn()

        # Stack both contexts for forward phase
        class CombinedForwardContext:
            def __enter__(self):
                self.selective = selective_forward_ctx.__enter__()
                self.cpu_offload = cpu_offload_forward_ctx.__enter__()
                return self

            def __exit__(self, *args):
                self.cpu_offload.__exit__(*args)
                self.selective.__exit__(*args)

        # Stack both contexts for recompute phase
        class CombinedRecomputeContext:
            def __enter__(self):
                self.selective = selective_recompute_ctx.__enter__()
                self.cpu_offload = cpu_offload_recompute_ctx.__enter__()
                return self

            def __exit__(self, *args):
                self.cpu_offload.__exit__(*args)
                self.selective.__exit__(*args)

        return (CombinedForwardContext(), CombinedRecomputeContext())

    return selective_checkpointing_with_cpu_offload


def apply_full_ac_with_cpu_offload(module: nn.Module, ac_config: ACConfig) -> nn.Module:
    """
    Apply full activation checkpointing with CPU offloading to the module.

    This will checkpoint all activations and offload them to CPU RAM.

    Args:
        module: The module to apply full AC with CPU offload to
        ac_config: The activation checkpointing config

    Returns:
        The wrapped module with full AC + CPU offload applied
    """
    logger.info("Applying full activation checkpointing with CPU offload")

    return ptd_checkpoint_wrapper(
        module,
        context_fn=_cpu_offload_context_fn,
        preserve_rng_state=ac_config.preserve_rng_state,
        determinism_check=ac_config.determinism_check,
        early_stop=ac_config.early_stop,
        debug=ac_config.debug,
    )


def apply_selective_ac_with_cpu_offload(
    module: nn.Module,
    ac_config: ACConfig,
    *,
    base_fqn: str | None = None,
) -> nn.Module:
    """
    Apply selective activation checkpointing with CPU offloading to the module.

    This selectively checkpoints certain operations while offloading saved tensors to CPU.

    Args:
        module: The module to apply selective AC with CPU offload to
        ac_config: The activation checkpointing config
        base_fqn: The base fully qualified name of the module

    Returns:
        The wrapped module with selective AC + CPU offload applied
    """
    logger.info("Applying selective activation checkpointing with CPU offload")

    # Collect mm shapes to force recompute if configured
    mm_recompute_shapes = set()
    if len(ac_config.per_op_sac_force_recompute_mm_shapes_by_fqns) > 0:
        for module_fqn, submod in module.named_modules():
            fqn = module_fqn
            if base_fqn is not None:
                fqn = f"{base_fqn}.{module_fqn}"
            if not any(
                filter_fqn in fqn
                for filter_fqn in ac_config.per_op_sac_force_recompute_mm_shapes_by_fqns
            ):
                continue
            if not isinstance(submod, nn.Linear):
                raise ValueError(
                    "per_op_sac_force_recompute_mm_shapes_by_fqns expected to match "
                    f"a nn.Linear, but got: {submod}"
                )
            out_f, in_f = submod.weight.shape
            mm_recompute_shapes.add((in_f, out_f))

    def context_fn_wrapper():
        return _cpu_offload_selective_context_fn(ac_config, mm_recompute_shapes)

    return ptd_checkpoint_wrapper(
        module,
        context_fn=context_fn_wrapper,
        preserve_rng_state=ac_config.preserve_rng_state,
        determinism_check=ac_config.determinism_check,
        early_stop=ac_config.early_stop,
        debug=ac_config.debug,
    )


class ActivationOffloadWrapper(nn.Module):
    """
    Wrapper that offloads layer activations to CPU without checkpointing/recomputation.

    This keeps all activations but moves them to CPU RAM to save GPU memory.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self._cpu_activations = []

    def forward(self, *args, **kwargs):
        # Move inputs to GPU if they were offloaded
        args = tuple(
            self._to_gpu(arg) if isinstance(arg, torch.Tensor) else arg for arg in args
        )
        kwargs = {
            k: self._to_gpu(v) if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }

        # Run forward pass
        output = self.module(*args, **kwargs)

        # Offload output to CPU during forward pass
        if isinstance(output, torch.Tensor):
            output_cpu = output.to("cpu", non_blocking=True)
            # Register hook to bring it back for backward
            output.register_hook(lambda grad: self._backward_hook(grad, output_cpu))
            return output_cpu
        elif isinstance(output, tuple):
            output_cpu = tuple(
                o.to("cpu", non_blocking=True) if isinstance(o, torch.Tensor) else o
                for o in output
            )
            # Register hooks for tensor outputs
            for i, (o, o_cpu) in enumerate(zip(output, output_cpu)):
                if isinstance(o, torch.Tensor):
                    o.register_hook(
                        lambda grad, oc=o_cpu: self._backward_hook(grad, oc)
                    )
            return output_cpu
        return output

    def _to_gpu(self, tensor):
        """Move tensor from CPU to GPU"""
        if tensor.device.type == "cpu":
            return tensor.to(torch.cuda.current_device(), non_blocking=True)
        return tensor

    def _backward_hook(self, grad, cpu_activation):
        """Called during backward to move activation back to GPU"""
        return cpu_activation.to(grad.device, non_blocking=True)

    def __getattr__(self, name):
        """Forward attribute access to the wrapped module"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def apply_offload_wrapper_only(module: nn.Module) -> nn.Module:
    """
    Apply activation offloading WITHOUT checkpointing.

    This wraps the module to offload all activations to CPU, keeping them in memory
    but freeing GPU RAM. No recomputation happens - activations are transferred
    back to GPU during backward pass.

    Args:
        module: The module to wrap

    Returns:
        The wrapped module with activation offloading
    """
    return ActivationOffloadWrapper(module)
