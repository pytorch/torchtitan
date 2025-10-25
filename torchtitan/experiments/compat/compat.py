# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
PyTorch compatibility shims for non-nightly versions (Experimental).

This experimental module provides compatibility between PyTorch nightly and stable releases
by shimming missing modules and functions. Import this module early in your
application to enable automatic shimming.

The shims are automatically installed when torchtitan is imported.

Usage:
    import torchtitan.experiments.compat  # noqa: F401
"""

import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType


class CompatShimLoader(Loader):
    """Loader that provides shim modules for missing PyTorch features."""

    def __init__(self, module_name: str, shim_factory):
        self.module_name = module_name
        self.shim_factory = shim_factory

    def create_module(self, spec):
        """Create the shim module."""
        return self.shim_factory()

    def exec_module(self, module):
        """Module is already populated by create_module."""
        pass


class CompatMetaPathFinder(MetaPathFinder):
    """Meta path finder that intercepts imports and provides compatibility shims."""

    # Registry of shims: module_name -> factory function
    SHIMS = {}

    def find_spec(self, fullname, path, target=None):
        """Find module spec for shimmed modules."""
        if fullname in self.SHIMS:
            return ModuleSpec(
                fullname,
                CompatShimLoader(fullname, self.SHIMS[fullname]),
                origin="torchtitan-compat-shim",
            )
        return None


def register_shim(module_name: str, factory):
    """Register a shim factory for a module.

    Args:
        module_name: Full module name to shim (e.g., 'torch.foo.bar')
        factory: Callable that returns a module object with the shimmed functionality
    """
    CompatMetaPathFinder.SHIMS[module_name] = factory


# ============================================================================
# Shim Definitions
# ============================================================================


def _shim_checkpoint_staging():
    """Shim for torch.distributed.checkpoint.staging missing classes"""
    from torch.distributed.checkpoint import staging

    # Create wrapper for StagingOptions
    class StagingOptions:
        """Shim for StagingOptions from PyTorch nightly."""

        __slots__ = ("args", "kwargs")

        def __init__(self, *args, **kwargs):
            # Store the arguments for potential future use
            self.args = args
            self.kwargs = kwargs

    # Create wrapper for DefaultStager
    class DefaultStager:
        """Shim for DefaultStager from PyTorch nightly."""

        def __init__(self, options=None):
            # In PyTorch 2.8, we can use BlockingAsyncStager as a fallback
            if hasattr(staging, "BlockingAsyncStager"):
                self._stager = staging.BlockingAsyncStager()
            else:
                self._stager = None
            self.options = options

        def __getattr__(self, name):
            # Delegate to the underlying stager if it exists
            if self._stager:
                return getattr(self._stager, name)
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        def close(self):
            """Close the stager."""
            if self._stager and hasattr(self._stager, "close"):
                self._stager.close()

    # Add the classes to the staging module
    staging.StagingOptions = StagingOptions
    staging.DefaultStager = DefaultStager

    return staging


def _shim_pipelining_schedules():
    """Shim for torch.distributed.pipelining.schedules missing classes"""
    from torch.distributed.pipelining import schedules

    # ScheduleDualPipeV is a nightly-only schedule class
    # For compatibility, we create a placeholder that raises an error if used
    # but allows the import to succeed
    class ScheduleDualPipeV:
        """Shim for ScheduleDualPipeV from PyTorch nightly.

        This is a placeholder to allow imports to succeed. If this schedule is
        actually used at runtime, it will raise an error.
        """

        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "ScheduleDualPipeV requires PyTorch nightly. "
                "This schedule is not available in PyTorch 2.8.0. "
                "Please use a different pipeline schedule or upgrade to PyTorch nightly."
            )

    # Add the class to the schedules module
    schedules.ScheduleDualPipeV = ScheduleDualPipeV

    return schedules


def _shim_flex_attention():
    """Shim for torch.nn.attention.flex_attention missing classes"""
    from typing import NamedTuple

    import torch
    from torch.nn.attention import flex_attention

    # AuxOutput is used for auxiliary outputs from flex_attention
    # It's a NamedTuple that contains logsumexp and per_sample_seed
    class AuxOutput(NamedTuple):
        """Shim for AuxOutput from PyTorch nightly.

        This is a simple NamedTuple to match the structure expected by flex_attention.
        """

        logsumexp: torch.Tensor
        per_sample_seed: torch.Tensor | None = None

    # Add the class to the flex_attention module
    flex_attention.AuxOutput = AuxOutput

    return flex_attention


def _shim_checkpoint_wrapper():
    """Shim for torch.distributed.algorithms._checkpoint.checkpoint_wrapper early_stop parameter"""
    import functools

    from torch.distributed.algorithms._checkpoint import (
        checkpoint_wrapper as checkpoint_module,
    )

    # Save the original checkpoint_wrapper
    _original_checkpoint_wrapper = checkpoint_module.checkpoint_wrapper

    @functools.wraps(_original_checkpoint_wrapper)
    def checkpoint_wrapper_shim(module, *args, early_stop=None, **kwargs):
        """Wrapper that filters out the early_stop parameter not supported in PyTorch 2.8.0.

        The early_stop parameter is a nightly-only feature for activation checkpointing.
        In PyTorch 2.8.0, we simply ignore it.
        """
        # Filter out early_stop parameter and pass everything else through
        return _original_checkpoint_wrapper(module, *args, **kwargs)

    # Replace the checkpoint_wrapper function with our shim
    checkpoint_module.checkpoint_wrapper = checkpoint_wrapper_shim

    return checkpoint_module


def _shim_consolidate_hf_safetensors():
    """Shim for torch.distributed.checkpoint._consolidate_hf_safetensors"""
    module = ModuleType("torch.distributed.checkpoint._consolidate_hf_safetensors")

    def consolidate_safetensor_files(checkpoint_id, save_path, *args, **kwargs):
        """Stub implementation that raises a helpful error."""
        raise NotImplementedError(
            "consolidate_safetensor_files requires PyTorch nightly. "
            "This feature is not available in PyTorch 2.8.0. "
            "Please either upgrade to PyTorch nightly or disable this feature."
        )

    def consolidate_safetensors_files_on_every_rank(
        input_dir, output_dir, fqn_to_index_mapping, num_threads=5, *args, **kwargs
    ):
        """Stub implementation that raises a helpful error."""
        raise NotImplementedError(
            "consolidate_safetensors_files_on_every_rank requires PyTorch nightly. "
            "This feature is not available in PyTorch 2.8.0. "
            "Please either upgrade to PyTorch nightly or disable HuggingFace checkpoint export."
        )

    module.consolidate_safetensor_files = consolidate_safetensor_files
    module.consolidate_safetensors_files_on_every_rank = (
        consolidate_safetensors_files_on_every_rank
    )

    return module


# ============================================================================
# Registration
# ============================================================================


def install_shims():
    """Install all compatibility shims."""
    # Register shims for missing modules
    register_shim(
        "torch.distributed.checkpoint._consolidate_hf_safetensors",
        _shim_consolidate_hf_safetensors,
    )

    # Install shims for existing modules with missing classes/parameters
    _shim_checkpoint_staging()
    _shim_pipelining_schedules()
    _shim_flex_attention()
    _shim_checkpoint_wrapper()

    # Install the meta path finder (only once)
    finder = CompatMetaPathFinder()
    if finder not in sys.meta_path:
        # Insert at the beginning so we can intercept before the import fails
        sys.meta_path.insert(0, finder)


# Auto-install shims when this module is imported
install_shims()
