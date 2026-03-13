"""
Aggressive Memory Manager for reducing CUDA memory fragmentation.

This module provides aggressive memory clearing strategies to minimize
fragmentation and allocation retries during distributed training.

Usage:
    from torchtitan.tools.aggressive_memory_manager import AggressiveMemoryManager

    # Initialize at start of training
    mem_manager = AggressiveMemoryManager(
        clear_after_backward=True,
        clear_after_optimizer=True,
        sync_before_clear=True,
        defrag_threshold_mb=1000,  # Defrag if fragmentation > 1GB
    )

    # In training loop:
    loss.backward()
    mem_manager.post_backward()

    optimizer.step()
    mem_manager.post_optimizer()

    mem_manager.step_complete()
"""

import gc
import os
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

from torchtitan.tools.logging import logger


@dataclass
class MemoryStats:
    """Current memory statistics"""

    allocated: int
    reserved: int
    active: int
    fragmentation: int
    fragmentation_pct: float
    num_alloc_retries: int


class AggressiveMemoryManager:
    """
    Aggressive memory management to minimize CUDA memory fragmentation.

    Key strategies:
    1. Clear cache at strategic points (post-backward, post-optimizer)
    2. Synchronize before clearing to ensure all async ops complete
    3. Force garbage collection to release Python references
    4. Monitor fragmentation and trigger defrag when threshold exceeded
    5. Set optimal allocator configuration

    Args:
        clear_after_backward: Clear cache after backward pass
        clear_after_optimizer: Clear cache after optimizer step
        clear_every_n_steps: Only clear every N steps (1 = every step)
        sync_before_clear: Synchronize CUDA before clearing cache
        defrag_threshold_mb: Trigger defrag if fragmentation exceeds this (MB)
        gc_generation: Python GC generation to collect (0-2, higher = more thorough)
        verbose: Log detailed memory stats
        rank: Distributed rank (auto-detected if None)
    """

    def __init__(
        self,
        clear_after_backward: bool = True,
        clear_after_optimizer: bool = True,
        clear_every_n_steps: int = 1,
        sync_before_clear: bool = True,
        defrag_threshold_mb: float = 500.0,
        gc_generation: int = 1,
        verbose: bool = False,
        rank: Optional[int] = None,
    ):
        self.clear_after_backward = clear_after_backward
        self.clear_after_optimizer = clear_after_optimizer
        self.clear_every_n_steps = clear_every_n_steps
        self.sync_before_clear = sync_before_clear
        self.defrag_threshold_mb = defrag_threshold_mb
        self.gc_generation = gc_generation
        self.verbose = verbose

        self.rank = (
            rank
            if rank is not None
            else (dist.get_rank() if dist.is_initialized() else 0)
        )

        self.step_count = 0
        self.total_clears = 0
        self.total_defrag_time_ms = 0.0

        # Disable automatic GC - we'll control it manually
        gc.disable()

        # Initial cleanup
        self._aggressive_clear("initialization")

        if self.rank == 0:
            logger.info(
                f"[AggressiveMemoryManager] Initialized: "
                f"clear_backward={clear_after_backward}, "
                f"clear_optimizer={clear_after_optimizer}, "
                f"every_n_steps={clear_every_n_steps}, "
                f"sync={sync_before_clear}, "
                f"defrag_threshold={defrag_threshold_mb}MB"
            )

    @staticmethod
    def configure_allocator(
        expandable_segments: bool = True,
        max_split_size_mb: int = 128,
        garbage_collection_threshold: float = 0.8,
        roundup_power2_divisions: int = 4,
    ) -> str:
        """
        Configure PyTorch CUDA allocator for minimal fragmentation.

        Call this BEFORE any CUDA operations (before model creation).

        Args:
            expandable_segments: Enable expandable memory segments
            max_split_size_mb: Max size of memory splits (smaller = less fragmentation)
            garbage_collection_threshold: Trigger GC when this fraction of memory is fragmented
            roundup_power2_divisions: Memory rounding granularity

        Returns:
            The PYTORCH_CUDA_ALLOC_CONF string that was set
        """
        config_parts = []

        if expandable_segments:
            config_parts.append("expandable_segments:True")

        config_parts.append(f"max_split_size_mb:{max_split_size_mb}")
        config_parts.append(
            f"garbage_collection_threshold:{garbage_collection_threshold}"
        )
        config_parts.append(f"roundup_power2_divisions:{roundup_power2_divisions}")

        config_str = ",".join(config_parts)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = config_str

        return config_str

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        if not torch.cuda.is_available():
            return MemoryStats(0, 0, 0, 0, 0.0, 0)

        stats = torch.cuda.memory_stats()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        active = stats.get("active_bytes.all.current", 0)
        fragmentation = reserved - allocated
        fragmentation_pct = (fragmentation / reserved * 100) if reserved > 0 else 0.0
        num_retries = stats.get("num_alloc_retries", 0)

        return MemoryStats(
            allocated=allocated,
            reserved=reserved,
            active=active,
            fragmentation=fragmentation,
            fragmentation_pct=fragmentation_pct,
            num_alloc_retries=num_retries,
        )

    def _should_clear(self) -> bool:
        """Check if we should clear cache this step"""
        return self.step_count % self.clear_every_n_steps == 0

    def _aggressive_clear(self, reason: str) -> float:
        """
        Perform aggressive memory clearing.

        Returns:
            Time taken in milliseconds
        """
        if not torch.cuda.is_available():
            return 0.0

        start = time.perf_counter()

        # 1. Synchronize all CUDA streams to ensure ops complete
        if self.sync_before_clear:
            torch.cuda.synchronize()

        # 2. Python garbage collection (releases tensor references)
        gc.collect(self.gc_generation)

        # 3. Clear CUDA cache (releases unused cached memory)
        torch.cuda.empty_cache()

        # 4. Optional: Force synchronization after clear
        if self.sync_before_clear:
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start) * 1000
        self.total_clears += 1
        self.total_defrag_time_ms += elapsed_ms

        if self.verbose and self.rank == 0:
            stats = self.get_memory_stats()
            logger.info(
                f"[AggressiveMemoryManager] {reason}: "
                f"cleared in {elapsed_ms:.1f}ms, "
                f"frag={stats.fragmentation_pct:.1f}%, "
                f"reserved={stats.reserved/1e9:.2f}GB"
            )

        return elapsed_ms

    def _check_and_defrag(self, phase: str) -> bool:
        """
        Check fragmentation and defrag if needed.

        Returns:
            True if defrag was triggered
        """
        stats = self.get_memory_stats()
        fragmentation_mb = stats.fragmentation / (1024 * 1024)

        if fragmentation_mb > self.defrag_threshold_mb:
            self._aggressive_clear(f"defrag_{phase}_frag={fragmentation_mb:.0f}MB")
            return True

        return False

    def post_backward(self):
        """Call after backward pass completes"""
        if self.clear_after_backward and self._should_clear():
            self._check_and_defrag("post_backward")
            self._aggressive_clear("post_backward")

    def post_optimizer(self):
        """Call after optimizer step completes"""
        if self.clear_after_optimizer and self._should_clear():
            self._check_and_defrag("post_optimizer")
            self._aggressive_clear("post_optimizer")

    def step_complete(self):
        """Call at the end of each training step"""
        self.step_count += 1

        # Always check for high fragmentation
        self._check_and_defrag("step_end")

    def get_summary(self) -> str:
        """Get summary of memory management activity"""
        avg_time = self.total_defrag_time_ms / max(1, self.total_clears)
        return (
            f"AggressiveMemoryManager Summary:\n"
            f"  Total clears: {self.total_clears}\n"
            f"  Total defrag time: {self.total_defrag_time_ms:.1f}ms\n"
            f"  Avg time per clear: {avg_time:.2f}ms\n"
            f"  Steps processed: {self.step_count}"
        )


class BackwardMemoryHook:
    """
    Register hooks on model parameters to clear memory during backward pass.

    This clears memory incrementally as gradients are computed, rather than
    waiting until the end of backward.

    Args:
        clear_every_n_params: Clear cache after every N parameter gradients
        sync_on_clear: Synchronize before clearing (slower but more thorough)
    """

    def __init__(
        self,
        clear_every_n_params: int = 10,
        sync_on_clear: bool = False,
    ):
        self.clear_every_n_params = clear_every_n_params
        self.sync_on_clear = sync_on_clear
        self.param_count = 0
        self.handles = []

    def _backward_hook(self, grad):
        """Hook called when gradient is computed for a parameter"""
        self.param_count += 1

        if self.param_count % self.clear_every_n_params == 0:
            if self.sync_on_clear:
                torch.cuda.synchronize()
            gc.collect(0)  # Fast GC (generation 0 only)
            torch.cuda.empty_cache()

        return grad

    def register(self, model: torch.nn.Module):
        """Register hooks on all model parameters"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                handle = param.register_post_accumulate_grad_hook(
                    lambda p, name=name: self._backward_hook(p.grad)
                )
                self.handles.append(handle)

        logger.info(
            f"[BackwardMemoryHook] Registered on {len(self.handles)} parameters, "
            f"clearing every {self.clear_every_n_params} params"
        )

    def remove(self):
        """Remove all registered hooks"""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def reset_count(self):
        """Reset parameter count (call at start of each backward)"""
        self.param_count = 0


def setup_aggressive_memory_environment():
    """
    Set up environment variables for aggressive memory management.

    Call this BEFORE importing torch or creating any CUDA tensors.
    """
    # Optimal allocator settings for minimal fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:True,"
        "max_split_size_mb:128,"
        "garbage_collection_threshold:0.8,"
        "roundup_power2_divisions:4"
    )

    # Disable NCCL async error handling (can cause memory issues)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"

    # Force synchronous CUDA operations for debugging
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Uncomment for debugging

    return os.environ.get("PYTORCH_CUDA_ALLOC_CONF")


# Convenience function for quick setup
def create_aggressive_memory_manager(
    mode: str = "balanced",
    verbose: bool = False,
) -> AggressiveMemoryManager:
    """
    Create an AggressiveMemoryManager with preset configurations.

    Args:
        mode: One of:
            - "minimal": Only clear on high fragmentation
            - "balanced": Clear after backward and optimizer
            - "aggressive": Clear frequently with sync
            - "maximum": Clear after every operation
        verbose: Enable verbose logging

    Returns:
        Configured AggressiveMemoryManager
    """
    if mode == "minimal":
        return AggressiveMemoryManager(
            clear_after_backward=False,
            clear_after_optimizer=False,
            clear_every_n_steps=10,
            sync_before_clear=False,
            defrag_threshold_mb=2000,
            gc_generation=0,
            verbose=verbose,
        )
    elif mode == "balanced":
        return AggressiveMemoryManager(
            clear_after_backward=True,
            clear_after_optimizer=True,
            clear_every_n_steps=1,
            sync_before_clear=False,
            defrag_threshold_mb=500,
            gc_generation=1,
            verbose=verbose,
        )
    elif mode == "aggressive":
        return AggressiveMemoryManager(
            clear_after_backward=True,
            clear_after_optimizer=True,
            clear_every_n_steps=1,
            sync_before_clear=True,
            defrag_threshold_mb=200,
            gc_generation=2,
            verbose=verbose,
        )
    elif mode == "maximum":
        return AggressiveMemoryManager(
            clear_after_backward=True,
            clear_after_optimizer=True,
            clear_every_n_steps=1,
            sync_before_clear=True,
            defrag_threshold_mb=100,
            gc_generation=2,
            verbose=verbose,
        )
    else:
        raise ValueError(
            f"Unknown mode: {mode}. Use minimal/balanced/aggressive/maximum"
        )
