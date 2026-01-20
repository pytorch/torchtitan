"""Memory defragmentation utilities for training"""
import logging
from typing import Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class MemoryDefragManager:
    """Manages memory defragmentation during training"""

    def __init__(
        self,
        enabled: bool = True,
        defrag_freq: int = 10,  # Defrag every N steps
        aggressive: bool = False,
    ):
        self.enabled = enabled
        self.defrag_freq = defrag_freq
        self.aggressive = aggressive
        self.step_count = 0

        if self.enabled:
            logger.info(
                f"MemoryDefragManager enabled: freq={defrag_freq}, aggressive={aggressive}"
            )

    def step(self, step_num: int):
        """Called after each training step"""
        if not self.enabled:
            return

        self.step_count += 1

        if self.step_count % self.defrag_freq == 0:
            self._defragment()

    def _defragment(self):
        """Perform memory defragmentation"""
        if not self.enabled:
            return

        device = torch.cuda.current_device()

        # Get memory stats before
        before_reserved = torch.cuda.memory_reserved(device)
        before_allocated = torch.cuda.memory_allocated(device)

        # Method 1: Empty cache (basic)
        torch.cuda.empty_cache()

        if self.aggressive:
            # Method 2: Synchronize and empty cache again
            torch.cuda.synchronize()
            if dist.is_initialized():
                dist.barrier()
            torch.cuda.empty_cache()

        # Get memory stats after
        after_reserved = torch.cuda.memory_reserved(device)
        after_allocated = torch.cuda.memory_allocated(device)

        freed_mb = (before_reserved - after_reserved) / (1024**2)

        if freed_mb > 0:
            logger.info(
                f"[Defrag] Freed {freed_mb:.2f} MB "
                f"(reserved: {before_reserved/(1024**3):.2f} GB → {after_reserved/(1024**3):.2f} GB, "
                f"allocated: {after_allocated/(1024**2):.2f} MB)"
            )


def setup_allocator_config(
    max_split_size_mb: Optional[int] = None,
    garbage_collection_threshold: Optional[float] = None,
    roundup_power2_divisions: Optional[int] = None,
):
    """Configure PyTorch CUDA allocator for reduced fragmentation"""
    import os

    config_parts = ["expandable_segments:True"]

    if max_split_size_mb is not None:
        config_parts.append(f"max_split_size_mb:{max_split_size_mb}")

    if garbage_collection_threshold is not None:
        config_parts.append(
            f"garbage_collection_threshold:{garbage_collection_threshold}"
        )

    if roundup_power2_divisions is not None:
        config_parts.append(f"roundup_power2_divisions:{roundup_power2_divisions}")

    config = ",".join(config_parts)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = config

    logger.info(f"Allocator config: {config}")

    return config
