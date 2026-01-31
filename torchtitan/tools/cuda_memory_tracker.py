"""Track CUDA memory directly from nvidia-smi and PyTorch"""
import logging
import subprocess
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)


class CUDAMemoryTracker:
    """Track memory from both PyTorch and CUDA/nvidia-smi"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.device = torch.cuda.current_device()
        self.device_name = torch.cuda.get_device_name(self.device)

        if self.enabled:
            logger.info(
                f"CUDAMemoryTracker enabled for device {self.device}: {self.device_name}"
            )

    def get_nvidia_smi_memory(self) -> Optional[Dict[str, int]]:
        """Get memory from nvidia-smi"""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.free,memory.total",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(self.device),
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )

            if result.returncode == 0:
                used, free, total = map(int, result.stdout.strip().split(","))
                return {"used_mb": used, "free_mb": free, "total_mb": total}
        except Exception as e:
            logger.warning(f"Failed to get nvidia-smi memory: {e}")

        return None

    def get_pytorch_memory(self) -> Dict[str, int]:
        """Get memory from PyTorch"""
        stats = torch.cuda.memory_stats(self.device)

        return {
            "reserved_bytes": torch.cuda.memory_reserved(self.device),
            "allocated_bytes": torch.cuda.memory_allocated(self.device),
            "active_bytes": stats.get("active_bytes.all.current", 0),
            "inactive_bytes": stats.get("inactive_split_bytes.all.current", 0),
            "peak_active_bytes": stats.get("active_bytes.all.peak", 0),
            "num_alloc_retries": stats.get("num_alloc_retries.all.current", 0),
            "num_ooms": stats.get("num_ooms.all.current", 0),
        }

    def get_cuda_device_memory(self) -> Dict[str, int]:
        """Get memory directly from CUDA device properties"""
        props = torch.cuda.get_device_properties(self.device)

        return {
            "total_memory": props.total_memory,
            "reserved_memory": torch.cuda.memory_reserved(self.device),
            "allocated_memory": torch.cuda.memory_allocated(self.device),
        }

    def measure_all(self, phase: str, step: int):
        """Comprehensive memory measurement"""
        if not self.enabled:
            return

        # PyTorch memory
        pytorch_mem = self.get_pytorch_memory()

        # CUDA device memory
        cuda_mem = self.get_cuda_device_memory()

        # nvidia-smi memory (if available)
        smi_mem = self.get_nvidia_smi_memory()

        # Calculate fragmentation
        reserved = pytorch_mem["reserved_bytes"]
        allocated = pytorch_mem["allocated_bytes"]
        active = pytorch_mem["active_bytes"]

        fragmentation = reserved - allocated
        frag_pct = (fragmentation / reserved * 100) if reserved > 0 else 0

        # Log PyTorch view
        logger.info(
            f"[PyTorch] Step {step:2d} | {phase:25s} | "
            f"Reserved: {reserved/1e9:6.2f} GB | "
            f"Allocated: {allocated/1e6:8.2f} MB | "
            f"Active: {active/1e6:8.2f} MB | "
            f"Frag: {frag_pct:5.1f}%"
        )

        # Log CUDA/nvidia-smi view
        if smi_mem:
            logger.info(
                f"[CUDA-SMI] Step {step:2d} | {phase:25s} | "
                f"Used: {smi_mem['used_mb']/1024:6.2f} GB | "
                f"Free: {smi_mem['free_mb']/1024:6.2f} GB | "
                f"Total: {smi_mem['total_mb']/1024:6.2f} GB"
            )

        # Log comparison
        if smi_mem:
            pytorch_used_gb = reserved / 1e9
            smi_used_gb = smi_mem["used_mb"] / 1024
            diff_gb = smi_used_gb - pytorch_used_gb

            logger.info(
                f"[Compare]  Step {step:2d} | {phase:25s} | "
                f"PyTorch reports: {pytorch_used_gb:6.2f} GB | "
                f"nvidia-smi reports: {smi_used_gb:6.2f} GB | "
                f"Diff: {diff_gb:+6.2f} GB"
            )
