from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parent
BUILD_DIR = ROOT / ".torch_extensions"
BUILD_DIR.mkdir(exist_ok=True)

_ROPE_MODULE = None


def _load_rope_module():
    global _ROPE_MODULE
    if _ROPE_MODULE is not None:
        return _ROPE_MODULE

    verbose = os.environ.get("LOCAL_RANK", "0") == "0"
    _ROPE_MODULE = load(
        name="llm_trainer_rope",
        sources=[
            str(ROOT / "rope_binding.cpp"),
            str(ROOT / "rope_kernel.cu"),
        ],
        build_directory=str(BUILD_DIR),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=verbose,
    )
    return _ROPE_MODULE


def rope_forward_pair(q, k, freqs):
    return _load_rope_module().rope_forward(q, k, freqs)


def rope_backward_pair(grad_k, grad_q, freqs):
    freqs_conj = torch.ops.aten.clone.default(torch.ops.aten._conj.default(freqs))
    return _load_rope_module().rope_forward(grad_k, grad_q, freqs_conj)
