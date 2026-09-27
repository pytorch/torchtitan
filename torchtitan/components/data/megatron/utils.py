# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import logging
import os
import re
from enum import Enum

import numpy as np
import torch.distributed as dist
from torch.utils.cpp_extension import load

from torchtitan.kernels.utils import is_multi_storage_client_available

__all__ = [
    "MSC_PREFIX",
    "Split",
    "barrier",
    "compile_helpers",
    "get_global_rank",
    "get_local_rank",
    "is_distributed_initialized",
    "is_multi_storage_client_available",
    "log_rank_0",
]

logger = logging.getLogger(__name__)

# Paths with this prefix are read through the (optional) multi-storage client.
MSC_PREFIX = "msc://"


def is_distributed_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_global_rank() -> int:
    return dist.get_rank() if is_distributed_initialized() else 0


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def barrier() -> None:
    if is_distributed_initialized():
        dist.barrier()


def log_rank_0(level: int, msg: str) -> None:
    if get_global_rank() == 0:
        logger.log(level, msg, stacklevel=2)


class Split(Enum):
    train = 0
    valid = 1
    test = 2


_HELPERS = None


def compile_helpers() -> None:
    """JIT-compile ``helpers.cpp`` (index builders) into the package's ``build``
    directory. Global rank 0 compiles first; the other ranks then load the
    cached build, as in lm-engine's ``compile_cpp_extension(distributed=True)``."""
    global _HELPERS
    log_rank_0(logging.INFO, "compiling helpers.cpp")

    build_directory = os.path.join(os.path.dirname(__file__), "build")
    os.makedirs(build_directory, exist_ok=True)

    def _compile():
        # The name must match PYBIND11_MODULE(helpers, ...) in helpers.cpp.
        return load(
            "helpers",
            sources=[os.path.join(os.path.dirname(__file__), "helpers.cpp")],
            extra_cflags=["-O3", "-Wall", "-shared", "-fPIC", "-fdiagnostics-color"],
            build_directory=build_directory,
            verbose=get_global_rank() == 0,
        )

    if get_global_rank() == 0:
        _HELPERS = _compile()
    barrier()
    if get_global_rank() != 0:
        _HELPERS = _compile()


def build_blending_indices(
    dataset_index: np.ndarray,
    dataset_sample_index: np.ndarray,
    weights: list[float],
    num_datasets: int,
    size: int,
) -> None:
    _HELPERS.build_blending_indices(
        dataset_index, dataset_sample_index, weights, num_datasets, size
    )


def build_sample_idx(
    sizes: np.ndarray,
    doc_idx: np.ndarray,
    sequence_length: int,
    num_epochs: int,
    tokens_per_epoch: int,
) -> np.ndarray:
    if doc_idx.dtype == np.int32:
        log_rank_0(logging.INFO, "using int32 for sample idx")
        sample_idx = _HELPERS.build_sample_idx_int32(
            sizes, doc_idx, sequence_length, num_epochs, tokens_per_epoch
        )
    elif doc_idx.dtype == np.int64:
        log_rank_0(logging.INFO, "using int64 for sample idx")
        sample_idx = _HELPERS.build_sample_idx_int64(
            sizes, doc_idx, sequence_length, num_epochs, tokens_per_epoch
        )
    else:
        raise ValueError("unexpected dtype for doc_idx")

    return sample_idx


def normalize(weights: list[float]) -> list[float]:
    w = np.array(weights, dtype=np.float64)
    w_sum = np.sum(w)
    w = (w / w_sum).tolist()
    return w


def parse_and_normalize_split(split: str) -> list[float]:
    split = list(map(float, re.findall(r"[.0-9]+", split)))
    split = split + [0.0 for _ in range(len(Split) - len(split))]

    assert len(split) == len(Split)
    assert all(map(lambda _: _ >= 0.0, split))

    return normalize(split)
