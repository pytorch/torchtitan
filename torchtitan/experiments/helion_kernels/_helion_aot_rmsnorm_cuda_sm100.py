"""
Auto-generated heuristic for kernels in rmsnorm.py.
Backend: decision_tree

Provides:
- key_rms_norm_helion_fwd_2d(*args): Returns config index (cache key)
- autotune_rms_norm_helion_fwd_2d(*args): Returns config dict for the given arguments
"""

import torch


def key_rms_norm_helion_fwd_2d(*args) -> int:
    """Select config index for the given arguments."""
    _arg0_dim0 = (
        int(args[0].shape[0])
        if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 0
        else 0
    )
    _arg0_dim1 = (
        int(args[0].shape[1])
        if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 1
        else 0
    )
    if _arg0_dim1 <= 3072.0:
        if _arg0_dim0 <= 2048.0:
            if _arg0_dim1 <= 2047.0:
                if _arg0_dim1 <= 96.0:
                    if _arg0_dim1 <= 48.0:
                        return 0
                    else:
                        return 2
                else:
                    return 0
            else:
                if _arg0_dim1 <= 2048.0:
                    return 3
                else:
                    return 0
        else:
            return 2
    else:
        if _arg0_dim1 <= 6144.0:
            if _arg0_dim0 <= 2048.0:
                return 5
            else:
                if _arg0_dim1 <= 4096.0:
                    return 5
                else:
                    return 1
        else:
            if _arg0_dim1 <= 8192.0:
                return 1
            else:
                if _arg0_dim0 <= 2048.0:
                    return 4
                else:
                    return 6


def autotune_rms_norm_helion_fwd_2d(*args) -> dict:
    """Select the optimal config for the given arguments."""
    configs = [
        {
            "block_sizes": [2],
            "reduction_loops": [None],
            "range_unroll_factors": [0],
            "range_warp_specializes": [None],
            "range_num_stages": [0],
            "range_multi_buffers": [None],
            "range_flattens": [None],
            "load_eviction_policies": ["last", "first", "first", "", ""],
            "num_warps": 4,
            "num_stages": 6,
            "indexing": [
                "pointer",
                "pointer",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
                "pointer",
            ],
            "atomic_indexing": [],
            "pid_type": "flat",
        },
        {
            "block_sizes": [1],
            "reduction_loops": [None],
            "range_unroll_factors": [0],
            "range_warp_specializes": [None],
            "range_num_stages": [0],
            "range_multi_buffers": [None],
            "range_flattens": [None],
            "load_eviction_policies": ["first", "last", "last", "", "last"],
            "num_warps": 16,
            "num_stages": 4,
            "indexing": [
                "pointer",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "pointer",
                "pointer",
                "pointer",
            ],
            "atomic_indexing": [],
            "pid_type": "flat",
        },
        {
            "block_sizes": [4],
            "reduction_loops": [None],
            "range_unroll_factors": [0],
            "range_warp_specializes": [None],
            "range_num_stages": [0],
            "range_multi_buffers": [None],
            "range_flattens": [None],
            "load_eviction_policies": ["first", "last", "", "first", ""],
            "num_warps": 2,
            "num_stages": 3,
            "indexing": [
                "pointer",
                "tensor_descriptor",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
            ],
            "atomic_indexing": [],
            "pid_type": "flat",
        },
        {
            "block_sizes": [2],
            "reduction_loops": [2048],
            "range_unroll_factors": [0],
            "range_warp_specializes": [None],
            "range_num_stages": [0],
            "range_multi_buffers": [None],
            "range_flattens": [None],
            "load_eviction_policies": ["last", "first", "last", "first", "first"],
            "num_warps": 2,
            "num_stages": 1,
            "indexing": [
                "tensor_descriptor",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
            ],
            "atomic_indexing": [],
            "pid_type": "flat",
        },
        {
            "block_sizes": [1],
            "reduction_loops": [8192],
            "range_unroll_factors": [0],
            "range_warp_specializes": [None],
            "range_num_stages": [0],
            "range_multi_buffers": [None],
            "range_flattens": [None],
            "load_eviction_policies": ["", "first", "last", "", ""],
            "num_warps": 8,
            "num_stages": 3,
            "indexing": [
                "pointer",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "pointer",
                "tensor_descriptor",
                "pointer",
            ],
            "atomic_indexing": [],
            "pid_type": "flat",
        },
        {
            "block_sizes": [1],
            "reduction_loops": [None],
            "range_unroll_factors": [0],
            "range_warp_specializes": [None],
            "range_num_stages": [0],
            "range_multi_buffers": [None],
            "range_flattens": [None],
            "load_eviction_policies": ["", "last", "last", "first", ""],
            "num_warps": 8,
            "num_stages": 2,
            "indexing": [
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "tensor_descriptor",
            ],
            "atomic_indexing": [],
            "pid_type": "flat",
        },
        {
            "block_sizes": [1],
            "reduction_loops": [2048],
            "range_unroll_factors": [0],
            "range_warp_specializes": [None],
            "range_num_stages": [0],
            "range_multi_buffers": [None],
            "range_flattens": [None],
            "load_eviction_policies": ["", "first", "last", "first", "last"],
            "num_warps": 8,
            "num_stages": 8,
            "indexing": [
                "pointer",
                "pointer",
                "pointer",
                "tensor_descriptor",
                "pointer",
                "tensor_descriptor",
                "pointer",
            ],
            "atomic_indexing": [],
            "pid_type": "flat",
        },
    ]
    return configs[key_rms_norm_helion_fwd_2d(*args)]
