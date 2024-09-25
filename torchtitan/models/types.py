# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Llama 2 is licensed under the LLAMA 2 Community License,
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000

    max_seq_len: int = 2048
    # If `True`, then each transformer block init uses its layer ID, and if
    # `False`, each uses the total number of transformer blocks
    depth_init: bool = True
    norm_type: str = "rmsnorm"

    # Multi-module model args
    encoder_embed_dim: int = 1024
    patch_size: int = 1
    tile_size: int = 128
    max_num_tiles: int = 8
    clip_num_layers: int = 16
    learnable_head_num_layers: int = 4
    activation: Callable = (nn.GELU,)
    # in_channels (int): The number of image input channels.
    in_channels: int = 3
    # return_intermediates (Optional[List[int]]): The indices of hidden layers to return.
    # If provided, it will return the intermediate results of the transformer layers
    # before they go through a next layer. For example, ``return_intermediate=[0,3]``
    # will return the tokens before they go through the first and fourth layers.
    return_intermediates: Optional[List[int]] = None
    is_causal: bool = True
