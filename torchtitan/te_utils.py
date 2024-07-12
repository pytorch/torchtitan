# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for testing TransformerEngine

Note: I attempted to hack in DTensor-based TP/SP to te.Linear in the 
link below, and gave up for now as it seemed to be a lot of remaining work.
We can power through that if needed later.
* https://gist.github.com/vkuzo/64d5362b63dd6c76410464e020d9a35f

Note: I looked into using te.LayerNormLinear, and that would require changing
how Attention and FFN are defined in torchtitan to use a single gemm for
attn.kqv and ffn.w1_w3.  Punting for now but we can do this later if needed.

Note: PyTorch's checkpointing does not work with TE float8, fails with
* https://gist.github.com/vkuzo/54c76c16d6a38610a1d78f4de07a71e7
TE does have a `transformer_engine.pytorch.checkpoint` function, but 
unclear where the code for that lives. For now, we have to use 
`--activation_checkpoint.mode none`.

Note: using `--activation_checkpoint.mode none` leads to poor performance, the
`WARNING - 164 CUDA memory allocation retries` from the logs seems relevant.
Full logs: https://gist.github.com/vkuzo/0d6ebac2df3f7c90464da1e16d75d24c

After decreasing the number of layers in LLaMa 3 8B from 32 to 16, we can
performantly train with TE.
"""

import contextlib
import os

# required for current build to work with fp8 on devgpu003.cco3
# context: https://github.com/NVIDIA/TransformerEngine/pull/575
# error stack trace if not enabled: https://gist.github.com/vkuzo/8e78282f4a986961753fba25249fdf77
os.environ["NVTE_UNFUSED_FP8_UPDATE"] = "1"

import torch

# import transformer_engine as te
import transformer_engine.pytorch as te

from transformer_engine.common.recipe import Format, DelayedScaling
te_fp8_format = Format.HYBRID
te_fp8_recipe = DelayedScaling(fp8_format=te_fp8_format, amax_history_len=16, amax_compute_algo="max")

def swap_linear_to_te_linear(model, fqn=''):
    for name, child in model.named_children():
        new_fqn = f"{fqn}.{name}"
        if isinstance(child, torch.nn.Linear):
            te_linear = te.Linear(child.in_features, child.out_features, bias=child.bias is not None)
            te_linear.weight = child.weight
            te_linear.bias = child.bias
            setattr(model, name, te_linear)
        else:
            swap_linear_to_te_linear(child, new_fqn)

def get_maybe_fp8_autocast(job_config):
    # not for land - set up TransformerEngine fp8 autocast
    # Note: te.fp8_autocast has to be created at every training iteration.
    # If we try to create it once and reuse, we get this error:
    # https://gist.github.com/vkuzo/d9840328c8bdc2901b8d04aa570ecb5b
    maybe_te_float8_ctx = contextlib.nullcontext()
    if job_config.training.use_te and job_config.training.use_te_float8:
        maybe_te_float8_ctx = te.fp8_autocast(enabled=True, fp8_recipe=te_fp8_recipe)
    return maybe_te_float8_ctx
