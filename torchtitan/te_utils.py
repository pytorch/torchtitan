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
"""

import contextlib
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.models.llama.model import apply_rotary_emb, repeat_kv

# import transformer_engine as te
import transformer_engine.pytorch as te

from transformer_engine.common.recipe import Format, DelayedScaling
te_fp8_format = Format.HYBRID
te_fp8_recipe = DelayedScaling(fp8_format=te_fp8_format, amax_history_len=16, amax_compute_algo="max")

def swap_linear_to_te_linear(model, fqn=''):
    for name, child in model.named_children():
        new_fqn = f"{fqn}.{name}"
        # if isinstance(child, torch.nn.Linear) and new_fqn != 'output' and 'norm_' in new_fqn:
        if isinstance(child, torch.nn.Linear) and name != 'output':
            te_linear = te.Linear(child.in_features, child.out_features, bias=child.bias is not None)
            te_linear.weight = child.weight
            te_linear.bias = child.bias
            setattr(model, name, te_linear)
        else:
            swap_linear_to_te_linear(child, new_fqn)

class ResettableIdentity(nn.Identity):
    def reset_parameters(self):
        pass


class NormFeedForward(torch.nn.Module):
    """
    A replacement for ffn_norm -> ffn which is TE swap friendly
    """

    def __init__(self, ffn_norm, ffn):
        super().__init__()
        # self.ffn_norm = ffn_norm

        # fuse w1 and w3, TE assumes this optimization is applied
        w13_in_feat = ffn.w1.in_features
        w13_out_feat = ffn.w1.out_features * 2
        with torch.device("meta"):
            w13 = nn.Linear(w13_in_feat, w13_out_feat, bias=False)
        w13.weight = torch.nn.Parameter(
            torch.cat([ffn.w1.weight, ffn.w3.weight], dim=0).contiguous()
        )

        # wrapped in a sequential for easy swap to either te.LayerNorm or
        # torch.compiling just this wrapper
        self.norm_w13 = nn.Sequential(ffn_norm, w13)

        self.w2 = ffn.w2
        self.split_dim = getattr(self.norm_w13, "1").out_features // 2 

    def forward(self, x):
        # x = self.ffn_norm(x)
        # x = self.w13(x)
        x = self.norm_w13(x)
        w1_out, w3_out = torch.split(
            x, 
            self.split_dim, 
            dim=-1,
        )
        out = self.w2(F.silu(w1_out) * w3_out)
        return out

    def init_weights(self, init_std: float):
        if isinstance(self.norm_w13, te.LayerNormLinear):
            torch.nn.init.ones_(self.norm_w13.layer_norm_weight)

            # slight difference from llama/model.py - init every weight to init_std
            for linear in (self.w2, self.norm_w13):
                nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)

        else:
            getattr(self.norm_w13, "0").reset_parameters()
        
            # slight difference from llama/model.py - init every weight to init_std
            for linear in (self.w2, getattr(self.norm_w13, "1")):
                nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class NormAttention(torch.nn.Module):
    """
    A replacement for attn_norm -> attn which is TE swap friendly
    """
    def __init__(self, attn_norm, attn):
        super().__init__()

        # fuse attn.qkv, TE assumes this optimization is applied
        self.split_dim = attn.wq.out_features
        with torch.device("meta"):
            wqkv = nn.Linear(attn.wq.in_features, attn.wq.out_features * 3, bias=False)
        wqkv.weight = torch.nn.Parameter(
            torch.cat([attn.wq.weight, attn.wk.weight, attn.wv.weight], dim=0).contiguous()
        )

        self.norm_wqkv = nn.Sequential(attn_norm, wqkv)
        self.wo = attn.wo

        self.n_heads = attn.n_heads
        self.n_kv_heads = attn.n_kv_heads
        self.n_rep = attn.n_rep
        self.head_dim = attn.head_dim

    def forward(self, x, freqs_cis):
        bs, seqlen, _ = x.shape
        # xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        x = self.norm_wqkv(x)
        xq, xk, xv = torch.split(
            x, 
            [
                self.n_heads * self.head_dim,
                self.n_kv_heads * self.head_dim,
                self.n_kv_heads * self.head_dim,
            ],
            dim=-1,
        )

        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        # we use casual mask for training
        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        output = output.transpose(
            1, 2
        ).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
        output = output.view(bs, seqlen, -1)
        return self.wo(output)

    def init_weights(self, init_std: float):
        if isinstance(self.norm_wqkv, te.LayerNormLinear):
            torch.nn.init.ones_(self.norm_wqkv.layer_norm_weight)

            # slight difference from llama/model.py - init every weight to init_std
            for linear in (self.wo, self.norm_wqkv):
                nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)

        else:
            getattr(self.norm_wqkv, "0").reset_parameters()
        
            # slight difference from llama/model.py - init every weight to init_std
            for linear in (self.wo, getattr(self.norm_wqkv, "1")):
                nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)

        

def swap_norm_ffn_to_te_friendly_norm_ffn(parent_module) -> None:
    """
    `parent_module` is a module with the following structure:

      parent_module
        ffn_norm: LayerNorm|RMSNorm
        ffn: FeedForward
          w1
          w2
          w3

    this function will rewrite the graph without changing numerics to the following structure

      parent_module
        ffn_norm: ResettableIdentity
        feed_forward: NormFeedForward
          norm_w13: Sequential
            0: LayerNorm|RMSNorm
            1: Linear (fused w1 and w3)
          w2: Linear

    this is done to then make it easier to then swap to te.LayerNormLinear
    """
    if hasattr(parent_module, "ffn_norm") and hasattr(parent_module, "feed_forward"):
        parent_module.feed_forward = NormFeedForward(
            parent_module.ffn_norm,
            parent_module.feed_forward,
        )
        parent_module.ffn_norm = ResettableIdentity()
    else:
        for name, child in parent_module.named_children():
            swap_norm_ffn_to_te_friendly_norm_ffn(child)

def swap_norm_attn_to_te_friendly_norm_attn(parent_module):
    """
    `parent_module` is a module with the following structure:

      parent_module
        attention_norm: LayerNorm|RMSNorm
        attention: Attention
          wq
          wk
          wv
          wo

    this function will rewrite the graph without changing numerics to the following structure

      parent_module
        attention_norm: ResettableIdentity
        attention: NormAttention
          norm_wqkv: Sequential
            0: LayerNorm|RMSNorm
            1: Linear (fused wq, wk, wv)
          wo: Linear

    this is done to then make it easier to then swap to te.LayerNormLinear
    """
    if hasattr(parent_module, "attention_norm") and hasattr(parent_module, "attention"):
        parent_module.attention = NormAttention(
            parent_module.attention_norm,
            parent_module.attention,
        )
        parent_module.attention_norm = ResettableIdentity()
    else:
        for name, child in parent_module.named_children():
            swap_norm_attn_to_te_friendly_norm_attn(child)

def swap_te_friendly_norm_ffn_to_te_layernorm_linear(parent_module):
    """
    In `NormFeedForward`, swaps `norm_w13` with `te.LayerNormLinear`
    In `NormAttention`, swaps `norm_wqkv` with `te.LayerNormLinear`
    """

    if isinstance(parent_module, NormFeedForward):

        te_ln_linear = te.LayerNormLinear(
            parent_module.norm_w13[1].in_features,
            parent_module.norm_w13[1].out_features,
            bias=False,
            normalization='RMSNorm',
        )

        te_ln_linear.layer_norm_weight = parent_module.norm_w13[0].weight
        te_ln_linear.weight = parent_module.norm_w13[1].weight
        parent_module.norm_w13 = te_ln_linear

    elif isinstance(parent_module, NormAttention):
        
        te_ln_linear = te.LayerNormLinear(
            parent_module.norm_wqkv[1].in_features,
            parent_module.norm_wqkv[1].out_features,
            bias=False,
            normalization='RMSNorm',
        )
        te_ln_linear.layer_norm_weight = parent_module.norm_wqkv[0].weight
        te_ln_linear.weight = parent_module.norm_wqkv[1].weight
        parent_module.norm_wqkv = te_ln_linear

    else:
        for name, child in parent_module.named_children():
            swap_te_friendly_norm_ffn_to_te_layernorm_linear(child)

def _monkey_patched_te_layernorm_mlp_init_weights(self):
    torch.nn.init.ones_(self.layer_norm_weight)
    torch.nn.init.trunc_normal_(self.fc1_weight, mean=0.0, std=init_std)
    torch.nn.init.trunc_normal_(self.fc2_weight, mean=0.0, std=init_std)

def swap_te_friendly_norm_ffn_to_te_layernorm_mlp(parent_module):
    """
    Swaps `NormFeedForward` with `te.LayerNormMLP`
    """

    for name, child in parent_module.named_children():
        if isinstance(child, NormFeedForward):
            te_ln_mlp = te.LayerNormMLP(
                child.norm_w13[1].in_features,
                child.norm_w13[1].out_features,
                bias=False,
                normalization='RMSNorm',
                activation='swiglu',
            )
            te_ln_mlp.layer_norm_weight = child.norm_w13[0].weight
            te_ln_mlp.fc1_weight = child.norm_w13[1].weight
            te_ln_mlp.fc2_weight = child.w2.weight
            setattr(parent_module, name, te_ln_mlp)

        else:
            swap_te_friendly_norm_ffn_to_te_layernorm_mlp(child)


def get_maybe_fp8_autocast(job_config):
    # not for land - set up TransformerEngine fp8 autocast
    # Note: te.fp8_autocast has to be created at every training iteration.
    # If we try to create it once and reuse, we get this error:
    # https://gist.github.com/vkuzo/d9840328c8bdc2901b8d04aa570ecb5b
    maybe_te_float8_ctx = contextlib.nullcontext()
    if job_config.training.te_float8_autocast:
        assert (
            job_config.training.te_swap_linear 
            or job_config.training.te_swap_ln_linear 
            or job_config.training.te_swap_ln_mlp
        )
        maybe_te_float8_ctx = te.fp8_autocast(enabled=True, fp8_recipe=te_fp8_recipe)
    return maybe_te_float8_ctx
