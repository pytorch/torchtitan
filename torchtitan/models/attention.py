# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

import math
from typing import Callable, ClassVar

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    BlockMask,
    create_block_mask,
    flex_attention,
)
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.utils import is_flash_attn_greater_or_equal_2_10
from torchtitan.tools.utils import has_cuda_capability
import kernels

activation = kernels.get_kernel("Motif-Technologies/activation")

# FlexAttention mask type. For each mask type, we initialize it at most once per
# batch. To record what it is initialized, FLEX_ATTN_MASK_T is used as the key to
# track the initialized mask.
FLEX_ATTN_MASK_T = tuple[str, int | None]


class FlexAttention(torch.nn.Module):
    """FlexAttention module that uses torch.nn.attention.flex_attention.

    This module is a wrapper around torch.nn.attention.flex_attention. This module
    implements certain common attention types, such as causal and block_causal.

    Args:
        attn_mask_type (str): The type of attention mask. Currently, we support
            "causal" and "block_causal". "causal" means the lower triangle of the
            attention matrix is masked. "block_causal" means the attention matrix
            is divided into blocks, where block boundary is defined by EOS token,
            and the lower triangle of each block is masked.
        fixed_block_size (int | None): The block size to be used to perform attention.
            If specified, each sequence will be further divided to blocks, where each
            block has the maximum size of ``fixed_block_size``. A query will only attend
            to the keys within the same block.
    """

    # We registered flex_attention related attributes as class variables as we
    # need to amortize the cost of compilation.
    flex_attn: ClassVar[Callable] = torch.compile(
        flex_attention, mode="max-autotune-no-cudagraphs"
    )
    compiled_create_block_mask: ClassVar[Callable] = torch.compile(create_block_mask)
    used_attn_mask_types: ClassVar[set[FLEX_ATTN_MASK_T]] = set()
    # Attention mask type to the created BlockMask.
    # This allows us to keep track the created block masks for each
    # new batch. We will use this to update the block mask when a
    # new batch is created. This also allows user to create different
    # block masks for different layers.
    block_masks: ClassVar[dict[FLEX_ATTN_MASK_T, BlockMask]] = {}

    # Instance variables.
    attn_mask_type: str

    def __init__(
        self, attn_mask_type: str, fixed_block_size: int | None = None
    ) -> None:
        super().__init__()
        if attn_mask_type not in ["causal", "block_causal"]:
            raise ValueError(f"Unrecognized attn_mask_type {attn_mask_type}.")
        self.attn_mask_type = attn_mask_type
        self.fixed_block_size = fixed_block_size

        FlexAttention.used_attn_mask_types.add(self.mask_key)

    @property
    def mask_key(self) -> FLEX_ATTN_MASK_T:
        return (self.attn_mask_type, self.fixed_block_size)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float | None = None,
    ) -> torch.Tensor:
        block_mask = FlexAttention.block_masks[self.mask_key]
        return FlexAttention.flex_attn(q, k, v, block_mask=block_mask, scale=scale)

    @staticmethod
    def _get_causal_mask_mod() -> _mask_mod_signature:
        def causal_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return q_idx >= kv_idx

        return causal_mask

    @staticmethod
    def _get_block_causal_mask_mod(
        batch: torch.Tensor, eos_id: int
    ) -> _mask_mod_signature:
        # batch is [b, s, h, d] shape
        mask = batch == eos_id
        mask[:, -1] = True
        acc_mask = torch.cumsum(torch.where(mask, 1, 0), dim=1)
        seq_idx = torch.zeros_like(acc_mask, dtype=torch.int32)
        seq_idx[:, 1:] = acc_mask[:, :-1]

        def block_causal_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (seq_idx[b, q_idx] == seq_idx[b, kv_idx]) & (q_idx >= kv_idx)

        return block_causal_mask

    @staticmethod
    def _fixed_block_mask_mod(
        mask_mod: _mask_mod_signature, fixed_block_size: int
    ) -> _mask_mod_signature:
        """
        Given an arbirary mask_mod, divide the input sequence to blocks
        and only allow attention within the same block.

        Args:
            mask_mod: The mask mod to apply to the documents
            fixed_block_size: The number of tokens in each block.
        """

        # Credit to @drisspg.
        def blocked_mask_mod(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            # Get the block index of the query and key
            q_block = q_idx // fixed_block_size
            kv_block = kv_idx // fixed_block_size
            # Only allow attention within the same block
            same_block = q_block == kv_block
            # Apply the original mask mod
            inner_mask = mask_mod(
                b, h, q_idx % fixed_block_size, kv_idx % fixed_block_size
            )

            return same_block & inner_mask

        blocked_mask_mod.__name__ = (
            f"blocked_mask_mod_{mask_mod.__name__}_fixed_block_size_{fixed_block_size}"
        )

        return blocked_mask_mod

    @staticmethod
    @torch.no_grad()
    def init_attention_mask(batch: torch.Tensor, eos_id: int | None) -> None:
        # batch is [b, s, h, d] shape
        for mask_key in FlexAttention.used_attn_mask_types:
            attn_mask_type, fixed_block_size = mask_key
            match attn_mask_type:
                case "causal":
                    if FlexAttention.block_masks.get(mask_key, None) is not None:
                        continue
                    # We don't care about batch dimension --
                    # all samples have the same lower triangle mask.
                    batch_dimension = 1
                    mask_mod = FlexAttention._get_causal_mask_mod()
                case "block_causal":
                    if eos_id is None:
                        raise RuntimeError(
                            "eos_id must be provided for block_causal mask."
                        )
                    batch_dimension = batch.shape[0]
                    mask_mod = FlexAttention._get_block_causal_mask_mod(batch, eos_id)
                case _:
                    raise RuntimeError(f"Shouldn't reach here. {attn_mask_type}")

            if fixed_block_size is not None and fixed_block_size > 0:
                mask_mod = FlexAttention._fixed_block_mask_mod(
                    mask_mod, fixed_block_size
                )

            seq_len = batch.shape[1]
            block_mask = FlexAttention.compiled_create_block_mask(
                mask_mod, batch_dimension, None, seq_len, seq_len
            )
            FlexAttention.block_masks[mask_key] = block_mask


class DiffAttention(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        layer_idx: int,
    ) -> None:
        super().__init__()

        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.dropout_rate = 0.0
        for name in ["lambda_q1", "lambda_k1", "lambda_q2", "lambda_k2"]:
            setattr(
                self,
                name,
                torch.nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32)),
            )
            getattr(self, name).data.normal_(mean=0.0, std=0.1)

        self.subln = activation.layers.RMSNorm(2 * self.head_dim, eps=1e-5)
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * (layer_idx - 1))
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
        self.is_causal = True

    def post_attn(self, attn1, attn2, bsz, q_seq_len):

        attn1 = attn1.view(bsz, q_seq_len, -1, 2, self.head_dim)
        attn2 = attn2.view(bsz, q_seq_len, -1, 2, self.head_dim)

        # q1 k1 v1, q2 k2 v1
        attn1_1, attn2_1 = attn1[..., 0, :], attn1[..., 1, :]
        # q1 k1 v2, q2 k2 v2
        attn1_2, attn2_2 = attn2[..., 0, :], attn2[..., 1, :]

        attn_1 = torch.cat([attn1_1, attn1_2], dim=-1)
        attn_2 = torch.cat([attn2_1, attn2_2], dim=-1)

        lambda_1 = torch.exp(
            torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        ).type_as(attn_1)
        lambda_2 = torch.exp(
            torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        ).type_as(attn_2)

        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        attn_output = (
            attn_1
            - lambda_full.unsqueeze(0).expand([bsz, 1]).view([bsz, 1, 1, 1]) * attn_2
        )

        return attn_output

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, q_seq_len, _, _ = q.shape
        kv_seq_len = k.shape[1]

        q = q.view(bsz, q_seq_len, self.n_heads, self.head_dim)
        k = k.view(bsz, kv_seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(bsz, kv_seq_len, self.n_kv_heads // 2, 2, self.head_dim)

        v1, v2 = v[..., 0, :].squeeze(-2), v[..., 1, :].squeeze(-2)

        v1 = torch.repeat_interleave(v1, dim=2, repeats=2)
        v2 = torch.repeat_interleave(v2, dim=2, repeats=2)

        def flash_attn_forward(q, k, v):
            return _flash_attention_forward(
                q,
                k,
                v,
                attention_mask,
                q_seq_len,
                position_ids=position_ids,
                dropout=self.dropout_rate,
                sliding_window=None,
                is_causal=self.is_causal,
                use_top_left_mask=self._flash_attn_uses_top_left_mask,
                attn_implementation="flash_attention_3",
            )

        attn1, attn2 = (
            flash_attn_forward(
                q,
                k,
                v1,
            ),
            flash_attn_forward(
                q,
                k,
                v2,
            ),
        )

        attn_output = self.post_attn(attn1, attn2, bsz, q_seq_len)
        attn_output = self.subln(attn_output)
        attn_output = attn_output * (1 - self.lambda_init)
        return attn_output

        
class ScaledDotProductAttention(torch.nn.Module):
    backends: ClassVar[list[SDPBackend]] = []

    def __init__(self, attn_mask_type: str) -> None:
        super().__init__()
        if attn_mask_type != "causal":
            raise ValueError(
                "TorchTitan with SDPA currently only supports causal mask."
            )

        ScaledDotProductAttention._init_backend()

    @classmethod
    def _init_backend(cls) -> None:
        if cls.backends:
            return

        # Add CuDNN on B200 w/ highest priority
        cls.backends = [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]
        if has_cuda_capability(10, 0):
            cls.backends.insert(0, SDPBackend.CUDNN_ATTENTION)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float | None = None,
    ) -> torch.Tensor:
        assert self.backends, "SDPA Backends should not be empty."
        with sdpa_kernel(self.backends, set_priority=True):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)


def build_attention(
    use_flex_attn: bool, attn_mask_type: str, fixed_block_size: int | None = None
):
    if use_flex_attn:
        return FlexAttention(attn_mask_type, fixed_block_size)
    else:
        if fixed_block_size is not None:
            raise ValueError(
                "TorchTitan with SDPA currently does not support fixed_block_size."
            )
        if attn_mask_type != "causal":
            raise ValueError(
                "TorchTitan with SDPA currently only supports causal mask."
            )
        return ScaledDotProductAttention(attn_mask_type)


def init_attention_mask(batch: torch.Tensor, eos_id: int | None) -> None:
    FlexAttention.init_attention_mask(batch, eos_id)
