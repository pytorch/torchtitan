# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen MRoPE with the Flash-Next reference's input-dtype rotation."""

from dataclasses import dataclass

import torch

from torchtitan.models.qwen3_5.rope import MRoPE as Qwen35MRoPE

# T = tokens, H = heads, R = rotary features.


class MRoPE(Qwen35MRoPE):
    """Reuse Qwen3.5's position selection; round cos/sin and rotation to BF16."""

    @dataclass(kw_only=True, slots=True)
    class Config(Qwen35MRoPE.Config):
        pass

    @staticmethod
    def apply_rotary_emb(
        query: torch.Tensor,
        key: torch.Tensor | None,
        rope_cache: torch.Tensor,
        *,
        inverse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if inverse:
            raise NotImplementedError("MRoPE does not support inverse rotation")
        rotary_dim = query.shape[-1]
        cos_T1R = rope_cache[..., :rotary_dim].to(query.dtype)
        sin_T1R = rope_cache[..., rotary_dim:].to(query.dtype)

        def rotate(x_THR: torch.Tensor) -> torch.Tensor:
            first_THR, second_THR = x_THR.chunk(2, -1)
            return x_THR * cos_T1R + torch.cat((-second_THR, first_THR), -1) * sin_T1R

        query_THR = rotate(query)
        return query_THR if key is None else (query_THR, rotate(key))
