# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Linear that runs its matmul in a fixed compute dtype, plus the converter
that swaps the decoder lm_head to it.

RL training (logprob / KL / advantage math) is sensitive to the precision of
the lm_head logits. With bf16 mixed-precision training the ``hidden @ Wᵀ``
matmul accumulates over the hidden dim in bf16, which introduces error in the
logits. ``CastedLinear`` casts the inputs and weight up to a higher precision
(fp32 by default) so the accumulation happens there, regardless of the dtype
the parameters are stored / all-gathered in.

This lives under ``experiments/rl`` because only RL configs opt into it; core
models keep the plain ``Linear`` so LoRA / quantization converters compose
with them unchanged.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.models.common.nn_modules import Linear
from torchtitan.protocols.model import ModelConfigConverter


class CastedLinear(Linear):
    """``Linear`` whose forward matmul runs in ``compute_dtype``.

    Inputs, weight, and bias are cast to ``compute_dtype`` before
    ``F.linear`` and the output is returned in that dtype. Because the cast
    happens in ``forward`` (not on the stored parameter), this is safe under
    weight tying — the shared embedding/lm_head parameter keeps its original
    dtype and only the lm_head matmul sees the cast.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        compute_dtype: str = "float32"
        """Dtype for the forward matmul (key into ``TORCH_DTYPE_MAP``)."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.compute_dtype = TORCH_DTYPE_MAP[config.compute_dtype]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bias = None if self.bias is None else self.bias.to(self.compute_dtype)
        return F.linear(
            input.to(self.compute_dtype), self.weight.to(self.compute_dtype), bias
        )


class LMHeadCastConverter(ModelConfigConverter):
    """Swap the decoder lm_head's ``Linear.Config`` to ``CastedLinear.Config``.

    Walks the model config tree and replaces the ``lm_head`` node in place.
    Targets only the lm_head, so every other Linear stays a plain ``Linear``
    and LoRA / quantization converters are unaffected.

    Note on trainer/inference bitwise parity: because the same ``model_spec``
    backs both the trainer and the vLLM generator, the lm_head sees a matched
    cast chain on both sides. The inference weight, synced from the trainer,
    goes fp32 (trainer) -> bf16 (weight-sync) -> fp32 (lm_head cast); the
    trainer's own lm_head input follows the analogous fp32 (FSDP-sharded
    params) -> bf16 (all-gather) -> fp32 (lm_head cast). Both paths share the
    same lossy bf16 round-trip, which tends to make trainer<->inference
    bitwise agreement *easier*, not harder. TODO: investigate whether this
    fp32->bf16->fp32 cast pair can be removed (keep the weight in fp32 end to
    end) once that path is supported on both sides.
    """

    _TARGET = "lm_head"

    @dataclass(kw_only=True, slots=True)
    class Config(ModelConfigConverter.Config):
        compute_dtype: str = "float32"
        """Forward matmul dtype for the lm_head (key into ``TORCH_DTYPE_MAP``)."""

    def __init__(self, config: Config):
        self.config = config

    def convert(self, model_config) -> None:
        for fqn, linear_config, parent, attr in model_config.traverse(Linear.Config):
            if fqn.rsplit(".", 1)[-1] != self._TARGET:
                continue
            new_config = CastedLinear.Config(
                in_features=linear_config.in_features,
                out_features=linear_config.out_features,
                bias=linear_config.bias,
                param_init=linear_config.param_init,
                sharding_config=linear_config.sharding_config,
                compute_dtype=self.config.compute_dtype,
            )
            if isinstance(parent, list):
                parent[attr] = new_config
            else:
                setattr(parent, attr, new_config)
