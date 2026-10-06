# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Liger fused linear cross-entropy overrides.

This override uses
``liger_kernel.transformers.functional.liger_fused_linear_cross_entropy``,
backed by Liger's ``LigerFusedLinearCrossEntropyFunction``. The kernel computes
the LM-head projection and summed cross-entropy together, chunking internally
without materializing logits for the full local token sequence.
``ChunkedLossWrapper.Config.num_chunks`` is therefore ignored; use the head
override's ``chunk_mem_const`` argument to tune Liger's internal chunk budget.

Both overrides must be enabled:

    --override torchtitan_recipes.overrides.liger_fused_linear_cross_entropy.liger_fused_linear_cross_entropy_head
    --override torchtitan_recipes.overrides.liger_fused_linear_cross_entropy.liger_fused_linear_cross_entropy_loss

The selected recipe must also set ``training.disable_cuda_graphs = True``.
This initial integration supports the standard Trainer with TP=1 and an exact
stock ``Linear.Config`` LM head. The fused loss itself must remain eager, though
other local compile regions may stay enabled. It does not support whole-step
``torch.compile``, GraphTrainer, multi-output losses, or transformed LM heads
such as quantized and LoRA linears. The public API used here is available in
``liger-kernel>=0.8.4``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import spmd_types as spmd
import torch
from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy

from torchtitan.components.loss import (
    ChunkedLossWrapper,
    CrossEntropyLoss,
    IGNORE_INDEX,
)
from torchtitan.config import derive, override
from torchtitan.distributed.spmd_types import current_spmd_mesh, spmd_mesh_size
from torchtitan.models.common.linear import Linear


__all__ = [
    "LigerFusedLinearCrossEntropyHead",
    "LigerFusedLinearCrossEntropyLoss",
    "liger_fused_linear_cross_entropy_head",
    "liger_fused_linear_cross_entropy_loss",
]


class LigerFusedLinearCrossEntropyHead(Linear):
    """Dense LM head with an opt-in Liger fused loss forward."""

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        chunk_mem_const: int = 1
        """Liger transient-logits budget multiplier."""

        def __post_init__(self) -> None:
            if self.chunk_mem_const < 1:
                raise ValueError("chunk_mem_const must be positive")

    def __init__(self, config: Config):
        super().__init__(config)
        self.chunk_mem_const = config.chunk_mem_const

    def forward(  # pyrefly: ignore [bad-override]
        self,
        input: torch.Tensor,
        *,
        target: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor]:
        if target is None:
            return super().forward(input)

        if spmd_mesh_size("tp") > 1:
            raise ValueError(
                "The Liger fused linear cross-entropy override requires TP=1."
            )
        if torch.compiler.is_compiling():
            raise RuntimeError(
                "The Liger fused linear cross-entropy loss must remain outside "
                "torch.compile."
            )
        if input.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "The Liger fused linear cross-entropy override does not support "
                "CUDA graph capture; set training.disable_cuda_graphs = True."
            )
        if input.ndim != 2 or target.ndim != 1 or input.shape[0] != target.shape[0]:
            raise ValueError(
                "Liger fused linear cross-entropy requires hidden states [T, D] "
                "and labels [T]."
            )

        loss = liger_fused_linear_cross_entropy(
            input,
            self.weight,
            target,
            bias=self.bias,
            ignore_index=IGNORE_INDEX,
            reduction="sum",
            chunk_mem_const=self.chunk_mem_const,
        )
        # A tuple bypasses the normal LM-head output redistribution, whose
        # sharding contract describes logits rather than a scalar partial loss.
        return (loss,)


class LigerFusedLinearCrossEntropyLoss(ChunkedLossWrapper):
    """Loss wrapper that delegates sequence chunking and CE to Liger."""

    @dataclass(kw_only=True, slots=True)
    class Config(ChunkedLossWrapper.Config):
        pass

    def __init__(self, config: Config):
        del config
        self.lm_head: torch.nn.Module | None = None

    def set_lm_head(self, lm_head: torch.nn.Module) -> None:
        if not isinstance(lm_head, LigerFusedLinearCrossEntropyHead):
            raise ValueError(
                "LigerFusedLinearCrossEntropyLoss requires the paired "
                "liger_fused_linear_cross_entropy_head override."
            )
        self.lm_head = lm_head

    def __call__(
        self,
        pred: torch.Tensor | tuple[torch.Tensor, ...],
        labels: torch.Tensor | tuple[torch.Tensor, ...],
        global_valid_tokens: torch.Tensor | None = None,
        **loss_inputs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not isinstance(pred, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise ValueError(
                "Liger fused linear cross-entropy does not support multi-output losses."
            )
        if loss_inputs:
            raise ValueError(
                "Liger fused linear cross-entropy does not support additional "
                f"loss inputs: {sorted(loss_inputs)}."
            )
        lm_head = self.lm_head
        assert isinstance(
            lm_head, LigerFusedLinearCrossEntropyHead
        ), "Set the paired Liger lm_head before calling LigerFusedLinearCrossEntropyLoss"

        with spmd.no_typecheck():
            (loss,) = lm_head(pred, target=labels)

        if current_spmd_mesh() is not None:
            if spmd.is_type_checking():
                loss = spmd.mutate_type(
                    loss,
                    src=spmd.R,
                    dst={"dp": spmd.P, "cp": spmd.P, "tp": spmd.I},
                )
            if global_valid_tokens is not None:
                spmd.assert_type(
                    global_valid_tokens,
                    {"dp": spmd.R, "cp": spmd.R, "tp": spmd.I},
                )
        if global_valid_tokens is not None:
            loss = loss / global_valid_tokens
        return loss, {}


@override(
    target=Linear.Config,
    fqns=["model.lm_head"],
    exact=True,
    description="Liger fused linear cross-entropy LM head (TP=1, eager).",
)
def liger_fused_linear_cross_entropy_head(
    cfg: Linear.Config,
    *,
    chunk_mem_const: int = 1,
) -> LigerFusedLinearCrossEntropyHead.Config:
    if cfg.num_linears != 1:
        raise ValueError(
            "The Liger fused linear cross-entropy override requires a single LM head."
        )
    return derive(
        cfg,
        LigerFusedLinearCrossEntropyHead.Config,
        chunk_mem_const=chunk_mem_const,
    )


@override(
    target=ChunkedLossWrapper.Config,
    fqns=["loss"],
    exact=True,
    description="Liger fused linear cross-entropy loss (TP=1, eager).",
)
def liger_fused_linear_cross_entropy_loss(
    cfg: ChunkedLossWrapper.Config,
) -> LigerFusedLinearCrossEntropyLoss.Config:
    if not isinstance(cfg.loss_fn, CrossEntropyLoss.Config):
        raise ValueError(
            "The Liger fused linear cross-entropy override requires "
            "ChunkedLossWrapper with CrossEntropyLoss."
        )
    return derive(cfg, LigerFusedLinearCrossEntropyLoss.Config)
