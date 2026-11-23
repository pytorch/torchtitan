# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in MXFP8 Muse head with FP32 logits and composed softcap/CE.

T: token rows, D: hidden features, V: vocabulary features.
"""

from dataclasses import dataclass, fields, replace
from typing import Any

import torch

from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss, LossResult
from torchtitan.config.override import override
from torchtitan.distributed.local_compile import local_compile
from torchtitan.models.common.linear import Linear
from torchtitan.models.muse_spark.model import (
    _postprocess_logits,
    MuseSparkLMHead,
    MuseSparkModel,
)
from torchtitan.quantization.mxfp8.linear import MXFP8Linear
from torchtitan.trainer import Trainer

__all__ = ["mxfp8_lm_head"]


class _MXFP8MuseSparkLMHead(MXFP8Linear):
    """Reuse native quantized FSDP/backward with Muse's FP32 logit math."""

    @dataclass(kw_only=True, slots=True)
    class Config(MXFP8Linear.Config):
        output_dtype: torch.dtype = torch.float32
        output_multiplier: float = 1.0
        output_soft_cap_temp: float | None = None
        apply_logit_postprocess: bool = True

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.output_multiplier = config.output_multiplier
        self.output_soft_cap_temp = config.output_soft_cap_temp
        self.apply_logit_postprocess = config.apply_logit_postprocess

    @local_compile("muse_spark_logit_postprocess", batch_invariant=True)
    def postprocess_logits(self, logits_TV: torch.Tensor) -> torch.Tensor:
        """Apply the same logit postprocessing as the HiMidLo head."""
        return _postprocess_logits(
            logits_TV,
            output_multiplier=self.output_multiplier,
            output_soft_cap_temp=self.output_soft_cap_temp,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Return FP32 logits, optionally leaving postprocessing to the loss."""
        logits_TV = super().forward(input)
        if self.apply_logit_postprocess:
            return self.postprocess_logits(logits_TV)
        return logits_TV


class _MuseSparkSoftcapCrossEntropy(CrossEntropyLoss):
    """Compile Muse logit postprocessing and the existing CE in one region."""

    @dataclass(kw_only=True, slots=True)
    class Config(CrossEntropyLoss.Config):
        output_multiplier: float = 1.0
        output_soft_cap_temp: float | None = None

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.output_multiplier = config.output_multiplier
        self.output_soft_cap_temp = config.output_soft_cap_temp

    @local_compile("muse_softcap_ce", batch_invariant=False)
    def __call__(
        self,
        pred_TV: torch.Tensor,
        labels_T: torch.Tensor,
        global_loss_token_counts: torch.Tensor | None = None,
        **loss_inputs: Any,
    ) -> LossResult:
        """Preserve token normalization, z-loss and vocab-parallel CE semantics."""
        logits_TV = _postprocess_logits(
            pred_TV,
            output_multiplier=self.output_multiplier,
            output_soft_cap_temp=self.output_soft_cap_temp,
        )
        return super().__call__(
            logits_TV, labels_T, global_loss_token_counts, **loss_inputs
        )


@override(target=Trainer.Config, exact=True)
def mxfp8_lm_head(
    config: Trainer.Config, *, fuse_softcap_loss: bool = True
) -> Trainer.Config:
    """Use MXFP8 head operands while keeping native FP32 GEMM outputs.

    Args:
        config: Muse training configuration with a MuseSparkLMHead.
        fuse_softcap_loss: Move the head's scale and smooth cap into the
            chunked cross-entropy region. Requires ChunkedLossWrapper with
            CrossEntropyLoss and retains its CP/TP loss contract.

    Returns:
        A configuration with an MXFP8 head and optional composed loss. The
        original configuration is not mutated; checkpoint parameter keys match.

    Raises:
        ValueError: The model/head or requested loss contract is incompatible.
    """
    model = config.model
    if not isinstance(model, MuseSparkModel.Config) or not isinstance(
        model.lm_head, MuseSparkLMHead.Config
    ):
        raise ValueError("mxfp8_lm_head requires MuseSparkModel with MuseSparkLMHead")
    head = model.lm_head
    loss = config.loss
    if fuse_softcap_loss:
        if (
            not isinstance(loss, ChunkedLossWrapper.Config)
            or type(loss.loss_fn) is not CrossEntropyLoss.Config
        ):
            raise ValueError("fuse_softcap_loss requires chunked CrossEntropyLoss")
        ce = loss.loss_fn
        loss = replace(
            loss,
            loss_fn=_MuseSparkSoftcapCrossEntropy.Config(
                global_vocab_size=ce.global_vocab_size,
                z_loss_weight=ce.z_loss_weight,
                output_multiplier=head.output_multiplier,
                output_soft_cap_temp=head.output_soft_cap_temp,
            ),
        )
    mxfp8_head = _MXFP8MuseSparkLMHead.Config(
        **{field.name: getattr(head, field.name) for field in fields(Linear.Config)},
        output_multiplier=head.output_multiplier,
        output_soft_cap_temp=head.output_soft_cap_temp,
        apply_logit_postprocess=not fuse_softcap_loss,
    )
    regions = list(model.local_compile.regions)
    if fuse_softcap_loss and "muse_softcap_ce" not in regions:
        regions.append("muse_softcap_ce")
    return replace(
        config,
        model=replace(
            model,
            lm_head=mxfp8_head,
            local_compile=replace(model.local_compile, regions=regions),
        ),
        loss=loss,
    )
