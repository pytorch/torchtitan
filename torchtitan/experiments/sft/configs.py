# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from torchtitan.experiments.sft.dataset import SFTDataLoader
from torchtitan.trainer import Trainer


@dataclass(kw_only=True, slots=True)
class SFTTrainerConfig(Trainer.Config):
    dataloader: SFTDataLoader.Config = field(default_factory=SFTDataLoader.Config)

    def __post_init__(self):
        Trainer.Config.__post_init__(self)
        if self.dataloader.pack_sequences:
            attn = self.model_spec.model.layer.attention
            if attn.attn_backend not in ("flex", "varlen"):
                raise ValueError(
                    f"pack_sequences=True requires 'flex' or 'varlen' attention "
                    f"to prevent cross-document attention in packed sequences, "
                    f"but got attn_backend='{attn.attn_backend}'. Either set "
                    f"pack_sequences=False or switch to flex/varlen attention."
                )
            if attn.attn_mask_type != "block_causal":
                raise ValueError(
                    f"pack_sequences=True requires attn_mask_type='block_causal' "
                    f"to prevent cross-document attention in packed sequences, "
                    f"but got attn_mask_type='{attn.attn_mask_type}'. Set "
                    f"model_spec.model.layer.attention.attn_mask_type='block_causal'."
                )

        if (
            self.validator.enable
            and self.validator.steps == -1
            and self.validator.dataloader.pack_sequences
        ):
            raise ValueError(
                "validator.steps=-1 with pack_sequences=True can cause hangs "
                "during validation. Packed sequences may produce different "
                "batch counts across DP ranks, causing all-reduce to deadlock. "
                "Set validator.steps to an explicit positive integer instead."
            )
