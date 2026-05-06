# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.observability import structured_logger as sl
from torchtitan.trainer import Trainer


class SFTTrainer(Trainer):
    """Trainer subclass that adds block-causal 4D attention mask support for
    HuggingFace models used in SFT with packed sequences.

    For non-SFT configs (attn_mask_type != "block_causal"), this behaves
    identically to the base Trainer.
    """

    @sl.log_trace_span("post_dataloading_process")
    def post_dataloading_process(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        inputs = input_dict["input"]
        extra_inputs = {k: v for k, v in input_dict.items() if k != "input"}
        extra_kwargs: dict[str, Any] = {}

        positions = extra_inputs.pop("positions", None)

        attn_mask_type = getattr(self.model_config, "attn_mask_type", "causal")

        if attn_mask_type == "block_causal":
            assert (
                positions is not None
            ), "block_causal mask requires per-document positions from the dataloader"
            extra_kwargs["positions"] = positions
            # Cast to FSDP compute dtype so the mask matches query dtype in SDPA
            compute_dtype = TORCH_DTYPE_MAP[
                self.config.training.mixed_precision_param
            ]
            extra_kwargs["attention_masks"] = _build_block_causal_4d_mask(
                inputs, positions
            ).to(dtype=compute_dtype)
        else:
            extra_kwargs["positions"] = positions

        self.ntokens_seen += labels.numel()

        return inputs, labels, extra_inputs, extra_kwargs


def _build_block_causal_4d_mask(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Build a 4D block-causal attention mask from per-document positions.

    Document boundaries are detected where positions reset (decrease).
    Within each document: causal (lower-triangular).
    Across documents: no attention (masked out).

    Returns a [batch, 1, seq_len, seq_len] float mask (0.0 = attend, -inf = masked).
    """
    seq_len = input_ids.shape[1]

    # Detect document boundaries: positions decrease at each new document
    doc_boundaries = (positions[:, 1:] <= positions[:, :-1]).int()
    document_ids = torch.zeros_like(input_ids, dtype=torch.int32)
    document_ids[:, 1:] = doc_boundaries.cumsum(dim=1)

    # same_doc[b, i, j] = True iff tokens i and j are in the same document
    same_doc = document_ids.unsqueeze(2) == document_ids.unsqueeze(1)  # [B, S, S]

    # Causal: token i can attend to token j only if i >= j
    causal = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device)
    )

    mask = same_doc & causal.unsqueeze(0)  # [B, S, S]

    # HF expects float mask: 0.0 for attend, -inf for masked
    attn_mask = torch.where(mask, 0.0, float("-inf"))
    return attn_mask.unsqueeze(1)  # [B, 1, S, S]
