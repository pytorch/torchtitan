# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn.attention.flex_attention import and_masks

from torchtitan.models.common.attention import (
    create_attention_mask,
    get_causal_mask_mod,
    get_document_mask_mod,
)
from torchtitan.observability import structured_logger as sl
from torchtitan.tools.logging import logger
from torchtitan.trainer import Trainer


class SFTTrainer(Trainer):
    """Trainer subclass that builds a BlockMask for HF models in SFT.

    Builds the BlockMask in post_dataloading_process (before any parallelism
    sharding) so it works correctly with TP + sequence parallel.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        hf_model: str = ""
        """HuggingFace model ID (e.g. 'Qwen/Qwen2.5-7B')."""

    @sl.log_trace_span("post_dataloading_process")
    def post_dataloading_process(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        inputs = input_dict["input"]
        # Everything else becomes a model-forward kwarg, forwarded to all PP
        # stages (matches the base Trainer contract). positions is read here so
        # we can build the block-causal mask.
        extra_kwargs: dict[str, Any] = {
            k: v for k, v in input_dict.items() if k != "input"
        }

        positions = extra_kwargs.get("positions", None)

        attn_mask_type = getattr(self.model_config, "attn_mask_type", "causal")

        if attn_mask_type == "block_causal":
            assert (
                positions is not None
            ), "block_causal mask requires per-document positions from the dataloader"
            mask_mod = and_masks(
                get_causal_mask_mod(), get_document_mask_mod(positions)
            )
            B, seq_len = positions.shape
            extra_kwargs["attention_masks"] = create_attention_mask(
                mask_mod, B, None, seq_len, seq_len
            )

        if self.parallel_dims.cp_enabled:
            from torchtitan.distributed.context_parallel import (
                prepare_context_parallel_input,
            )

            load_balancer = self.config.parallelism.context_parallel_load_balancer
            if attn_mask_type == "block_causal" and load_balancer == "headtail":
                # "headtail" is the SDPA default and cannot shard a FlexAttention
                # BlockMask; switch to the flex-compatible "ptrr". An explicit
                # None (disable balancing) or "ptrr" is respected.
                logger.warning(
                    "context_parallel_load_balancer='headtail' is SDPA-only and "
                    "cannot shard a block_causal BlockMask; using 'ptrr'. Set it "
                    "explicitly (ptrr or None) to silence this."
                )
                load_balancer = "ptrr"
            inputs, labels, extra_kwargs = prepare_context_parallel_input(
                inputs,
                labels,
                extra_kwargs,
                self.parallel_dims.get_mesh("cp"),
                self.device,
                load_balancer,
            )

        self.ntokens_seen += labels.numel()

        return inputs, labels, extra_kwargs
