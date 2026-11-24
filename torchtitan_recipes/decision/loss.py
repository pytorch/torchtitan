# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Answer-position projection shared by the decision recipes."""

from dataclasses import dataclass
from typing import Any

import torch

from torchtitan.components.loss import ChunkedLossWrapper, IGNORE_INDEX


class AnswerTokenLoss(ChunkedLossWrapper):
    """Apply lm_head and the loss only at supervised positions.

    Gathering a fixed number of positions keeps shapes static for CUDA graphs
    and skips lm_head and cross-entropy over the ~99% of positions whose labels
    are ``IGNORE_INDEX``. ``DecisionLetterProcessor`` supervises one token per
    row and packing caps documents per microbatch at ``max_num_documents``, so
    setting ``max_num_supervised = max_num_documents`` is enough. An on-device
    assert catches a violation instead of silently dropping supervised tokens.

    The gathered rows go through ``ChunkedLossWrapper.__call__``, which applies
    lm_head and the loss with its FSDP and loss-parallel handling.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ChunkedLossWrapper.Config):
        max_num_supervised: int
        """Upper bound on supervised positions per microbatch."""
        num_chunks: int = 1
        """The gathered rows are few, so they are not chunked further."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.max_num_supervised = config.max_num_supervised

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_loss_token_counts: torch.Tensor | None = None,
        **loss_inputs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        is_supervised_T = labels != IGNORE_INDEX
        torch._assert_async(
            is_supervised_T.sum() <= self.max_num_supervised,
            "more supervised positions than max_num_supervised",
        )
        # Unused slots hold -1: they gather the last row and get ignored labels.
        index_S = torch.nonzero_static(
            is_supervised_T, size=self.max_num_supervised, fill_value=-1
        ).squeeze(-1)
        labels_S = labels[index_S].masked_fill(index_S < 0, IGNORE_INDEX)
        return super().__call__(
            pred[index_S], labels_S, global_loss_token_counts, **loss_inputs
        )
