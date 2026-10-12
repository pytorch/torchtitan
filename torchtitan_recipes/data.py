# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dataset-specific defaults shared by training recipes."""

import logging

from torchtitan.observability.logging import warn_once


logger = logging.getLogger(__name__)


def c4_document_capacity(num_tokens_per_microbatch: int) -> int:
    """Estimate C4 document capacity from the per-rank token budget."""
    warn_once(
        logger,
        "C4 document capacity is an estimate. Tune dataloader.max_num_documents "
        "for your workload to balance attention overhead and packing padding.",
    )
    # Estimate 16 document slots per 8K tokens, with at least 16 for short batches.
    return max(16, (num_tokens_per_microbatch + 511) // 512)
