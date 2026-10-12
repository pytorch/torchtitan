# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dataset-specific defaults shared by training recipes."""


def c4_document_capacity(num_tokens_per_microbatch: int) -> int:
    """Size C4 metadata from the per-rank token budget, with packing headroom."""
    # C4 sample with Llama-3.1: 99% of 8K-token batches fit within 32 documents.
    # Scale capacity by 8192 / 32 = 256 tokens per slot.
    # Keep at least 32 slots: 4K-token batches can exceed 16 documents.
    return max(32, (num_tokens_per_microbatch + 255) // 256)
