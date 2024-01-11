# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# copied from torchtune

from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# TokenPair is a pair (tuple) of two lists: tokenized text inputs and labels.
TokenPair = Tuple[List[int], List[int]]

_DEFAULT_INPUT_PADDING_IDX: int = 0
_DEFAULT_LABEL_PADDING_IDX: int = -100


def pad_batch_to_longest_seq(
    batch: List[TokenPair],
    input_padding_idx: int = _DEFAULT_INPUT_PADDING_IDX,
    label_padding_idx: int = _DEFAULT_LABEL_PADDING_IDX,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a batch of sequences to the longest sequence length in the batch, and
    convert integer lists to tensors.

    Args:
        batch (List[TokenPair]): A list of tuples containing input, label pairs.
        input_padding_idx (int): Padding index for input ids. Defaults to 0.
        label_padding_idx (int): Padding index for labels. Defaults to -100.
    Returns:
        Collated input and label tensors.

    Example:
        token_pairs = [
            ([1, 2, 3], [4, 5, 6]),
            ([7,], [10,],),
        ]
        inputs, labels = batch_pad_to_longest_seq(
            batch=token_pairs,
            input_padding_idx=input_padding_idx,
            label_padding_idx=label_padding_idx,
        )
        >>> inputs
            tensor([[1, 2, 3], [7, 0, 0]])
        >>> labels
            tensor([[4,5,6], [10,-100,-100]])
    """
    input_ids = pad_sequence(
        [torch.tensor(x[0]) for x in batch],
        batch_first=True,
        padding_value=input_padding_idx,
    )
    labels = pad_sequence(
        [torch.tensor(x[1]) for x in batch],
        batch_first=True,
        padding_value=label_padding_idx,
    )

    input_ids_seq_len = input_ids.shape[-1]
    labels_seq_len = labels.shape[-1]

    # Hack to pad correctly and not use max_seq_len, which is costly
    if input_ids_seq_len > labels_seq_len:
        labels = F.pad(
            labels, (0, input_ids_seq_len - labels_seq_len), value=label_padding_idx
        )
    elif labels_seq_len > input_ids_seq_len:
        input_ids = F.pad(
            input_ids,
            (0, labels_seq_len - input_ids_seq_len),
            value=input_padding_idx,
        )
    return input_ids, labels
