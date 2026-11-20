# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch


def test_hf_indexer_comparison_ignores_order_and_causal_fillers():
    pytest.importorskip("transformers")
    from scripts.checkpoint_conversion.numerical_tests_glm5 import (
        compare_indexer_choices,
    )

    hf = {0: torch.tensor([[[0, 2], [1, 0], [2, 0]]])}
    native = {0: torch.tensor([[0, -1], [0, 1], [0, 2]])}
    compare_indexer_choices(hf, native, seq_len=3)

    # A changed causal key must not be hidden by ordering/filler normalization.
    native[0][-1] = torch.tensor([1, 2])
    with pytest.raises(AssertionError):
        compare_indexer_choices(hf, native, seq_len=3)


def test_hf_indexer_comparison_rejects_missing_producers():
    pytest.importorskip("transformers")
    from scripts.checkpoint_conversion.numerical_tests_glm5 import (
        compare_indexer_choices,
    )

    with pytest.raises(ValueError, match="producer layers"):
        compare_indexer_choices({}, {}, seq_len=3)
