# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch.optim

from torchtitan.optimizer import get_optimizer


def test__get_optimizer__adam_optimizers_registered():
    assert get_optimizer("Adam") is torch.optim.Adam
    assert get_optimizer("AdamW") is torch.optim.AdamW


def test__get_optimizer__raise_error_if_unknown():
    with pytest.raises(RuntimeError):
        get_optimizer("MyOwnOptimizer")
