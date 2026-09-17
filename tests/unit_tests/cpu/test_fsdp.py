# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import pytest
import torch.nn as nn

from torchtitan.config import FSDPSymmMemScope
from torchtitan.distributed.fsdp import enable_fsdp_symm_mem


class _FSDPModule(nn.Module):
    def __init__(self, *, moe_enabled: bool) -> None:
        super().__init__()
        self.moe_enabled = moe_enabled
        self.force_sum_enabled = False
        self.symm_mem_enabled = False

    def set_force_sum_reduction_for_comms(self, enable: bool) -> None:
        self.force_sum_enabled = enable

    def set_symm_mem_for_comm(self) -> None:
        self.symm_mem_enabled = True


@pytest.mark.parametrize(
    ("scope", "dense_enabled", "sparse_enabled"),
    [
        (None, False, False),
        ("all", True, True),
        ("dense", True, False),
    ],
)
def test_enable_fsdp_symm_mem_scope(
    scope: FSDPSymmMemScope, dense_enabled: bool, sparse_enabled: bool
) -> None:
    model = nn.Module()
    dense = _FSDPModule(moe_enabled=False)
    sparse = _FSDPModule(moe_enabled=True)
    model.add_module("dense", dense)
    model.add_module("sparse", sparse)

    with mock.patch("torchtitan.distributed.fsdp.FSDPModule", _FSDPModule):
        enable_fsdp_symm_mem(model, scope)

    assert dense.force_sum_enabled == dense_enabled
    assert dense.symm_mem_enabled == dense_enabled
    assert sparse.force_sum_enabled == sparse_enabled
    assert sparse.symm_mem_enabled == sparse_enabled
