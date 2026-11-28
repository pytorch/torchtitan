# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.distributed.flex_shard.dist_muon import (
    _valid_ns_coefficients,
    _zeropower_via_newtonschulz,
)


@pytest.mark.filterwarnings("ignore:mkldnn_matmul failed.*")
def test_dist_muon_applies_shared_or_per_step_ns_coefficients() -> None:
    update = torch.arange(1, 13, dtype=torch.float32).reshape(4, 3)
    shared = (3.4445, -4.7750, 2.0315)
    repeated = (shared, shared)
    per_step = (shared, (3.0, -4.0, 2.0))

    shared_result = _zeropower_via_newtonschulz(
        update,
        ns_coefficients=shared,
        ns_steps=2,
        eps=1e-7,
    )
    repeated_result = _zeropower_via_newtonschulz(
        update,
        ns_coefficients=repeated,
        ns_steps=2,
        eps=1e-7,
    )
    per_step_result = _zeropower_via_newtonschulz(
        update,
        ns_coefficients=per_step,
        ns_steps=2,
        eps=1e-7,
    )

    assert torch.equal(shared_result, repeated_result)
    assert not torch.equal(shared_result, per_step_result)
    assert _valid_ns_coefficients(shared, ns_steps=2)
    assert _valid_ns_coefficients(per_step, ns_steps=2)
    assert not _valid_ns_coefficients(per_step, ns_steps=3)
