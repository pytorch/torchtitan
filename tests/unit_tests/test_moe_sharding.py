# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import spmd_types as spmd

from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_sequence_parallel_placement,
)
from torchtitan.models.common.moe_sharding import (
    _moe_sharding_config,
    _routed_experts_sharding_configs,
    _shared_experts_sharding_configs,
)


@pytest.mark.parametrize(
    ("enable_ep", "enable_sp"),
    ((False, False), (False, True), (True, False), (True, True)),
)
def test_moe_internal_output_layouts_match(enable_ep: bool, enable_sp: bool):
    if enable_sp:
        expected_output_layout = dense_sequence_parallel_placement()
    elif enable_ep:
        expected_output_layout = dense_activation_placement(tp=spmd.I)
    else:
        expected_output_layout = dense_activation_placement(tp=spmd.P)

    shared_configs = _shared_experts_sharding_configs(
        enable_ep=enable_ep,
        enable_sp=enable_sp,
    )
    routed_config, _ = _routed_experts_sharding_configs(
        enable_ep=enable_ep,
        enable_sp=enable_sp,
        expert_param_layout={"weight": spmd.S(1)},
    )
    moe_config = _moe_sharding_config(
        enable_ep=enable_ep,
        enable_sp=enable_sp,
    )

    assert shared_configs[2].out_dst_shardings == expected_output_layout
    assert routed_config.out_dst_shardings == expected_output_layout
    assert moe_config.out_src_shardings == expected_output_layout
