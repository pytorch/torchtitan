# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.kimi_k3 import MODEL_FLAVORS
from torchtitan.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter


def test_hf_load_writes_routed_experts_in_place():
    build_config, _ = MODEL_FLAVORS["debugmodel"]
    config = build_config(
        attn_backend="flex", seq_len=128, parallelism=ParallelismConfig()
    )
    model = config.build()
    model.init_states()
    state_dict = model.state_dict()
    adapter = KimiK3StateDictAdapter(config, hf_assets_path=None)
    expert_keys = [key for key in state_dict if ".routed_experts." in key]
    assert expert_keys

    # An HF load fills to_hf's tensors in place before from_hf runs.
    hf_state_dict = adapter.to_hf(state_dict)
    with torch.no_grad():
        for value in hf_state_dict.values():
            if value.is_floating_point():
                value.copy_(torch.randn_like(value))
    restored = adapter.from_hf(hf_state_dict)

    for key in expert_keys:
        # Routed experts come back as views of the parameters to_hf split,
        # which already hold the loaded values, not as a second copy.
        assert (
            restored[key].untyped_storage().data_ptr()
            == state_dict[key].untyped_storage().data_ptr()
        )
        torch.testing.assert_close(restored[key], state_dict[key], rtol=0, atol=0)
    model.load_state_dict(restored, strict=True)
