# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Released Kimi MXFP4 schema used by the debug recipe and fixture.

Source: moonshotai/Kimi-K3 config.json, revision c5d1dd4.
Import validates the checkpoint's own metadata; this is not a loader default.
"""

MXFP4_QUANTIZATION_CONFIG = {
    "format": "mxfp4-pack-quantized",
    "quant_method": "compressed-tensors",
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 4,
                "type": "float",
                "strategy": "group",
                "group_size": 32,
                "symmetric": True,
                "dynamic": False,
                "scale_dtype": "torch.uint8",
            },
        },
    },
    "ignore": [
        "re:.*self_attn.*",
        "re:.*shared_experts.*",
        r"re:.*mlp\.(gate|up|gate_up|down)_proj.*",
        "re:.*lm_head.*",
        "re:.*vision_tower.*",
        "re:.*mm_projector.*",
    ],
}
