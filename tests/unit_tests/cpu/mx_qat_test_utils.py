# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Small checkpoint metadata fixtures; tensor payloads are deliberately absent."""

import json
from pathlib import Path

from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.kimi_k3.quantization import MXFP4_QUANTIZATION_CONFIG


def write_mixed_checkpoint_metadata(path: Path, adapter) -> dict[str, str]:
    """Mirror release storage: packed expert matrices, ordinary dense weights."""
    groups = {
        fqn for fqn, _, _, _ in adapter.kimi_config.traverse(GroupedExperts.Config)
    }
    weight_map = {}
    for hf_name, target in adapter.hf_linear_weight_mapping().items():
        if target is not None and target.rsplit(".", 1)[0] in groups:
            prefix = hf_name.removesuffix(".weight")
            weight_map[prefix + ".weight_packed"] = "weights.safetensors"
            weight_map[prefix + ".weight_scale"] = "scales.safetensors"
        else:
            weight_map[hf_name] = "dense.safetensors"
    (path / "config.json").write_text(
        json.dumps({"text_config": {"quantization_config": MXFP4_QUANTIZATION_CONFIG}})
    )
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    return weight_map
