# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .model import HFTransformerModel


class HFTransformerStateDictAdapter(StateDictAdapter):
    """State dict adapter for HFTransformerModel.

    Since HFTransformerModel wraps an HF ForCausalLM as self.model, the only
    difference between TorchTitan FQNs and HF safetensors keys is a "model."
    prefix. No weight reshaping or renaming is needed.
    """

    def __init__(
        self,
        model_config: HFTransformerModel.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        return {k.removeprefix("model."): v for k, v in state_dict.items()}

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        return {"model." + k: v for k, v in hf_state_dict.items()}
