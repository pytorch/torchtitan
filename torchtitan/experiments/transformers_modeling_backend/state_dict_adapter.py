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

    Handles weight tying: when tie_word_embeddings is True, some HF checkpoints
    omit lm_head.weight from safetensors (it shares storage with embed_tokens).
    """

    def __init__(
        self,
        model_config: HFTransformerModel.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)
        self._tie_word_embeddings = getattr(model_config, "tie_word_embeddings", False)

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        hf_state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
        # When weights are tied, lm_head.weight may not exist in the
        # safetensors file. Exclude it so DCP doesn't fail on missing key.
        if (
            self._tie_word_embeddings
            and "lm_head.weight" in hf_state_dict
            and "model.embed_tokens.weight" in hf_state_dict
        ):
            del hf_state_dict["lm_head.weight"]
        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        # Reconstruct lm_head.weight from embed_tokens if it was excluded
        if (
            "lm_head.weight" not in hf_state_dict
            and "model.embed_tokens.weight" in hf_state_dict
        ):
            hf_state_dict["lm_head.weight"] = hf_state_dict["model.embed_tokens.weight"]
        return {"model." + k: v for k, v in hf_state_dict.items()}
